"""표본외 나우캐스트 (config.PREREG_NOWCAST_OOS): "실제로 얼마나 맞히나".

흐름 (미래 정보 차단이 핵심):
  1) 학습 기간의 '실제 분기 날씨 변화'와 물량 성장률로 계수 추정 (분기더미 + 날씨, 가스는 지역FE 추가)
  2) 검증 기간 각 분기, 분기 말 30일 전 시점의 날씨 짐작 = 어제까지 관측 + 그날 06시 중기예보(편향은 그해 이전 오차로 보정) + 평년
  3) 짐작한 날씨 변화(짐작 − 작년 같은 분기 실제)를 계수에 넣어 성장률 예상
  4) 같은 모형에서 날씨만 뺀 예상(분기별 평균 성장률)과 오차 비교. 실제 날씨를 알았을 때(상한)도 함께 보고
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from swi import forecast_value as fv


def daily_degree_days(w: pd.DataFrame, base: float = 18.0) -> dict:
    """{stn: DataFrame(date → HDD, CDD, tavg)}"""
    w = w.copy()
    w["date"] = pd.to_datetime(w["date"])
    out = {}
    for stn, g in w.groupby(w["stn"].astype(int)):
        s = g.set_index("date")["tavg"].sort_index()
        s = s[~s.index.duplicated()]
        out[int(stn)] = pd.DataFrame({"HDD": (base - s).clip(lower=0).where(s.notna()),
                                      "CDD": (s - base).clip(lower=0).where(s.notna())})
    return out


def forecast_dd_by_asof(fc: pd.DataFrame, w: pd.DataFrame, stn: int, base: float = 18.0, issue_hour: str = "06") -> dict:
    """{as-of 날짜: DataFrame(대상날짜 → HDD, CDD)} — 그날 06시 발표, 리드별 편향은 그해 이전 발표분 오차로만 보정."""
    f = fc.copy()
    f["stn"] = f["stn"].astype(int)
    f["issued_at"] = f["issued_at"].astype(str)
    f = f[f["issued_at"].str[8:10] == issue_hour]
    bias = fv.lead_bias_by_year(f, w.assign(stn=w["stn"].astype(int)))
    f = f[f["stn"] == stn].copy()
    f["asof"] = pd.to_datetime(f["issued_at"].str[:8])
    f["date"] = pd.to_datetime(f["target_date"])
    f["b"] = [bias.get(int(y), pd.Series(dtype=float)).get(int(l), 0.0) for y, l in zip(f["asof"].dt.year, f["lead_day"])]
    tav = (f["tmin"] + f["tmax"]) / 2 - f["b"]
    f["HDD"], f["CDD"] = (base - tav).clip(lower=0), (tav - base).clip(lower=0)
    return {d: g.set_index("date")[["HDD", "CDD"]] for d, g in f.groupby("asof")}


def quarter_nowcast(dd: pd.DataFrame, q: pd.Period, fc_by_asof: dict, days_before_end: int) -> dict | None:
    """한 지점·한 분기: 실제 합계와 '분기 말 N일 전' 짐작 합계 (HDD, CDD). 관측 결측일은 평년으로 채움(실제·짐작 동일 처리)."""
    days = pd.date_range(q.start_time, q.end_time.normalize())
    asof = days[-1] - pd.Timedelta(days=days_before_end)
    out = {}
    for v in ("HDD", "CDD"):
        clim = fv.climatology_hdd(dd[v], q.year, days)
        if clim.isna().any():
            return None
        obs = dd[v].reindex(days).fillna(clim)
        real = float(obs.sum())
        est = obs[days < asof].sum() + clim[days >= asof].sum()
        f = fc_by_asof.get(asof)
        if f is not None:
            f = f[(f.index >= asof) & (f.index <= days[-1])][v].dropna()
            est = est - clim.reindex(f.index).sum() + f.sum()
        out[v] = real
        out[f"{v}_hat"] = float(est)
    out["asof"] = asof
    return out


def weather_table(w: pd.DataFrame, fc: pd.DataFrame | None, stations: list[int], years: tuple[int, int],
                  days_before_end: int, national: list[int]) -> pd.DataFrame:
    """지점(+NATIONAL)×분기: HDD, CDD 실제와 짐작, 그리고 작년 같은 분기 실제."""
    dd = daily_degree_days(w)
    rows = []
    for stn in stations:
        fcd = forecast_dd_by_asof(fc, w, stn) if fc is not None else {}
        for q in pd.period_range(f"{years[0]}Q1", f"{years[1]}Q4", freq="Q"):
            r = quarter_nowcast(dd[stn], q, fcd, days_before_end)
            if r:
                rows.append(dict(stn=str(stn), year=q.year, q=q.quarter, **r))
    t = pd.DataFrame(rows)
    nat = t[t["stn"].isin([str(s) for s in national])].groupby(["year", "q"])[["HDD", "CDD", "HDD_hat", "CDD_hat"]].mean()
    nat = nat.reset_index().assign(stn="NATIONAL")
    t = pd.concat([t, nat], ignore_index=True)
    ly = t[["stn", "year", "q", "HDD", "CDD"]].assign(year=t["year"] + 1).rename(columns={"HDD": "HDD_ly", "CDD": "CDD_ly"})
    return t.merge(ly, on=["stn", "year", "q"], how="left")


def _X(d: pd.DataFrame, xs: list[str], fe: bool, cols=None) -> pd.DataFrame:
    X = pd.get_dummies(d["q"].astype(int), prefix="q").astype(float)
    if fe:
        X = pd.concat([X, pd.get_dummies(d["unit"], prefix="u", drop_first=True).astype(float)], axis=1)
    for x in xs:
        X[x] = d[x].to_numpy(float)
    if cols is not None:
        X = X.reindex(columns=cols, fill_value=0.0)
    return X


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, xs: list[str], fe: bool) -> dict:
    """학습 계수로 검증 예상: 날씨 짐작(xs 의 _hat 버전), 날씨 없음, 실제 날씨(상한)."""
    Xtr = _X(train, xs, fe)
    beta = np.linalg.lstsq(Xtr.to_numpy(), train["y"].to_numpy(float), rcond=None)[0]
    Xtr0 = _X(train, [], fe)
    beta0 = np.linalg.lstsq(Xtr0.to_numpy(), train["y"].to_numpy(float), rcond=None)[0]
    pred_real = _X(test, xs, fe, Xtr.columns).to_numpy() @ beta
    hat = test.copy()
    for x in xs:
        hat[x] = test[f"{x}_hatd"]
    pred_hat = _X(hat, xs, fe, Xtr.columns).to_numpy() @ beta
    pred0 = _X(test, [], fe, Xtr0.columns).to_numpy() @ beta0
    return dict(coef={x: float(b) for x, b in zip(Xtr.columns, beta) if x in xs},
                pred_hat=pred_hat, pred0=pred0, pred_real=pred_real)


def evaluate(results: dict, alpha: float = 0.05) -> tuple[pd.DataFrame, dict]:
    """results: {name: DataFrame(year, q, y, pred_hat, pred0, pred_real)} → 물량별 요약 + 묶음 판정."""
    summ, pooled = [], []
    for name, r in results.items():
        e1, e0, er = r["y"] - r["pred_hat"], r["y"] - r["pred0"], r["y"] - r["pred_real"]
        mse0 = float((e0 ** 2).mean())
        summ.append(dict(series=name, n=len(r), quarters=r[["year", "q"]].drop_duplicates().shape[0],
                         rmse_no_weather=np.sqrt(mse0), rmse_nowcast=float(np.sqrt((e1 ** 2).mean())),
                         rmse_actual_weather=float(np.sqrt((er ** 2).mean())),
                         mse_reduction=1 - float((e1 ** 2).mean()) / mse0,
                         hit_nowcast=float((np.sign(r["pred_hat"]) == np.sign(r["y"])).mean()),
                         hit_no_weather=float((np.sign(r["pred0"]) == np.sign(r["y"])).mean())))
        pooled.append(r.assign(d=(e0 ** 2 - e1 ** 2) / mse0)[["year", "q", "d"]])
    by_q = pd.concat(pooled).groupby(["year", "q"])["d"].mean()
    t, p2 = stats.ttest_1samp(by_q.to_numpy(), 0.0)
    p_one = p2 / 2 if t > 0 else 1 - p2 / 2
    verdict = dict(n_quarters=len(by_q), mean_gain=float(by_q.mean()), t=float(t), p_one=float(p_one),
                   quarters_better=int((by_q > 0).sum()), passed=bool(p_one < alpha and by_q.mean() > 0))
    return pd.DataFrame(summ), verdict
