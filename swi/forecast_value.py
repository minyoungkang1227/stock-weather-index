"""예보의 값어치 검증 (config.PREREG_FORECAST_VALUE).

분기 중 어느 날(as-of, 그날 06시 발표 예보까지 앎) 이번 분기 난방도일(HDD) 합계를 짐작할 때
  [A] 어제까지 관측 + 남은 날 전부 평년
  [B] 어제까지 관측 + 예보가 있는 날은 예보 + 나머지 평년
중 어느 쪽이 실제 분기 합계에 더 가까웠는지 비교한다.

왜 HDD 로 비교하나: 날씨→판매량 관계(100도일당 +5.2%)는 검정으로 확인됐으므로, 판매량을 짐작하는 정확도는
HDD 를 짐작하는 정확도로 정해진다. 판매량 자체에는 날씨 외 잡음이 섞여 비교가 흐려진다.
미래 정보 차단: 평년은 그해 '이전' 30년만, 예보 편향 보정은 그해 '이전' 해들의 예보 오차만 쓴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _md(idx: pd.DatetimeIndex) -> np.ndarray:
    md = idx.strftime("%m-%d")
    return np.where(md == "02-29", "02-28", md)


def climatology_hdd(hdd: pd.Series, year: int, dates: pd.DatetimeIndex, years: int = 30, min_years: int = 20,
                    smooth: int = 15) -> pd.Series:
    """그해 이전 years 년의 같은 날짜 평균 HDD, 앞뒤로 다듬음 (system/nowcast.climatology 와 같은 방식)."""
    prior = hdd[(hdd.index.year >= year - years) & (hdd.index.year <= year - 1)].dropna()
    if prior.index.year.nunique() < min_years:
        return pd.Series(np.nan, index=dates)
    by_md = prior.groupby(_md(prior.index)).mean()
    half = smooth // 2
    ext = pd.date_range(dates.min() - pd.Timedelta(days=half), dates.max() + pd.Timedelta(days=half))
    v = pd.Series(by_md.reindex(_md(ext)).to_numpy(), index=ext).rolling(smooth, center=True, min_periods=1).mean()
    return v.reindex(dates)


def quarter_errors(hdd_q: pd.Series, clim_q: pd.Series, fc_by_asof: dict) -> pd.DataFrame:
    """한 지점·한 분기. hdd_q/clim_q: 분기 각 날짜의 실제 HDD / 평년 HDD.
    fc_by_asof: {as-of 날짜: Series(대상날짜 → 예보 HDD)} (그날 06시 발표분).
    as-of 날짜 d 마다: 관측 = d 전날까지 실제, 남은 날 [d, 분기말] = 평년(A) 또는 예보 우선(B)."""
    days = hdd_q.index
    real = float(hdd_q.sum())
    cum_obs = np.concatenate([[0.0], np.cumsum(hdd_q.to_numpy())])     # cum_obs[i] = 처음 i일 합
    clim = clim_q.to_numpy()
    cum_clim_rev = np.concatenate([np.cumsum(clim[::-1])[::-1], [0.0]])  # cum_clim_rev[i] = i번째 날부터 끝까지 평년 합
    rows = []
    for i, d in enumerate(days):
        a = cum_obs[i] + cum_clim_rev[i]
        b = a
        f = fc_by_asof.get(d)
        n_fc = 0
        if f is not None and len(f):
            f = f[(f.index >= d) & (f.index <= days[-1])].dropna()
            if len(f):
                pos = days.get_indexer(f.index)
                b = a - clim[pos].sum() + f.to_numpy().sum()
                n_fc = len(f)
        rows.append(dict(asof=d, days_left=len(days) - i, n_fc=n_fc, err_A=a - real, err_B=b - real))
    return pd.DataFrame(rows)


def lead_bias_by_year(fc: pd.DataFrame, obs_tavg: pd.DataFrame) -> dict:
    """{연도 y: Series(리드 → 평균오차)}, y 이전 해에 발표된 예보의 오차만 사용 (미래 정보 차단)."""
    f = fc.copy()
    f["date"] = pd.to_datetime(f["target_date"])
    f["fc_tavg"] = (f["tmin"] + f["tmax"]) / 2
    o = obs_tavg[["stn", "date", "tavg"]].copy()
    o["date"] = pd.to_datetime(o["date"])
    m = f.merge(o, on=["stn", "date"], how="inner").dropna(subset=["fc_tavg", "tavg"])
    m["err"] = m["fc_tavg"] - m["tavg"]
    m["iy"] = m["issued_at"].astype(str).str[:4].astype(int)
    out = {}
    for y in sorted(m["iy"].unique()) + [int(m["iy"].max()) + 1]:
        prev = m[m["iy"] < y]
        out[int(y)] = prev.groupby("lead_day")["err"].mean() if len(prev) else pd.Series(dtype=float)
    return out


def run(w: pd.DataFrame, fc: pd.DataFrame, stations: list[int], quarters: list[int], start: str, end: str,
        base: float = 18.0, issue_hour: str = "06") -> pd.DataFrame:
    """전 지점·분기의 as-of 오차표. w: stn,date,tavg / fc: issued_at,stn,lead_day,target_date,tmin,tmax."""
    w = w.copy()
    w["date"] = pd.to_datetime(w["date"])
    w["stn"] = w["stn"].astype(int)
    fc = fc.copy()
    fc["stn"] = fc["stn"].astype(int)
    fc["issued_at"] = fc["issued_at"].astype(str)
    fc = fc[fc["issued_at"].str[8:10] == issue_hour]
    bias = lead_bias_by_year(fc, w)
    fc["asof"] = pd.to_datetime(fc["issued_at"].str[:8])
    fc["date"] = pd.to_datetime(fc["target_date"])
    fc["y"] = fc["asof"].dt.year
    fc["bias"] = [bias.get(int(y), pd.Series(dtype=float)).get(int(l), 0.0) for y, l in zip(fc["y"], fc["lead_day"])]
    fc["hdd"] = (base - ((fc["tmin"] + fc["tmax"]) / 2 - fc["bias"])).clip(lower=0)

    out = []
    qs_all = pd.period_range(pd.Timestamp(start).to_period("Q"), pd.Timestamp(end).to_period("Q"), freq="Q")
    for stn in stations:
        s = w[w["stn"] == stn].set_index("date")["tavg"].sort_index()
        hdd = (base - s).clip(lower=0).where(s.notna())
        f_stn = fc[fc["stn"] == stn]
        f_groups = {d: g.set_index("date")["hdd"] for d, g in f_stn.groupby("asof")}
        for q in qs_all:
            if q.quarter not in quarters:
                continue
            days = pd.date_range(q.start_time, q.end_time.normalize())
            hq = hdd.reindex(days)
            if hq.isna().any():
                continue                                   # 관측이 빠진 지점·분기는 비교에서 제외
            cq = climatology_hdd(hdd, q.year, days)
            if cq.isna().any():
                continue
            e = quarter_errors(hq, cq, {d: f_groups[d] for d in days if d in f_groups})
            e["stn"], e["quarter"] = stn, str(q)
            out.append(e)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def verdict(e: pd.DataFrame, alpha: float = 0.05) -> dict:
    """분기별로 (지점·날짜 평균) 손실 차 A²−B² 를 구해 분기 단위 단측 t검정."""
    e = e.assign(dl=e["err_A"] ** 2 - e["err_B"] ** 2)
    by_q = e.groupby("quarter")["dl"].mean()
    t, p2 = stats.ttest_1samp(by_q.to_numpy(), 0.0)
    p_one = p2 / 2 if t > 0 else 1 - p2 / 2
    rmse_a, rmse_b = float(np.sqrt((e["err_A"] ** 2).mean())), float(np.sqrt((e["err_B"] ** 2).mean()))
    return dict(n_quarters=len(by_q), n_points=len(e), rmse_A=rmse_a, rmse_B=rmse_b,
                rmse_reduction=1 - rmse_b / rmse_a, mean_dl=float(by_q.mean()), t=float(t), p_one=float(p_one),
                quarters_B_better=int((by_q > 0).sum()), passed=bool(p_one < alpha and by_q.mean() > 0))


def by_days_left(e: pd.DataFrame, bins=(0, 10, 30, 60, 92)) -> pd.DataFrame:
    e = e.assign(bucket=pd.cut(e["days_left"], bins=bins))
    g = e.groupby("bucket", observed=True)
    return pd.DataFrame({"n": g.size(), "rmse_A": g["err_A"].apply(lambda x: np.sqrt((x ** 2).mean())),
                         "rmse_B": g["err_B"].apply(lambda x: np.sqrt((x ** 2).mean()))}).assign(
        reduction=lambda d: 1 - d["rmse_B"] / d["rmse_A"]).reset_index()
