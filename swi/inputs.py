"""1층: 입력 데이터 (기상정보 · 기업정보 · 금융정보)

핵심은 '월말 시점에 실제로 알 수 있었던 정보'만 쓰는 것입니다.
- 기상: 분기 중 이미 관측된 달 + 남은 달은 평년값(직전 30년 평균). 예보 자료를 쌓으면 남은 달 자리에 예보를 넣으면 됩니다.
- 실적: 분기말 + DISCLOSURE_LAG_DAYS 가 지난 분기만 사용.
- 주가: 월말 종가까지만 특징으로 쓰고, 다음 달 수익률은 맞혀야 할 정답으로만 씁니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTH_VARS = ["HDD", "CDD", "HOT", "COLD", "RAIN", "TSUM"]


def monthly_weather(w: pd.DataFrame, base=18.0, hot=33.0, cold=-12.0, max_missing=0.10) -> pd.DataFrame:
    """지점×연×월 합계. 결측일이 10% 넘으면 NaN, 아니면 일수 비율로 보정. 강수 빈칸은 0(무강수)."""
    d = w.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["HDD"] = (base - d["tavg"]).clip(lower=0)
    d["CDD"] = (d["tavg"] - base).clip(lower=0)
    d["HOT"] = np.where(d["tmax"].isna(), np.nan, (d["tmax"] >= hot).astype(float))
    d["COLD"] = np.where(d["tmin"].isna(), np.nan, (d["tmin"] <= cold).astype(float))
    d["RAIN"] = d["rain"].fillna(0.0) if "rain" in d else 0.0
    d["TSUM"] = d["tavg"]
    d["year"], d["month"] = d["date"].dt.year, d["date"].dt.month
    g = d.groupby(["stn", "year", "month"])
    m = g[MONTH_VARS].sum(min_count=1)
    m["n_valid"] = g["tavg"].count()
    m = m.reset_index()
    m["n_days"] = pd.to_datetime(dict(year=m["year"], month=m["month"], day=1)).dt.days_in_month
    ratio = m["n_days"] / m["n_valid"].replace(0, np.nan)
    bad = m["n_valid"] < (1 - max_missing) * m["n_days"]
    for c in MONTH_VARS:
        m[c] = (m[c] * ratio).where(~bad)
    m["quarter"] = (m["month"] - 1) // 3 + 1
    return m


def _prior_roll(s: pd.Series, years: int, min_years: int, fn: str) -> pd.Series:
    r = s.shift(1).rolling(years, min_periods=min_years)
    return getattr(r, fn)()


def expected_quarter(m: pd.DataFrame, years: int = 30, min_years: int = 20) -> pd.DataFrame:
    """각 지점·월말 시점의 '이번 분기 예상값'과 그 불확실성.

    반환 열 (지표 V마다): V_exp(관측+평년 예상), V_sd(남은 달의 과거 변동성), V_ly(작년 같은 분기 실제),
    V_norm(분기 평년), V_qsd(분기 합계의 과거 표준편차), obs_share(관측된 달 비율)
    평년·표준편차는 해당 연도 '이전' 자료만 씁니다."""
    rows = []
    for stn, g in m.groupby("stn"):
        g = g.set_index(["year", "month"]).sort_index()
        years_idx = sorted(g.index.get_level_values(0).unique())
        full = g.reindex(pd.MultiIndex.from_product([range(min(years_idx), max(years_idx) + 1), range(1, 13)],
                                                    names=["year", "month"]))
        out = {}
        for v in MONTH_VARS:
            piv = full[v].unstack("month")                        # 연 × 월
            norm_m = piv.apply(lambda c: _prior_roll(c, years, min_years, "mean"))
            for q in (1, 2, 3, 4):
                ms = [3 * q - 2, 3 * q - 1, 3 * q]
                qtot = piv[ms].sum(axis=1, min_count=3)
                q_norm = _prior_roll(qtot, years, min_years, "mean")
                q_sd = _prior_roll(qtot, years, min_years, "std")
                q_ly = qtot.shift(1)
                for k, mcut in enumerate(ms):                      # 월말 = 분기 k+1번째 달 말
                    obs = piv[ms[:k + 1]].sum(axis=1, min_count=k + 1)
                    rem_ms = ms[k + 1:]
                    if rem_ms:
                        rem_norm = norm_m[rem_ms].sum(axis=1, min_count=len(rem_ms))
                        rem_hist = piv[rem_ms].sum(axis=1, min_count=len(rem_ms))
                        rem_sd = _prior_roll(rem_hist, years, min_years, "std")
                    else:
                        rem_norm = pd.Series(0.0, index=piv.index)
                        rem_sd = pd.Series(0.0, index=piv.index)
                    for y in piv.index:
                        key = (stn, y, mcut)
                        rec = out.setdefault(key, {"stn": stn, "year": y, "month": mcut, "quarter": q,
                                                   "obs_share": (k + 1) / 3})
                        rec[f"{v}_exp"] = obs.get(y, np.nan) + rem_norm.get(y, np.nan)
                        rec[f"{v}_sd"] = rem_sd.get(y, np.nan)
                        rec[f"{v}_ly"] = q_ly.get(y, np.nan)
                        rec[f"{v}_norm"] = q_norm.get(y, np.nan)
                        rec[f"{v}_qsd"] = q_sd.get(y, np.nan)
                        rec[f"{v}_act"] = qtot.get(y, np.nan)
        rows.extend(out.values())
    e = pd.DataFrame(rows)
    # 평균기온: 분기 기온합 / 분기 일수
    days = pd.to_datetime(dict(year=e["year"], month=3 * e["quarter"] - 2, day=1))
    ndays = ((days + pd.offsets.QuarterEnd(0)) - days).dt.days + 1
    for suf in ("exp", "sd", "ly", "norm", "qsd", "act"):
        e[f"TAVG_{suf}"] = e[f"TSUM_{suf}"] / ndays
    e["stn"] = e["stn"].astype(str)
    return e


def add_national(e: pd.DataFrame, members: list) -> pd.DataFrame:
    """전국 6개 도시 평균. 표준편차는 평균을 쓰므로 지역 간 상쇄를 반영하지 못해 다소 보수적입니다."""
    sub = e[e["stn"].isin([str(s) for s in members])]
    num = [c for c in e.columns if c not in ("stn", "year", "month", "quarter", "obs_share")]
    nat = sub.groupby(["year", "month", "quarter", "obs_share"])[num].mean().reset_index()
    nat["stn"] = "NATIONAL"
    return pd.concat([e, nat], ignore_index=True)


def month_end_prices(px: pd.DataFrame, mkt: pd.DataFrame) -> pd.DataFrame:
    """월말 종가 기준: 지난 1·3개월 초과수익률, 거래량 변화, 다음 달 초과수익률(정답)."""
    p = px.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["ym"] = p["date"].dt.to_period("M")
    last = p.sort_values("date").groupby(["code", "ym"]).agg(close=("close", "last"), date=("date", "last"))
    vol = p.groupby(["code", "ym"])["volume"].mean() if "volume" in p else None
    k = mkt.copy()
    k["date"] = pd.to_datetime(k["date"])
    k["ym"] = k["date"].dt.to_period("M")
    kl = k.sort_values("date").groupby("ym")["close"].last()
    out = last.reset_index()
    out["mret"] = out.groupby("code")["close"].pct_change()
    out = out.merge(kl.pct_change().rename("mkt").reset_index(), on="ym", how="left")
    out["xret"] = out["mret"] - out["mkt"]
    g = out.groupby("code")["xret"]
    out["mom1"] = out["xret"]
    out["mom3"] = g.transform(lambda s: s.rolling(3).sum())
    out["fwd_xret"] = g.shift(-1)
    if vol is not None:
        v = vol.reset_index().rename(columns={"volume": "v"})
        out = out.merge(v, on=["code", "ym"], how="left")
        out["vol_chg"] = np.log(out["v"] / out.groupby("code")["v"].transform(lambda s: s.shift(1).rolling(12, min_periods=6).mean()))
    else:
        out["vol_chg"] = np.nan
    out["year"], out["month"] = out["ym"].dt.year, out["ym"].dt.month
    return out


def realized_quarters(e: pd.DataFrame) -> pd.DataFrame:
    """분기말(3·6·9·12월) 행만 남겨 '실제 분기값 - 작년 같은 분기'를 만듭니다 (민감도 학습용)."""
    q = e[e["obs_share"] == 1.0].copy()
    for v in ("HDD", "CDD", "HOT", "COLD", "RAIN", "TAVG"):
        q[f"{v}_yoy"] = q[f"{v}_act"] - q[f"{v}_ly"]
    return q[["stn", "year", "quarter"] + [f"{v}_yoy" for v in ("HDD", "CDD", "HOT", "COLD", "RAIN", "TAVG")]]
