"""오늘 기준 '이번 분기 날씨' 추정 = 관측(지난 날) + 예보(3~10일 뒤) + 평년(그 이후).

월말 백테스트(swi/inputs.py)와 같은 지표를 일 단위로 만듭니다. 출력 열 이름도 같아서 같은 점수 공식을 그대로 씁니다.
- 평년: 그 해 '이전' 30년의 같은 날짜 값을 평균하고 앞뒤 15일로 다듬음 (하루 단위 평년은 잡음이 크기 때문)
- 불확실성: 평년으로 채운 기간은 과거 같은 기간 합계의 표준편차, 예보로 채운 날은 예보 오차(FORECAST_ERR_SD)를 더함
- 예보에는 강수량이 없어 강수는 예보일에도 평년을 씁니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VARS = ["HDD", "CDD", "HOT", "COLD", "RAIN", "TSUM"]


def daily_index(w: pd.DataFrame, base=18.0, hot=33.0, cold=-12.0) -> pd.DataFrame:
    d = w.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["HDD"] = (base - d["tavg"]).clip(lower=0)
    d["CDD"] = (d["tavg"] - base).clip(lower=0)
    d["HOT"] = np.where(d["tmax"].isna(), np.nan, (d["tmax"] >= hot).astype(float))
    d["COLD"] = np.where(d["tmin"].isna(), np.nan, (d["tmin"] <= cold).astype(float))
    d["RAIN"] = d["rain"].fillna(0.0) if "rain" in d else 0.0
    d["TSUM"] = d["tavg"]
    return d


def _md(dates: pd.DatetimeIndex) -> np.ndarray:
    md = dates.strftime("%m-%d")
    return np.where(md == "02-29", "02-28", md)


def _shift_year(dates: pd.DatetimeIndex, year: int) -> pd.DatetimeIndex:
    md = _md(dates)
    return pd.to_datetime([f"{year}-{m}" for m in md])


def climatology(s: pd.DataFrame, dates: pd.DatetimeIndex, year: int, years: int, min_years: int,
                smooth: int) -> pd.DataFrame:
    """s: 한 지점의 일별 지표(date 인덱스). dates에 대한 평년 일값(VARS)."""
    prior = s[(s.index.year >= year - years) & (s.index.year <= year - 1)]
    if prior.index.year.nunique() < min_years:
        return pd.DataFrame(np.nan, index=dates, columns=VARS)
    half = smooth // 2
    ext = pd.date_range(dates.min() - pd.Timedelta(days=half), dates.max() + pd.Timedelta(days=half))
    by_md = prior[VARS].groupby(_md(prior.index)).mean()
    vals = by_md.reindex(_md(ext))
    vals.index = ext
    vals = vals.rolling(smooth, center=True, min_periods=1).mean()
    return vals.reindex(dates)


def window_hist(s: pd.DataFrame, a: pd.Timestamp, b: pd.Timestamp, year: int, years: int, min_years: int,
                min_cov: float = 0.9) -> pd.DataFrame:
    """과거 각 해의 같은 달력 구간 [a, b] 합계 (VARS × 연도)."""
    out = {}
    if b < a:
        return pd.DataFrame(columns=VARS)
    span = pd.date_range(a, b)
    for yy in range(year - years, year):
        dd = _shift_year(span, yy).unique()
        seg = s.reindex(dd)
        if seg["TSUM"].notna().mean() >= min_cov:
            out[yy] = seg[VARS].sum(min_count=1) * (len(dd) / max(seg["TSUM"].notna().sum(), 1))
    h = pd.DataFrame(out).T
    return h if len(h) >= min_years else pd.DataFrame(columns=VARS)


def nowcast_station(s: pd.DataFrame, fc: pd.DataFrame | None, asof: pd.Timestamp, cfg) -> dict:
    y, q = asof.year, asof.quarter
    qs = pd.Timestamp(year=y, month=3 * q - 2, day=1)
    qe = qs + pd.offsets.QuarterEnd(0)
    days = pd.date_range(qs, qe)
    n = len(days)
    obs_known = s[(s.index <= asof) & s["TSUM"].notna()]
    last_obs = obs_known.index.max() if len(obs_known) else qs - pd.Timedelta(days=1)
    obs_days = days[days <= min(last_obs, asof)]
    clim = climatology(s, days, y, cfg.NORMAL_YEARS, cfg.MIN_NORMAL_YEARS, cfg.CLIM_SMOOTH_DAYS)

    # 관측 구간 (빠진 날은 평년으로)
    obs = s.reindex(obs_days)[VARS]
    obs = obs.where(obs.notna(), clim.reindex(obs_days))
    part_obs = obs.sum()

    # 예보 구간: 관측 이후, 최신 발표분
    fc_days = pd.DatetimeIndex([])
    part_fc = pd.Series(0.0, index=VARS)
    if fc is not None and len(fc):
        f = fc[(pd.to_datetime(fc["target_date"]) > last_obs) & (pd.to_datetime(fc["target_date"]) <= qe)].copy()
        if len(f):
            f = f.sort_values("issued_at").groupby("target_date").last()
            f.index = pd.to_datetime(f.index)
            fc_days = f.index
            tav = (f["tmin"] + f["tmax"]) / 2
            part_fc = pd.Series({
                "HDD": (cfg.BASE_TEMP - tav).clip(lower=0).sum(), "CDD": (tav - cfg.BASE_TEMP).clip(lower=0).sum(),
                "HOT": (f["tmax"] >= cfg.HEATWAVE_TMAX).sum(), "COLD": (f["tmin"] <= cfg.COLD_TMIN).sum(),
                "RAIN": clim.reindex(fc_days)["RAIN"].sum(), "TSUM": tav.sum()})

    rest_days = days[(days > min(last_obs, asof)) & ~days.isin(fc_days)]
    part_clim = clim.reindex(rest_days)[VARS].sum()

    # 불확실성: 평년 구간은 과거 같은 구간 합계의 표준편차(연속 구간일 때), 예보일은 예보 오차
    if len(rest_days):
        hist = window_hist(s, rest_days.min(), rest_days.max(), y, cfg.NORMAL_YEARS, cfg.MIN_NORMAL_YEARS)
        rest_sd = hist.std() if len(hist) else pd.Series(np.nan, index=VARS)
        if len(rest_days) != (rest_days.max() - rest_days.min()).days + 1:  # 예보가 중간에 끼면 비율로 근사
            span = (rest_days.max() - rest_days.min()).days + 1
            rest_sd = rest_sd * np.sqrt(len(rest_days) / span)
    else:
        rest_sd = pd.Series(0.0, index=VARS)
    fc_var = pd.Series(0.0, index=VARS)
    for v in ("HDD", "CDD", "TSUM"):
        fc_var[v] = len(fc_days) * cfg.FORECAST_ERR_SD ** 2
    sd = np.sqrt(rest_sd.fillna(np.nan) ** 2 + fc_var)

    # 작년 같은 분기, 분기 평년·표준편차
    lyq = window_hist(s, qs, qe, y, 1, 1)                       # y-1 한 해
    qhist = window_hist(s, qs, qe, y, cfg.NORMAL_YEARS, cfg.MIN_NORMAL_YEARS)
    rec = dict(asof=asof.strftime("%Y-%m-%d"), year=y, quarter=q, obs_share=len(obs_days) / n,
               n_obs=len(obs_days), n_fc=len(fc_days), n_clim=len(rest_days), last_obs=last_obs.strftime("%Y-%m-%d"))
    for v in VARS:
        rec[f"{v}_obs"], rec[f"{v}_fc"], rec[f"{v}_clim"] = part_obs[v], part_fc[v], part_clim[v]
        rec[f"{v}_exp"] = part_obs[v] + part_fc[v] + part_clim[v]
        rec[f"{v}_sd"] = sd[v]
        rec[f"{v}_ly"] = lyq[v].iloc[0] if len(lyq) else np.nan
        rec[f"{v}_norm"] = qhist[v].mean() if len(qhist) else np.nan
        rec[f"{v}_qsd"] = qhist[v].std() if len(qhist) else np.nan
    for suf in ("obs", "fc", "clim", "exp", "sd", "ly", "norm", "qsd"):
        rec[f"TAVG_{suf}"] = rec[f"TSUM_{suf}"] / n
    return rec


def nowcast(w: pd.DataFrame, fc: pd.DataFrame | None, asof: pd.Timestamp, cfg) -> pd.DataFrame:
    d = daily_index(w, cfg.BASE_TEMP, cfg.HEATWAVE_TMAX, cfg.COLD_TMIN)
    rows = []
    for stn, g in d.groupby("stn"):
        s = g.set_index("date").sort_index()
        f = fc[fc["stn"].astype(int) == int(stn)] if fc is not None and len(fc) else None
        rec = nowcast_station(s, f, asof, cfg)
        rec["stn"] = str(stn)
        rows.append(rec)
    out = pd.DataFrame(rows)
    nat = out[out["stn"].isin([str(s) for s in cfg.NATIONAL])]
    if len(nat):
        num = nat.select_dtypes("number").mean()
        r = {**{k: nat.iloc[0][k] for k in ("asof", "year", "quarter", "last_obs")}, **num.to_dict(), "stn": "NATIONAL"}
        out = pd.concat([out, pd.DataFrame([r])], ignore_index=True)
    return out
