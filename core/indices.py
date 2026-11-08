"""일자료 -> 날씨지수(분기). 평년은 '해당 연도 이전' 자료만 써서 미래정보를 배제합니다."""
from __future__ import annotations

import numpy as np
import pandas as pd

VALUE_COLS = ["HDD", "CDD", "HOT", "TAVG"]


def daily_indices(df: pd.DataFrame, base: float = 18.0, hot: float = 33.0) -> pd.DataFrame:
    """난방도일(HDD), 냉방도일(CDD), 폭염일(HOT)을 일별로 계산합니다.
    결측 기온은 결측으로 남기고 0으로 채우지 않습니다."""
    d = df.copy()
    d["HDD"] = (base - d["tavg"]).clip(lower=0)
    d["CDD"] = (d["tavg"] - base).clip(lower=0)
    d["HOT"] = np.where(d["tmax"].isna(), np.nan, (d["tmax"] >= hot).astype(float))
    d["TAVG"] = d["tavg"]
    return d


def quarterly(d: pd.DataFrame, max_missing_share: float = 0.10) -> pd.DataFrame:
    """지점×연도×분기 집계. 결측일이 분기 일수의 10%를 넘으면 그 분기는 NaN."""
    d = d.copy()
    d["year"] = d["date"].dt.year
    d["q"] = d["date"].dt.quarter
    g = d.groupby(["stn", "year", "q"])
    out = g.agg(HDD=("HDD", "sum"), CDD=("CDD", "sum"), HOT=("HOT", "sum"),
                TAVG=("TAVG", "mean"), n_valid=("tavg", "count")).reset_index()
    # 분기 달력일수 (윤년 반영)
    qs = pd.to_datetime(dict(year=out["year"], month=3 * out["q"] - 2, day=1))
    out["n_days"] = ((qs + pd.offsets.QuarterEnd(0)) - qs).dt.days + 1
    bad = out["n_valid"] < (1 - max_missing_share) * out["n_days"]
    out.loc[bad, VALUE_COLS] = np.nan
    # 결측일이 있어도 허용 범위면 합계는 일수 비율로 보정 (평균 × 일수)
    ratio = out["n_days"] / out["n_valid"].replace(0, np.nan)
    for c in ("HDD", "CDD", "HOT"):
        out[c] = out[c] * ratio
    return out.drop(columns=["n_days"])


def national_average(qdf: pd.DataFrame, stations: list[int], min_stations: int = 5) -> pd.DataFrame:
    """여러 지점 평균. 유효 지점이 min_stations 미만이면 NaN."""
    sub = qdf[qdf["stn"].isin(stations)]
    g = sub.groupby(["year", "q"])
    out = g[VALUE_COLS].mean()
    cnt = g[VALUE_COLS].count()
    out = out.where(cnt >= min_stations)
    out = out.reset_index()
    out["stn"] = "NATIONAL"
    return out


def add_normals(qdf: pd.DataFrame, years: int = 30, min_years: int = 20) -> pd.DataFrame:
    """각 지점·분기·연도 y에 대해 [y-years, y-1] 구간의 평균·표준편차(평년)를 붙입니다.
    y년 자신과 그 이후 값은 절대 쓰지 않습니다."""
    out = []
    for (stn, q), g in qdf.groupby(["stn", "q"]):
        g = g.sort_values("year").set_index("year")
        full = g.reindex(range(g.index.min(), g.index.max() + 1))
        for c in VALUE_COLS:
            prior = full[c].shift(1)  # y-1까지만
            roll = prior.rolling(years, min_periods=min_years)
            full[f"{c}_normal"] = roll.mean()
            full[f"{c}_sd"] = roll.std()
            full[f"{c}_anom"] = full[c] - full[f"{c}_normal"]
            full[f"{c}_yoy"] = full[c] - full[c].shift(1)  # 전년 동분기 대비 변화
        full["stn"] = stn
        full["q"] = q
        out.append(full.dropna(subset=["q"]).reset_index().rename(columns={"index": "year"}))
    res = pd.concat(out, ignore_index=True)
    return res[res[VALUE_COLS].notna().any(axis=1)].reset_index(drop=True)
