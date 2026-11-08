"""데이터 품질 점검. 누락·이상치·신선도·회계 정합성을 각각 따로 봅니다.
각 함수는 문제 행 표를 돌려주고, 판단(멈출지)은 호출하는 쪽에서 합니다."""
from __future__ import annotations

import pandas as pd


def weather_missing(df: pd.DataFrame) -> pd.DataFrame:
    """지점별 기간 내 빠진 날짜 수."""
    rows = []
    for stn, g in df.groupby("stn"):
        full = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        rows.append({"stn": stn, "first": g["date"].min().date(), "last": g["date"].max().date(),
                     "missing_days": len(full.difference(g["date"])),
                     "missing_tavg": int(g["tavg"].isna().sum()),
                     "missing_tmax": int(g["tmax"].isna().sum())})
    return pd.DataFrame(rows)


def weather_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """물리적으로 이상한 값: 기온 범위 밖, 최저>최고, 평균이 최저~최고 밖."""
    bad = (
        (df["tavg"] < -35) | (df["tavg"] > 40) |
        (df["tmin"] > df["tmax"]) |
        (df["tavg"] < df["tmin"] - 0.5) | (df["tavg"] > df["tmax"] + 0.5)
    )
    return df[bad.fillna(False)]


def weather_freshness(df: pd.DataFrame, need_until: str) -> pd.DataFrame:
    last = df.groupby("stn")["date"].max().reset_index()
    return last[last["date"] < pd.Timestamp(need_until)]


def fin_checks(qf: pd.DataFrame, cum: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """분기 실적 점검: 음수/0 매출, 결측 분기, 공시 단독금액과 차분값 불일치."""
    out = {
        "nonpositive_revenue": qf[qf["revenue"] <= 0],
        "missing_revenue": qf[qf["revenue"].isna()],
    }
    if cum is not None and {"rev_cur"}.issubset(cum.columns):
        m = qf.merge(cum[["code", "year", "q", "rev_cur"]], on=["code", "year", "q"])
        m = m[m["q"].isin([2, 3])]  # 반기·3분기 보고서의 3개월 단독 금액과 대조
        rel = (m["revenue"] - m["rev_cur"]).abs() / m["rev_cur"].abs()
        out["derived_vs_reported_gt1pct"] = m[rel > 0.01]
    return out
