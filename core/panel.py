"""기업×분기 패널 만들기: 실적 성장률, 초과수익률, 기업별 날씨지수를 합칩니다."""
from __future__ import annotations

import numpy as np
import pandas as pd


def firm_weather(wq: pd.DataFrame, firms: dict) -> pd.DataFrame:
    """config.FIRMS 매핑대로 기업마다 해당 지점(또는 전국평균)의 날씨지수를 붙입니다."""
    rows = []
    for code, (name, group, stn, main) in firms.items():
        w = wq[wq["stn"].astype(str) == str(stn)].copy()
        w["code"], w["name"], w["group"], w["main_index"] = code, name, group, main
        rows.append(w)
    return pd.concat(rows, ignore_index=True)


def fin_growth(qf: pd.DataFrame) -> pd.DataFrame:
    """전년 동분기 대비: 매출은 로그 성장률, 영업이익은 '전년 동분기 매출 대비 변화'
    (영업이익은 음수가 될 수 있어 로그를 쓸 수 없기 때문)."""
    f = qf.sort_values(["code", "year", "q"]).copy()
    prev = f[["code", "year", "q", "revenue", "op_income"]].copy()
    prev["year"] += 1
    f = f.merge(prev, on=["code", "year", "q"], how="left", suffixes=("", "_ly"))
    ok = (f["revenue"] > 0) & (f["revenue_ly"] > 0)
    f["rev_g"] = np.where(ok, np.log(f["revenue"] / f["revenue_ly"]), np.nan)
    f["op_chg"] = np.where(f["revenue_ly"] > 0, (f["op_income"] - f["op_income_ly"]) / f["revenue_ly"], np.nan)
    return f


def quarterly_returns(px: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """분기 수익률과 시장 대비 초과수익률, 다음 분기 초과수익률(지연 반응 확인용)."""
    def q_ret(df: pd.DataFrame, key: str | None) -> pd.DataFrame:
        df = df.sort_values("date").copy()
        df["year"], df["q"] = df["date"].dt.year, df["date"].dt.quarter
        cols = ["year", "q"] if key is None else [key, "year", "q"]
        last = df.groupby(cols)["close"].last()
        if key is None:
            r = last.pct_change()
        else:
            r = last.groupby(level=0).pct_change()
        return r.rename("ret").reset_index()

    fr = q_ret(px, "code")
    mr = q_ret(market.assign(code="MKT"), None).rename(columns={"ret": "mkt_ret"})
    fr = fr.merge(mr, on=["year", "q"], how="left")
    fr["xret"] = fr["ret"] - fr["mkt_ret"]
    fr = fr.sort_values(["code", "year", "q"])
    fr["xret_next"] = fr.groupby("code")["xret"].shift(-1)
    return fr


def build_panel(fw: pd.DataFrame, fin: pd.DataFrame | None, rets: pd.DataFrame | None) -> pd.DataFrame:
    p = fw.copy()
    if fin is not None:
        p = p.merge(fin, on=["code", "year", "q"], how="left")
    if rets is not None:
        p = p.merge(rets[["code", "year", "q", "xret", "xret_next"]], on=["code", "year", "q"], how="left")
    p["t"] = p["year"] * 10 + p["q"]
    return p
