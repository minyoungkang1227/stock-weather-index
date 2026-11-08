"""개선 2: 실적 공시일 이벤트 스터디 — "시장은 날씨 영향을 언제 주가에 반영하는가?"

날씨는 누구나 관측하는 공개 정보입니다. 시장이 효율적이면 분기 '중'에 이미 반영되고,
공시일이나 공시 후에는 날씨로 설명되는 주가 반응이 없어야 합니다.
반대로 공시 전후·후에도 반응이 남으면(지연 반응) 날씨 정보로 예측할 여지가 있습니다.

검정: 구간별 누적 초과수익률(CAR, 종목 − 시장) = a_업종 + b × 날씨 수혜(업종 내 표준화) + e
      표준오차는 같은 분기 이벤트끼리 묶은 클러스터 (같은 분기 같은 업종은 같은 날씨 충격을 공유).
날씨 수혜 = 공시 시점에 알던 민감도 × 그 분기 실제 날씨 변화 (미래 실적으로 만든 민감도를 쓰지 않음).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def daily_abnormal(px: pd.DataFrame, mkt: pd.DataFrame) -> pd.DataFrame:
    p = px.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["code", "date"])
    p["r"] = p.groupby("code")["close"].pct_change()
    k = mkt.copy()
    k["date"] = pd.to_datetime(k["date"])
    k = k.sort_values("date")
    k["rm"] = k["close"].pct_change()
    p = p.merge(k[["date", "rm"]], on="date", how="inner")
    p["ar"] = p["r"] - p["rm"]
    return p[["code", "date", "ar"]].dropna()


def event_dates(fin_keys: pd.DataFrame, disc: pd.DataFrame | None, fallback_days: int) -> pd.DataFrame:
    """(code, year, q) 마다 공시일. 실제 공시일이 없으면 분기말 + fallback_days 로 두고 'kind=가정'으로 표시."""
    k = fin_keys[["code", "year", "q"]].drop_duplicates().copy()
    qe = pd.to_datetime(dict(year=k["year"], month=3 * k["q"], day=1)) + pd.offsets.MonthEnd(0)
    k["qs"] = pd.to_datetime(dict(year=k["year"], month=3 * k["q"] - 2, day=1))
    k["qe"] = qe
    k["date"], k["kind"] = qe + pd.Timedelta(days=fallback_days), "가정(분기말+%d일)" % fallback_days
    if disc is not None and not disc.empty:
        dd = disc.copy()
        dd["date"] = pd.to_datetime(dd["date"])
        k = k.merge(dd[["code", "year", "q", "date", "kind"]], on=["code", "year", "q"], how="left", suffixes=("", "_d"))
        has = k["date_d"].notna()
        k.loc[has, "date"], k.loc[has, "kind"] = k.loc[has, "date_d"], k.loc[has, "kind_d"]
        k = k.drop(columns=["date_d", "kind_d"])
    return k


def cars(ev: pd.DataFrame, ar: pd.DataFrame, windows: dict) -> pd.DataFrame:
    """이벤트별 구간 CAR. None 구간 = 분기 시작~분기말(분기 중). (a, b) = 공시일 기준 거래일 a~b."""
    out = []
    by = {c: g.set_index("date")["ar"].sort_index() for c, g in ar.groupby("code")}
    for _, e in ev.iterrows():
        s = by.get(e["code"])
        if s is None or s.empty:
            continue
        rec = dict(code=e["code"], year=e["year"], q=e["q"], kind=e["kind"])
        pos0 = s.index.searchsorted(e["date"])  # 공시일 당일 또는 그 다음 첫 거래일 = 0
        for name, w in windows.items():
            if w is None:
                seg = s[(s.index >= e["qs"]) & (s.index <= e["qe"])]
                rec[name] = seg.sum() if len(seg) >= 40 else np.nan
            else:
                a, b = pos0 + w[0], pos0 + w[1]
                rec[name] = s.iloc[a:b + 1].sum() if a >= 0 and b < len(s) else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def regress(df: pd.DataFrame, ycol: str) -> dict:
    d = df.dropna(subset=[ycol, "surprise_z"])
    if len(d) < 20 or d["cluster"].nunique() < 5:
        return dict(window=ycol, n=len(d), coef_pct=np.nan, t=np.nan, p=np.nan)
    X = pd.get_dummies(d["group"], drop_first=False, dtype=float)
    X["surprise_z"] = d["surprise_z"].values
    res = sm.OLS(d[ycol].values, X.values).fit(cov_type="cluster", cov_kwds={"groups": d["cluster"].values})
    i = list(X.columns).index("surprise_z")
    return dict(window=ycol, n=int(len(d)), clusters=int(d["cluster"].nunique()),
                coef_pct=100 * res.params[i], t=float(res.tvalues[i]), p=float(res.pvalues[i]))


def verdict(tab: pd.DataFrame, pre: str, later: list[str], t_cut: float = 2.0) -> str:
    t = tab.set_index("window")["t"]
    pre_sig = t.get(pre, np.nan) > t_cut
    late_sig = any(t.get(w, np.nan) > t_cut for w in later)
    if late_sig:
        return "공시 전후·후에도 날씨 수혜 방향의 초과수익이 남음 → 시장이 늦게 반영(지연 반응). 날씨로 예측할 여지가 있음."
    if pre_sig:
        return "분기 중에 이미 날씨 수혜가 주가에 반영되고 공시 이후 추가 반응 없음 → 날씨 '예측'보다 '실시간 관측'이 중요. 상승확률(P)의 근거는 약함."
    return "어느 구간에서도 날씨 수혜와 주가의 유의한 관계가 확인되지 않음 → 주가 예측 부분은 근거 부족. 수혜도·위험경고 중심 활용 권장."
