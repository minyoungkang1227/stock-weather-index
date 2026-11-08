"""4층: 투자 의사결정 지원 — 관심종목 · 위험경고 · 포트폴리오 백테스트"""
from __future__ import annotations

import numpy as np
import pandas as pd


def flags(latest: pd.DataFrame, watch=(65, 50), warn=(30, 60)) -> pd.DataFrame:
    d = latest.copy()
    d["관심종목"] = (d["SWI"] >= watch[0]) & (d["C"] >= watch[1])
    d["위험경고"] = (d["B"] <= warn[0]) & (d["C"] >= warn[1])
    return d.sort_values("SWI", ascending=False)


def newey_west_t(x: np.ndarray, lags: int = 3) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        return np.nan
    e = x - x.mean()
    s = e @ e / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / np.sqrt(s / n))


def long_short(df: pd.DataFrame, score: str, frac: float = 1 / 3, cost_bps: float = 30, min_n: int = 6) -> pd.DataFrame:
    """매월 점수 상위 1/3 매수, 하위 1/3 회피(공매도 가정)한 다음 달 초과수익률. 동일가중, 거래비용 반영."""
    rows, prev_long, prev_short = [], set(), set()
    for t, g in df.dropna(subset=[score, "fwd_xret"]).groupby("ym", sort=True):
        if len(g) < min_n:
            continue
        k = max(1, int(round(len(g) * frac)))
        g = g.sort_values(score)
        short, long = set(g["code"].head(k)), set(g["code"].tail(k))
        turn = (len(long ^ prev_long) + len(short ^ prev_short)) / (2 * k) / 2 if prev_long else 1.0
        cost = turn * cost_bps / 1e4
        rl = g[g["code"].isin(long)]["fwd_xret"].mean()
        rs = g[g["code"].isin(short)]["fwd_xret"].mean()
        rows.append(dict(ym=t, long=rl, short=rs, spread=rl - rs - 2 * cost, long_net=rl - cost, turnover=turn))
        prev_long, prev_short = long, short
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame, label: str) -> dict:
    if bt.empty:
        return dict(score=label, months=0)
    return dict(score=label, months=len(bt),
                spread_mean_pct=100 * bt["spread"].mean(), spread_t_NW=newey_west_t(bt["spread"].values),
                long_mean_pct=100 * bt["long_net"].mean(), long_t_NW=newey_west_t(bt["long_net"].values),
                hit_rate=float((bt["spread"] > 0).mean()), avg_turnover=float(bt["turnover"].mean()))


def long_short_within(df: pd.DataFrame, score: str, group: str = "group", frac: float = 1 / 3,
                      cost_bps: float = 30, min_n: int = 3) -> pd.DataFrame:
    """업종 안에서만 상위 1/3 vs 하위 1/3을 만들고, 그 달에 가능한 업종들의 결과를 동일가중 평균.
    전 종목을 한 줄로 세우면 '계절별 업종 갈아타기'가 되어 종목 선택 능력을 볼 수 없기 때문."""
    parts = []
    for gname, g in df.groupby(group):
        bt = long_short(g, score, frac, cost_bps, min_n=min_n)
        if not bt.empty:
            parts.append(bt.assign(**{group: gname}))
    if not parts:
        return pd.DataFrame()
    allp = pd.concat(parts)
    return allp.groupby("ym", as_index=False)[["long", "short", "spread", "long_net", "turnover"]].mean()
