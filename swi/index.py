"""3층: Stock Weather Index (종목별 0~100)

- 기상수혜도 B: 날씨 때문에 기대되는 매출 성장률 변화(β·ΔW)를 과거 분포 대비 정규점수로 → 100·Φ(수혜/규모). 50 = 영향 없음
- 상승확률 P : 2층 모델이 낸 '다음 달 시장 대비 상승' 확률 × 100
- 신뢰도   C : 수혜 방향이 맞을 확률(민감도 추정오차 + 남은 기간 날씨 불확실성 반영) 0~100
- 종합 SWI  : 50 + [wB(B−50) + wP(P−50)] × (0.5 + 0.5·C/100)
  신뢰도가 낮으면 점수를 50 쪽으로 당겨 '확신 없는 신호'를 약하게 만듭니다.
  이 결합 방식은 설계상 선택이므로, 4층 백테스트에서 B·P·SWI 각각의 성과를 따로 확인합니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def benefit_score(df: pd.DataFrame, default_scale: float = 0.02, min_obs: int = 24,
                  by: str | None = None) -> pd.Series:
    """규모(분모)는 그 시점 '이전' 달들의 수혜 값 표준편차 (미래 분포를 쓰지 않음).
    by='group' 이면 업종마다 따로 계산 — 업종별로 수혜 크기의 단위·폭이 달라 한 잣대로 비교하지 않기 위함."""
    if by is not None:
        out = pd.Series(np.nan, index=df.index)
        for _, g in df.groupby(by):
            out.loc[g.index] = benefit_score(g, default_scale, min_obs, None)
        return out
    d = df.sort_values("ym")
    out = pd.Series(np.nan, index=df.index)
    hist = []
    for t, g in d.groupby("ym", sort=True):
        sc = np.std(hist) if len(hist) >= min_obs else default_scale
        sc = sc if sc > 0 else default_scale
        out.loc[g.index] = 100 * stats.norm.cdf(g["benefit"].values / sc)
        hist.extend(g["benefit"].dropna().tolist())
    return out


def combine(B, P, C, wB: float = 0.5, wP: float = 0.5) -> np.ndarray:
    B, P, C = (np.asarray(x, dtype=float) for x in (B, P, C))
    P = np.where(np.isnan(P), 50.0, P)
    return 50 + (wB * (B - 50) + wP * (P - 50)) * (0.5 + 0.5 * C / 100)
