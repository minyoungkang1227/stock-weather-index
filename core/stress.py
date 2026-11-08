"""날씨 스트레스 테스트: 날씨 시나리오 × 업종 민감도 -> 기업·업종·포트폴리오 실적 영향.

영향 계산 (분기 q, 기업 i):
  ΔW = 시나리오 날씨 - 기대 날씨(시뮬레이션 평균)          ← 사업계획이 '평년 날씨'를 가정한다는 기준
  매출 영향   = 기준매출_q × (exp(β_rev × ΔW/단위) − 1)       ← 회귀가 로그 성장률이므로
  영업이익 영향 = β_op × ΔW/단위 × 기준매출_q                  ← 회귀가 '영업이익 변화 / 전년 매출'이므로
손실 = −영향. VaR(α) = 손실 분포의 α 분위수, ES(α) = VaR을 넘는 손실의 평균.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

SCALE = {"HDD": 100.0, "CDD": 10.0, "HOT": 1.0, "TAVG": 1.0}


def parse_x(x: str) -> tuple[str, int | None]:
    """회귀 변수명 -> (지표, 분기 또는 None).  예: 'TAVG_yoy_Q4' -> ('TAVG', 4)"""
    m = re.match(r"(HDD|CDD|HOT|TAVG)_yoy(?:_\d+)?(?:_Q(\d))?$", x)
    if not m:
        raise ValueError(f"알 수 없는 변수명: {x}")
    return m.group(1), (int(m.group(2)) if m.group(2) else None)


def load_betas(results: pd.DataFrame, spec: dict) -> dict:
    """results.csv(연도더미 없는 기본 모형)에서 업종별 계수를 꺼냅니다.
    반환: {group: {"rev_g"/"op_chg": {분기(1~4): (지표, coef, se)}}}"""
    r = results[results["year_fe"].astype(str).isin(["False", "false", "0"])]
    out = {}
    for group, index in spec.items():
        g = r[r["group"] == group]
        for y in ("rev_g", "op_chg"):
            per_q = {}
            for _, row in g[g["y"] == y].iterrows():
                idx, q = parse_x(row["x"])
                if idx != index:
                    continue
                for qq in ([q] if q else [1, 2, 3, 4]):
                    per_q[qq] = (idx, float(row["coef"]), float(row["se"]))
            if per_q:
                out.setdefault(group, {})[y] = per_q
    return out


def firm_impacts(W: np.ndarray, W_exp: np.ndarray, base_rev: np.ndarray, betas: dict,
                 index: str, rng: np.random.Generator | None) -> dict[str, np.ndarray]:
    """한 기업의 경로별 연간 영향(원).
    W: [n_paths, 4] 시나리오 날씨, W_exp: [4] 기대 날씨, base_rev: [4] 분기 기준매출."""
    n = W.shape[0]
    dW = (W - W_exp[None, :]) / SCALE[index]
    out = {}
    for y in ("rev_g", "op_chg"):
        imp = np.zeros(n)
        for q in range(4):
            if (q + 1) not in betas.get(y, {}):
                continue
            _, b, se = betas[y][q + 1]
            bq = b + se * rng.standard_normal(n) if rng is not None else np.full(n, b)
            if y == "rev_g":
                imp += base_rev[q] * (np.exp(bq * dW[:, q]) - 1)
            else:
                imp += bq * dW[:, q] * base_rev[q]
        out["revenue" if y == "rev_g" else "op_income"] = imp
    return out


def risk_metrics(impact: np.ndarray, levels=(0.95, 0.99)) -> dict:
    """영향(+는 이익) -> 손실 기준 VaR·ES. 결과는 '손실 금액'(양수일수록 큰 손실)."""
    loss = -np.asarray(impact)
    out = {"mean": float(np.mean(impact)), "sd": float(np.std(impact))}
    for a in levels:
        var = float(np.quantile(loss, a))
        tail = loss[loss >= var]
        out[f"VaR{int(a*100)}"] = var
        out[f"ES{int(a*100)}"] = float(tail.mean())
    return out


def kupiec(n: int, x: int, p: float) -> tuple[float, float]:
    """Kupiec 비율검정(POF). 기대 이탈확률 p에서 n번 중 x번 이탈. (LR, p값)"""
    if n == 0:
        return float("nan"), float("nan")
    phat = x / n
    def ll(q):
        a = (n - x) * np.log(1 - q) if x < n else 0.0
        b = x * np.log(q) if x > 0 else 0.0
        return a + b
    lr = -2 * (ll(p) - ll(phat)) if 0 < phat < 1 else -2 * (ll(p) - 0.0)
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def coverage(realized: np.ndarray, sims: np.ndarray, lo: float = 0.05, hi: float = 0.95) -> tuple[int, int]:
    """실제값이 시뮬레이션 [lo, hi] 분위 구간 밖에 떨어진 횟수. realized[k], sims[:, k]."""
    ql = np.quantile(sims, lo, axis=0)
    qh = np.quantile(sims, hi, axis=0)
    ok = ~np.isnan(realized)
    out = ((realized < ql) | (realized > qh)) & ok
    return int(out.sum()), int(ok.sum())
