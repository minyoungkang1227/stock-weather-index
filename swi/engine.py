"""2층: AI 기상·주가 분석 엔진

(1) 기상 민감도: 종목별 회귀계수를 업종 평균 쪽으로 당겨 안정화(부분 풀링, 경험적 베이즈).
    종목당 분기 표본이 30~40개뿐이라 종목 단독 계수는 들쭉날쭉합니다. 추정오차가 큰 종목일수록 업종 계수에 가깝게 둡니다.
(2) 예보 변화: inputs.expected_quarter 의 '이번 분기 예상 날씨 - 작년 같은 분기'
(3) 수익률 학습: 다음 달 시장 대비 초과수익률이 양(+)일 확률. 로지스틱 회귀와 그래디언트 부스팅을
    매월 워크포워드(그 시점까지 정답이 확정된 자료로만 학습)로 돌리고, 표본외 앞부분에서 고른 모델을 뒷부분에서 평가합니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCALE = {"HDD": 100.0, "CDD": 10.0, "HOT": 1.0, "TAVG": 1.0, "COLD": 1.0, "RAIN": 100.0}


# ---------- (1) 기상 민감도 ----------
def known_quarter(date: pd.Timestamp, lag_days: int) -> tuple[int, int]:
    """date 시점에 실적이 확정돼 있다고 볼 수 있는 가장 최근 분기 (연, 분기)."""
    t = date - pd.Timedelta(days=lag_days)
    qe = t + pd.offsets.QuarterEnd(0)  # t가 속한 분기의 말일
    if qe > t:
        qe = qe - pd.offsets.QuarterEnd(1)
    return qe.year, qe.quarter


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    n, k = X.shape
    if n <= k + 2 or np.linalg.matrix_rank(X) < k:
        return None
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    resid = y - X @ b
    s2 = resid @ resid / (n - k)
    return b, np.sqrt(np.diag(XtX_inv) * s2)


def _xcols(d: pd.DataFrame, index: str) -> list[np.ndarray]:
    x = d[f"{index}_yoy"].values / SCALE[index]
    if index == "TAVG":
        return [x * (d["quarter"].values == k) for k in (1, 2, 3, 4)]
    return [x]


def sensitivities(panel: pd.DataFrame, index: str, min_obs: int, y: str = "rev_g") -> dict:
    """업종 하나의 패널(code, quarter, y, {index}_yoy)로 종목별 계수를 추정해 부분 풀링.
    반환 {code: (b[4], se[4])}  — 분기별 계수(기온은 분기마다 다르고, 나머지는 네 분기 같음)."""
    d = panel.dropna(subset=[y, f"{index}_yoy"])
    if d.empty:
        return {}
    codes = sorted(d["code"].unique())
    qd = [(d["quarter"].values == k).astype(float) for k in (2, 3, 4)]
    xs = _xcols(d, index)
    # 업종 공통 계수: 종목 더미 + 분기 더미 + x
    fd = [(d["code"].values == c).astype(float) for c in codes]
    g = _ols(np.column_stack(fd + qd + xs), d[y].values)
    if g is None:
        return {}
    nx = len(xs)
    bg, seg = g[0][-nx:], g[1][-nx:]
    firm = {}
    for c in codes:
        di = d[d["code"] == c]
        if len(di) < min_obs:
            continue
        Xi = np.column_stack([np.ones(len(di))] + [(di["quarter"].values == k).astype(float) for k in (2, 3, 4)]
                             + _xcols(di, index))
        r = _ols(Xi, di[y].values)
        if r is not None:
            firm[c] = (r[0][-nx:], r[1][-nx:])
    # 경험적 베이즈: τ² = (종목 계수의 분산 - 평균 추정오차²), 가중치 w = τ²/(τ²+se²)
    if len(firm) >= 2:
        B = np.array([v[0] for v in firm.values()])
        S = np.array([v[1] for v in firm.values()])
        tau2 = np.maximum(B.var(axis=0, ddof=1) - (S ** 2).mean(axis=0), 0.0)
    else:
        tau2 = np.zeros(nx)
    out = {}
    for c in codes:
        if c in firm:
            bi, si = firm[c]
            w = np.where(tau2 > 0, tau2 / (tau2 + si ** 2), 0.0)
        else:
            bi, si, w = bg, seg, np.zeros(nx)
        b = w * bi + (1 - w) * bg
        se = np.sqrt(w * si ** 2 + (1 - w) ** 2 * seg ** 2)
        if nx == 1:
            b, se = np.repeat(b, 4), np.repeat(se, 4)
        out[c] = (b, se)
    return out


# ---------- (2) 날씨 수혜 기대값과 신뢰도 ----------
def benefit_and_confidence(b, se, dW, s) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """수혜 = β·ΔW. β~N(b,se²), ΔW~N(dW,s²) 독립일 때 곱의 분산 = b²s² + dW²se² + se²s².
    신뢰도 = 수혜의 부호가 맞을 확률을 0~100으로: 100·(2Φ(|z|)−1)."""
    b, se, dW, s = map(np.asarray, (b, se, dW, s))
    ben = b * dW
    var = b ** 2 * s ** 2 + dW ** 2 * se ** 2 + se ** 2 * s ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(var > 0, ben / np.sqrt(var), 0.0)
    conf = 100 * (2 * stats.norm.cdf(np.abs(z)) - 1)
    return ben, np.sqrt(var), conf


# ---------- (3) 수익률 학습 ----------
def make_model(kind: str):
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=150,
                                              min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    raise ValueError(kind)


def walk_forward(df: pd.DataFrame, features: list[str], kind: str, min_train: int, refit_every: int,
                 date_col: str = "ym", target: str = "up") -> pd.Series:
    """월별 워크포워드 예측. 시점 t의 예측에는 t-1월까지의 행(정답이 t월 말에 확정)만 학습에 씁니다."""
    d = df.dropna(subset=features + [target]).sort_values(date_col)       # 학습용 (정답 확정된 행)
    dates = sorted(df.dropna(subset=features)[date_col].unique())         # 예측할 달 (정답 미확정인 최신 달 포함)
    pred = pd.Series(np.nan, index=df.index)
    model, last_fit = None, None
    for i, t in enumerate(dates):
        if (d[date_col] < t).sum() == 0 or d.loc[d[date_col] < t, date_col].nunique() < min_train:
            continue
        if model is None or (i - last_fit) >= refit_every:
            tr = d[d[date_col] < t]
            if tr[target].nunique() < 2:
                continue
            model = make_model(kind).fit(tr[features].values, tr[target].values)
            last_fit = i
        te = df[(df[date_col] == t)].dropna(subset=features)
        if len(te):
            pred.loc[te.index] = model.predict_proba(te[features].values)[:, 1]
    return pred


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    ok = ~np.isnan(p) & ~np.isnan(y)
    y, p = y[ok], np.clip(p[ok], 1e-6, 1 - 1e-6)
    if len(y) < 30 or len(np.unique(y)) < 2:
        return dict(n=int(len(y)), auc=np.nan, logloss=np.nan, logloss_base=np.nan, brier=np.nan, hit=np.nan)
    base = np.full_like(p, y.mean())
    return dict(n=int(len(y)), auc=roc_auc_score(y, p), logloss=log_loss(y, p), logloss_base=log_loss(y, base),
                brier=brier_score_loss(y, p), hit=float(((p > 0.5) == y).mean()))


def auc_diff_bootstrap(df: pd.DataFrame, p1: str, p2: str, reps: int = 300, seed: int = 0) -> tuple[float, float, float]:
    """월 단위로 묶어 재표집한 AUC 차이(p1 − p2)의 점추정과 90% 구간."""
    d = df.dropna(subset=[p1, p2, "up"])
    months = d["ym"].unique()
    rng = np.random.default_rng(seed)
    base = roc_auc_score(d["up"], d[p1]) - roc_auc_score(d["up"], d[p2])
    diffs = []
    groups = {m: g for m, g in d.groupby("ym")}
    for _ in range(reps):
        s = pd.concat([groups[m] for m in rng.choice(months, len(months))])
        if s["up"].nunique() < 2:
            continue
        diffs.append(roc_auc_score(s["up"], s[p1]) - roc_auc_score(s["up"], s[p2]))
    lo, hi = np.quantile(diffs, [0.05, 0.95]) if diffs else (np.nan, np.nan)
    return float(base), float(lo), float(hi)
