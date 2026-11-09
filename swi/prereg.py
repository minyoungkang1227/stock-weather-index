"""사전 등록한 판정 규칙 (config.PREREG_*). 결과를 보기 전에 정한 단 하나의 기준으로 성공/실패를 가립니다.

왜 따로 두나: 분석은 업종 × 지표 × 결과변수 × 모형(연도더미 유무) 조합으로 수십 개의 회귀를 돌립니다.
그중 잘 나온 것을 골라 '효과 있음'이라 하면 우연과 구별할 수 없습니다(다중비교·사후선택).
그래서 업종마다 검정 하나를 미리 지정하고, 그 4개에만 다중검정 보정을 적용해 판정합니다.
나머지 회귀는 '탐색 결과'로만 보고합니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def stage1(res: pd.DataFrame, spec: dict, bh_q: float = 0.10, min_t: float | None = 2.0,
           p_override: dict | None = None) -> pd.DataFrame:
    """res: run_analysis 의 회귀표(group, y, x, year_fe, coef, se, t, p, nobs ...).
    업종마다 사전 지정 검정 1개를 골라, 부호 방향 단측 p → 4개에 BH 보정 → 통과 여부.
    p_override = {업종: 순열검정 단측 p} 를 주면 DK t 대신 그 p 를 쓴다 (min_t=None 이면 |t| 조건 없음)."""
    rows = []
    for g, sp in spec.items():
        r = res[(res["group"] == g) & (res["y"] == sp["y"]) & (res["x"] == sp["x"])
                & (res["year_fe"].astype(bool) == bool(sp["year_fe"]))]
        if r.empty:
            rows.append(dict(group=g, **sp, coef=np.nan, se=np.nan, t=np.nan, nobs=np.nan, p_one=np.nan,
                             note="회귀 결과 없음(자료 부족)"))
            continue
        r = r.iloc[0]
        t = float(r["t"])
        # Driscoll-Kraay 는 시점 간 상관을 감안한 오차라 유효 표본은 '분기 수'에 가깝다 → 자유도 = 분기 수 - 1 (보수적)
        dof = max(int(r["n_periods"]) - 1, 1) if pd.notna(r.get("n_periods")) else 30
        p_dk = float(stats.t.sf(t * sp["sign"], dof))   # 기대 부호 방향 단측 (DK, 참고)
        p_one = float(p_override[g]) if p_override is not None and g in p_override else p_dk
        rows.append(dict(group=g, **sp, coef=float(r["coef"]), se=float(r["se"]), t=t, nobs=r.get("nobs"),
                         p_dk=p_dk, p_one=p_one, method="perm" if p_override is not None else "DK", note=""))
    out = pd.DataFrame(rows)
    ok = out["p_one"].notna()
    out["q_bh"] = np.nan
    if ok.any():
        out.loc[ok, "q_bh"] = multipletests(out.loc[ok, "p_one"].values, method="fdr_bh")[1]
    out["sign_ok"] = np.sign(out["coef"]) == out["sign"]
    t_ok = (out["t"].abs() >= min_t) if min_t is not None else True
    out["pass"] = ok & out["sign_ok"] & t_ok & (out["q_bh"] <= bh_q)
    return out


def stage2(d_test: pd.DataFrame, passed_groups: list[str], spec: dict, long_short_within, summarize,
           cost_bps: float) -> dict:
    """1차 통과 업종만 남겨 SWI 업종 내 롱숏(평가구간)의 NW t 로 판정."""
    if not passed_groups:
        return dict(verdict="해당 없음", reason="1차(날씨→실적)를 통과한 업종이 없어 2차 판정을 하지 않습니다.",
                    groups=[], months=0, spread_mean_pct=np.nan, spread_t_NW=np.nan)
    sub = d_test[d_test["group"].isin(passed_groups)]
    r = summarize(long_short_within(sub, spec["score"], cost_bps=cost_bps), spec["score"])
    t = r.get("spread_t_NW", np.nan)
    ok = bool(np.isfinite(t) and t >= spec["min_t"])
    return dict(verdict="통과" if ok else "미통과",
                reason=f"{spec['ranking']} 롱숏({spec['score']}, {spec['window']}) NW t = {t:.2f} (기준 ≥ {spec['min_t']})"
                if np.isfinite(t) else "평가구간 표본 부족",
                groups=passed_groups, months=r.get("months", 0),
                spread_mean_pct=r.get("spread_mean_pct", np.nan), spread_t_NW=t)


def stage1_markdown(s1: pd.DataFrame, bh_q: float, min_t: float, prereg_date: str) -> list[str]:
    cols = [c for c in ["group", "x", "sign", "coef", "t", "p_dk", "p_one", "q_bh", "sign_ok", "pass", "note"] if c in s1]
    show = s1[cols].copy()
    perm = "method" in s1 and (s1["method"] == "perm").any()
    rule = (f"통과 = 연도 순열검정 단측 p 의 BH q ≤ {bh_q}, 부호 일치. t·p_dk(Driscoll-Kraay)는 참고용"
            if perm else f"통과 = q ≤ {bh_q}, |t| ≥ {min_t}, 부호 일치")
    return [f"## 사전 등록 판정 — 1차: 날씨 → 매출 (등록일 {prereg_date})", "",
            f"업종마다 미리 정한 검정 1개, 기대 부호 방향 단측 p → 4개 BH 보정. {rule}.",
            "이 표 밖의 회귀 결과는 탐색용이며 '효과 있음'의 근거로 쓰지 않습니다.", "",
            show.round(4).to_markdown(index=False), ""]


# ---------- 연도 순열검정 (Driscoll-Kraay 대안) ----------
def _design(d: pd.DataFrame, cols: list[str], year_fe: bool) -> np.ndarray:
    parts = [d[cols].to_numpy(float),
             pd.get_dummies(d["code"], drop_first=False).to_numpy(float),          # 기업 고정효과
             pd.get_dummies(d["q"], drop_first=True).to_numpy(float)]              # 분기더미
    if year_fe:
        parts.append(pd.get_dummies(d["year"], drop_first=True).to_numpy(float))
    return np.column_stack(parts)


def _coef(yv: np.ndarray, X: np.ndarray) -> float:
    return float(np.linalg.lstsq(X, yv, rcond=None)[0][0])


def perm_pvalue(g: pd.DataFrame, y: str, xs: list[str], x_test: str, controls: list[str], sign: int,
                year_fe: bool = True, B: int = 999, seed: int = 0) -> dict:
    """날씨 변수(xs 전부)를 '연도 단위로 통째로' 다른 해의 값으로 바꿔 끼워 계수의 귀무분포를 만든다.

    왜: 업종당 시점이 약 40분기·기업 4~6개뿐이라 Driscoll-Kraay 표준오차가 실제 변동보다 작게 나와
    가짜 날씨로도 '유의'가 자주 나온다(귀무 시뮬레이션에서 t의 표준편차 1.6~1.9, 이론 1).
    연도를 섞으면 그 해 안의 분기 패턴·지점 간 관계·기업 간 상관은 그대로 두고 '날씨와 실적의 시점 대응'만 끊는다.
    p(단측) = (1 + #{부호 맞춘 순열 계수 >= 실제}) / (1 + B).
    """
    cols = [x_test] + [x for x in xs if x != x_test] + controls
    d = g.dropna(subset=[y] + xs + controls).copy()
    years = np.array(sorted(d["year"].unique()))
    yv = d[y].to_numpy(float)
    b0 = _coef(yv, _design(d, cols, year_fe))
    # (기업, 연도, 분기) -> 날씨값. 순열 해의 같은 기업·분기 값을 가져옴 (없으면 그 순열은 건너뜀)
    # 날씨값 조회표는 y 결측으로 빠진 행까지 포함한 원래 패널에서 만든다 (날씨는 그 해에도 있으므로)
    look = g.dropna(subset=xs).drop_duplicates(["code", "year", "q"]).set_index(["code", "year", "q"])[xs]
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(B):
        perm = dict(zip(years, rng.permutation(years)))
        idx = pd.MultiIndex.from_arrays([d["code"], d["year"].map(perm), d["q"]])
        xv = look.reindex(idx).to_numpy(float)
        if np.isnan(xv).any():
            continue
        dp = d.copy()
        dp[xs] = xv
        null.append(_coef(yv, _design(dp, cols, year_fe)))
    null = np.asarray(null)
    if len(null) < 0.9 * B:
        raise ValueError(f"순열 {B}회 중 {len(null)}회만 계산됨 — 날씨값이 빠진 해가 있습니다. 자료를 확인하세요.")
    p = (1 + np.sum(null * sign >= b0 * sign)) / (1 + len(null))
    return dict(coef=b0, p_perm=float(p), B_used=len(null), null_sd=float(null.std()) if len(null) else np.nan)
