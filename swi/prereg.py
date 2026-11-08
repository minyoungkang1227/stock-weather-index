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


def stage1(res: pd.DataFrame, spec: dict, bh_q: float = 0.10, min_t: float = 2.0) -> pd.DataFrame:
    """res: run_analysis 의 회귀표(group, y, x, year_fe, coef, se, t, p, nobs ...).
    업종마다 사전 지정 검정 1개를 골라, 부호 방향 단측 p → 4개에 BH 보정 → 통과 여부."""
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
        p_one = float(stats.t.sf(t * sp["sign"], dof))   # 기대 부호 방향 단측
        rows.append(dict(group=g, **sp, coef=float(r["coef"]), se=float(r["se"]), t=t, nobs=r.get("nobs"),
                         p_one=p_one, note=""))
    out = pd.DataFrame(rows)
    ok = out["p_one"].notna()
    out["q_bh"] = np.nan
    if ok.any():
        out.loc[ok, "q_bh"] = multipletests(out.loc[ok, "p_one"].values, method="fdr_bh")[1]
    out["sign_ok"] = np.sign(out["t"]) == out["sign"]
    out["pass"] = ok & out["sign_ok"] & (out["t"].abs() >= min_t) & (out["q_bh"] <= bh_q)
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
    show = s1[["group", "x", "sign", "coef", "t", "p_one", "q_bh", "sign_ok", "pass", "note"]].copy()
    return [f"## 사전 등록 판정 — 1차: 날씨 → 매출 (등록일 {prereg_date})", "",
            f"업종마다 미리 정한 검정 1개, 기대 부호 방향 단측 p → 4개 BH 보정. 통과 = q ≤ {bh_q}, |t| ≥ {min_t}, 부호 일치.",
            "이 표 밖의 회귀 결과는 탐색용이며 '효과 있음'의 근거로 쓰지 않습니다.", "",
            show.round(4).to_markdown(index=False), ""]
