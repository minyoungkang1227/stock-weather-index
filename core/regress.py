"""패널 회귀와 표본외 검증.

기본 모형 (업종 그룹별):
  y_it = a_i + 분기더미 + b * ΔW_it (+ 통제변수) + e_it
  y: 전년 동분기 대비 매출 성장률 / 영업이익 변화 / 초과수익률
  ΔW: 전년 동분기 대비 날씨지수 변화 (결과변수가 전년 대비이므로 날씨도 전년 대비로 맞춤)
표준오차: Driscoll-Kraay. 같은 업종 기업들은 같은 날씨 충격을 공유해 오차가 서로 상관되는데,
기업 수가 적어(4~6개) 기업 클러스터 표준오차는 믿기 어렵기 때문에 시간축 상관을 허용하는 방식을 씁니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from statsmodels.stats.multitest import multipletests


def _prep(df: pd.DataFrame, y: str, xs: list[str], controls: list[str],
          quarter_dummies: bool, year_dummies: bool) -> tuple[pd.DataFrame, list[str]]:
    d = df.dropna(subset=[y] + xs + controls).copy()
    d["period"] = pd.to_datetime(dict(year=d["year"], month=3 * d["q"] - 2, day=1))
    cols = xs + controls
    if quarter_dummies:
        for k in (2, 3, 4):
            d[f"Q{k}"] = (d["q"] == k).astype(float)
            cols.append(f"Q{k}")
    if year_dummies:
        for yv in sorted(d["year"].unique())[1:]:
            d[f"Y{yv}"] = (d["year"] == yv).astype(float)
            cols.append(f"Y{yv}")
    d = d.set_index(["code", "period"])
    return d, cols


def panel_reg(df: pd.DataFrame, y: str, xs: list[str], controls: list[str] | None = None,
              quarter_dummies: bool = True, year_dummies: bool = False) -> pd.DataFrame:
    controls = controls or []
    d, cols = _prep(df, y, xs, controls, quarter_dummies, year_dummies)
    if len(d) < 20 or d.index.get_level_values(0).nunique() < 2:
        return pd.DataFrame()
    mod = PanelOLS(d[y], d[cols], entity_effects=True, drop_absorbed=True, check_rank=False)
    res = mod.fit(cov_type="kernel", kernel="bartlett")
    rows = []
    for x in xs:
        if x not in res.params.index:
            continue
        rows.append(dict(y=y, x=x, coef=res.params[x], se=res.std_errors[x], t=res.tstats[x],
                         p=res.pvalues[x], nobs=int(res.nobs),
                         n_firms=int(d.index.get_level_values(0).nunique()),
                         n_periods=int(d.index.get_level_values(1).nunique()),
                         r2_within=float(res.rsquared_within),
                         year_fe=year_dummies))
    return pd.DataFrame(rows)


def _ols_dummies(train: pd.DataFrame, test: pd.DataFrame, y: str, xs: list[str]) -> np.ndarray:
    """기업 고정효과 + 분기더미 + xs를 학습구간에서 추정해 검증구간을 예측."""
    def design(d: pd.DataFrame, firms: list[str]) -> np.ndarray:
        parts = [(d["code"].to_numpy(dtype=object)[:, None] == np.array(firms, dtype=object)[None, :]).astype(float)]
        parts.append(np.column_stack([(d["q"] == k).astype(float) for k in (2, 3, 4)]))
        if xs:
            parts.append(d[xs].values)
        return np.hstack(parts)
    firms = sorted(train["code"].unique())
    beta, *_ = np.linalg.lstsq(design(train, firms), train[y].values, rcond=None)
    return design(test, firms) @ beta


def oos_gain(df: pd.DataFrame, y: str, xs: list[str], split_year: int) -> dict:
    """표본외 R² 개선: 1 - MSE(날씨 포함) / MSE(날씨 없음). 0보다 커야 날씨가 예측에 도움."""
    d = df.dropna(subset=[y] + xs)
    train, test = d[d["year"] <= split_year], d[d["year"] > split_year]
    test = test[test["code"].isin(train["code"].unique())]
    if len(train) < 20 or len(test) < 8:
        return dict(y=y, x="+".join(xs), oos_gain=np.nan, n_train=len(train), n_test=len(test))
    base = _ols_dummies(train, test, y, [])
    full = _ols_dummies(train, test, y, xs)
    mse_b = np.mean((test[y].values - base) ** 2)
    mse_f = np.mean((test[y].values - full) ** 2)
    return dict(y=y, x="+".join(xs), oos_gain=1 - mse_f / mse_b, n_train=len(train), n_test=len(test))


def add_multiple_testing(tab: pd.DataFrame) -> pd.DataFrame:
    """여러 업종·지표·결과변수를 한꺼번에 검정하므로 Benjamini-Hochberg로 보정한 p값을 함께 둡니다."""
    if tab.empty:
        return tab
    tab = tab.copy()
    tab["p_bh"] = multipletests(tab["p"].values, method="fdr_bh")[1]
    return tab
