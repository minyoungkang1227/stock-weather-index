"""업종 넓히기 (config.PREREG_SECTOR_VOLUME): 가격이 섞이지 않은 '물량' 자료로 날씨 영향을 검정.

자료 읽기는 원본 형식별로 따로 두고(수집과 분석 분리), 검정은 모두 같은 틀:
  물량의 전년 대비 로그성장률 ~ 전국(6개 도시 평균) Δ날씨 + 분기더미, 연도 순열검정 단측.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _read(path: str, **kw) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 읽을 수 없습니다: {path}")


def read_epsis_monthly(path: str) -> pd.DataFrame:
    """EPSIS 월별 판매전력량(가로형: 연도, 1월..12월, MWh) → item='합계', ym, volume."""
    raw = _read(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    ycol = raw.columns[0]
    months = [c for c in raw.columns[1:] if str(c).replace("월", "").isdigit()]
    if len(months) != 12:
        raise ValueError(f"1~12월 열을 찾지 못했습니다: {list(raw.columns)}")
    long = raw.melt(id_vars=ycol, value_vars=months, var_name="m", value_name="volume")
    long["year"] = pd.to_numeric(long[ycol], errors="coerce")
    long = long.dropna(subset=["year"])
    long["m"] = long["m"].str.replace("월", "").astype(int)
    long["ym"] = long["year"].astype(int).astype(str) + "-" + long["m"].map("{:02d}".format)
    long["volume"] = pd.to_numeric(long["volume"], errors="coerce")
    return long.assign(item="합계")[["item", "ym", "volume"]].sort_values("ym").reset_index(drop=True)


def read_epsis_annual_by_use(path: str) -> pd.DataFrame:
    """EPSIS 용도별 판매전력량(연, 머리글 3줄: 대분류/중분류/점유율) → item, year, volume.
    '점유율' 열은 쓰지 않음 (2024~2025년 점유율 값이 실제 비율과 맞지 않음 — 확인됨)."""
    raw = _read(path, header=None)
    h0, h1 = raw.iloc[0].ffill(), raw.iloc[1]
    body = raw.iloc[3:].copy()
    out = []
    for j in range(1, raw.shape[1]):
        name = str(h1.iloc[j]) if pd.notna(h1.iloc[j]) else str(h0.iloc[j])
        if name in ("점유율", "nan") or (pd.notna(raw.iloc[2, j]) and str(raw.iloc[2, j]) == "점유율"):
            continue
        out.append(pd.DataFrame({"item": name, "year": pd.to_numeric(body[0], errors="coerce"),
                                 "volume": pd.to_numeric(body[j], errors="coerce")}))
    d = pd.concat(out).dropna(subset=["year"])
    d["year"] = d["year"].astype(int)
    return d.drop_duplicates(["item", "year"])


def quarterly_growth(v: pd.DataFrame) -> pd.DataFrame:
    """item, ym, volume (월) → item, year, q, vol_g (분기 합의 전년동기 로그성장률, 3개월 다 있을 때만)."""
    d = v.dropna(subset=["volume"]).copy()
    d["year"] = d["ym"].str[:4].astype(int)
    d["q"] = (d["ym"].str[5:7].astype(int) - 1) // 3 + 1
    g = d.groupby(["item", "year", "q"])["volume"].agg(["sum", "count"]).reset_index()
    g = g[g["count"] == 3].rename(columns={"sum": "vol"})
    ly = g[["item", "year", "q", "vol"]].assign(year=g["year"] + 1).rename(columns={"vol": "vol_ly"})
    g = g.merge(ly, on=["item", "year", "q"], how="left")
    g["vol_g"] = np.where((g["vol"] > 0) & (g["vol_ly"] > 0), np.log(g["vol"] / g["vol_ly"]), np.nan)
    return g[["item", "year", "q", "vol_g"]]


def annual_growth(v: pd.DataFrame) -> pd.DataFrame:
    d = v.sort_values(["item", "year"]).copy()
    d["vol_ly"] = d.groupby("item")["volume"].shift(1)
    ok = (d["year"].diff() == 1) & (d["vol_ly"] > 0) & (d["volume"] > 0)
    d["vol_g"] = np.where(ok, np.log(d["volume"] / d["vol_ly"]), np.nan)
    return d.assign(q=1)[["item", "year", "q", "vol_g"]]


def national_weather(wq: pd.DataFrame, annual: bool = False) -> pd.DataFrame:
    """전국(6개 도시 평균) 분기 HDD·CDD 와 전년 대비 변화(100도일 단위). annual=True 면 연 합계 기준."""
    n = wq[wq["stn"].astype(str) == "NATIONAL"][["year", "q", "HDD", "CDD"]].copy()
    if annual:
        n = n.groupby("year")[["HDD", "CDD"]].sum(min_count=4).reset_index().assign(q=1)
    n = n.sort_values(["q", "year"])
    for c in ("HDD", "CDD"):
        prev = n.groupby("q")[c].shift(1)
        same = n.groupby("q")["year"].diff() == 1
        n[f"{c}_yoy_100"] = np.where(same, (n[c] - prev) / 100, np.nan)
    return n[["year", "q", "HDD_yoy_100", "CDD_yoy_100"]]


def placebo_check(g: pd.DataFrame, q: int | None, tol: float, y0: int, y1: int) -> dict:
    """날씨 영향이 거의 없는 대조 분기(q)의 전년비가 tol 을 넘는 해 = 날씨 밖 충격 의심. q=None(연간)이면 전체 연도."""
    sub = g[g["year"].between(y0, y1)]
    if q is not None:
        sub = sub[sub["q"] == q]
    bad = sub[sub["vol_g"].abs() > tol]
    return {int(r.year): float(r.vol_g) for r in bad.itertuples()}
