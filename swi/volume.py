"""개선 1: 도시가스 판매량.

도시가스 매출은 원료비(LNG) 연동 요금 변화에 크게 좌우되어 날씨 효과를 가립니다.
날씨가 직접 움직이는 것은 '얼마나 썼는가(판매량)'이므로, 지역 판매량 성장률로 민감도를 추정합니다.

입력 data/raw/citygas_volume.csv (직접 내려받아 열 이름만 맞춤):
  region (서울/경기/인천/대구/울산 등), ym (YYYY-MM), volume (단위 무관, 같은 지역은 같은 단위)
  용도별 자료라면 use 열(예: 주택용)을 두고 config 에서 고를 수 있게 남겨둡니다.
한계: 지역 합계라 그 지역의 다른 공급사 물량도 섞입니다(예: 서울은 여러 회사가 나눠 공급).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def quarterly_volume_growth(v: pd.DataFrame, use: str | None = None) -> pd.DataFrame:
    """지역×분기 판매량과 전년 동분기 대비 로그성장률. 분기 3개월이 다 있어야 계산."""
    d = v.copy()
    if use is not None and "use" in d:
        d = d[d["use"] == use]
    d["ym"] = pd.PeriodIndex(d["ym"].astype(str), freq="M")
    d["year"], d["quarter"] = d["ym"].dt.year, d["ym"].dt.quarter
    g = d.groupby(["region", "year", "quarter"])["volume"].agg(["sum", "count"]).reset_index()
    g = g[g["count"] == 3].rename(columns={"sum": "vol"})
    ly = g[["region", "year", "quarter", "vol"]].assign(year=g["year"] + 1).rename(columns={"vol": "vol_ly"})
    g = g.merge(ly, on=["region", "year", "quarter"], how="left")
    g["vol_g"] = np.where((g["vol"] > 0) & (g["vol_ly"] > 0), np.log(g["vol"] / g["vol_ly"]), np.nan)
    return g[["region", "year", "quarter", "vol_g"]]


def attach(fin_p: pd.DataFrame, vg: pd.DataFrame | None, region_map: dict, enabled: bool) -> pd.DataFrame:
    """민감도 추정에 쓸 결과변수 y_sens 와 그 출처 y_src 를 붙입니다.
    판매량이 있는 도시가스 종목은 판매량 성장률, 나머지는 기존 매출 성장률."""
    f = fin_p.copy()
    f["y_sens"], f["y_src"] = f["rev_g"], "매출"
    if not enabled or vg is None or vg.empty:
        return f
    f["region"] = f["code"].map(region_map)
    f = f.merge(vg, on=["region", "year", "quarter"], how="left")
    use_vol = f["region"].notna() & f["code"].isin(
        [c for c, r in region_map.items() if vg[vg["region"] == r]["vol_g"].notna().sum() >= 8])
    f.loc[use_vol, "y_sens"] = f.loc[use_vol, "vol_g"]
    f.loc[use_vol, "y_src"] = "판매량"
    return f
