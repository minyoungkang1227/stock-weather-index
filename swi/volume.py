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


REGIONS_KR = ["서울", "인천", "경기", "부산", "대구", "광주", "대전", "울산", "세종", "강원", "충북", "충남",
              "전북", "전남", "경북", "경남", "제주"]


def read_kogas_wide(path: str) -> pd.DataFrame:
    """공공데이터포털 '한국가스공사_월별 시도별 도시가스 판매현황'(15040819) CSV → region, ym, volume(천㎥).
    원본은 '연월' 한 열 + 시도별 열(가로형). 인코딩은 cp949/utf-8 둘 다 시도."""
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            raw = pd.read_csv(path, encoding=enc, thousands=",")
            break
        except UnicodeDecodeError:
            continue
    raw.columns = [str(c).strip() for c in raw.columns]
    ym_col = next((c for c in raw.columns if "연월" in c or c.lower() in ("ym", "date")), raw.columns[0])
    regions = [c for c in raw.columns if c in REGIONS_KR]
    if len(regions) < 2:
        raise ValueError(f"시도 열을 찾지 못했습니다: {list(raw.columns)[:20]}")
    ym = raw[ym_col].astype(str).str.replace(r"[^0-9]", "", regex=True).str[:6]
    if not ym.str.fullmatch(r"\d{6}").all():
        raise ValueError(f"연월 형식이 예상과 다릅니다: {raw[ym_col].head(3).tolist()}")
    long = raw[regions].assign(ym=ym.str[:4] + "-" + ym.str[4:]).melt(id_vars="ym", var_name="region", value_name="volume")
    long["volume"] = pd.to_numeric(long["volume"], errors="coerce")
    return long[["region", "ym", "volume"]]


def region_panel(vg: pd.DataFrame, wq: pd.DataFrame, region_stn: dict) -> pd.DataFrame:
    """지역×분기 판매량 성장률 + 그 지역 관측소의 전년동기 대비 HDD 변화 (회귀·순열검정용, code=지역)."""
    w = wq[["stn", "year", "q", "HDD_yoy"]].copy()
    w["stn"] = w["stn"].astype(str)
    m = pd.DataFrame([dict(region=r, stn=str(s)) for r, s in region_stn.items()])
    d = vg.rename(columns={"quarter": "q"}).merge(m, on="region").merge(w, on=["stn", "year", "q"], how="left")
    d["HDD_yoy_100"] = d["HDD_yoy"] / 100
    d["code"] = d["region"]
    return d


def volume_quality(vg: pd.DataFrame, regions: list[str], y0: int, y1: int, tol: float = 0.15) -> dict:
    """판매량 자료 점검 — 날씨와 무관한 3분기(난방 없음) 전년비로 '날씨 밖의 충격'을 찾는다.
    common_breaks: 여러 지역이 같은 해 3분기에 함께 tol 넘게 움직인 해 (통계 기준 변경·가격 충격 의심)
    volatile_regions: 3분기 전년비 표준편차가 tol 을 넘는 지역 (산업·발전용 비중이 커서 난방 신호가 묻힘 의심)"""
    q3 = vg[(vg["quarter"] == 3) & vg["region"].isin(regions) & vg["year"].between(y0, y1)]
    t = q3.pivot_table(index="year", columns="region", values="vol_g")
    med = t.median(axis=1)
    return dict(q3_table=t, common_breaks={int(y): float(m) for y, m in med.items() if abs(m) > tol},
                volatile_regions={r: float(sd) for r, sd in t.std().items() if sd > tol})
