"""수집한 원자료로 날씨지수를 만들고 업종별 회귀·표본외 검증을 합니다.
사용법: python run_analysis.py   (결과: output/results.csv, output/report.md)"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

import config as C
from core import indices, panel, quality, regress
from swi import prereg

R = C.RAW


def load(name: str, **kw) -> pd.DataFrame | None:
    """DB(config.DB_PATH)에 같은 이름의 테이블이 있으면 DB에서, 없으면 기존처럼 data/raw/*.csv 에서 읽습니다."""
    df = None
    if os.path.exists(C.DB_PATH):
        from system import db
        df = db.read(C.DB_PATH, name)
        if df is not None:
            for c in kw.get("parse_dates", []) or []:
                df[c] = pd.to_datetime(df[c])
    if df is None:
        path = f"{R}/{name}.csv"
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, **kw)
    if "code" in df:  # 엑셀로 열었다 저장하면 앞자리 0이 사라지므로 6자리로 복원
        df["code"] = df["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    return df


def stop(msg: str) -> None:
    print(f"\n[중단] {msg}\n데이터를 확인한 뒤 다시 실행하세요.")
    sys.exit(1)


def weather_panel(w: pd.DataFrame) -> pd.DataFrame:
    w["date"] = pd.to_datetime(w["date"])
    out = quality.weather_outliers(w)
    if len(out):
        out.to_csv(f"{C.OUTPUT}/weather_outliers.csv", index=False, encoding="utf-8-sig")
        stop(f"기상 이상치 {len(out)}건 (output/weather_outliers.csv)")
    stale = quality.weather_freshness(w, f"{C.ANALYSIS_END_YEAR}-12-01")
    if len(stale):
        print(f"[경고] 분석 종료 시점까지 자료가 없는 지점:\n{stale}")
    d = indices.daily_indices(w, C.BASE_TEMP, C.HEATWAVE_TMAX)
    q = indices.quarterly(d)
    q = pd.concat([q, indices.national_average(q, C.NATIONAL)], ignore_index=True)
    q["stn"] = q["stn"].astype(str)
    return indices.add_normals(q, C.NORMAL_YEARS, C.MIN_NORMAL_YEARS)


def tariff_control() -> pd.DataFrame | None:
    m = load("cpi_citygas", parse_dates=["month"])
    if m is None:
        return None
    m["year"], m["q"] = m["month"].dt.year, m["month"].dt.quarter
    t = m.groupby(["year", "q"])["value"].mean().reset_index()
    ly = t.assign(year=t["year"] + 1).rename(columns={"value": "ly"})
    t = t.merge(ly, on=["year", "q"])
    t["tariff_g"] = np.log(t["value"] / t["ly"])
    return t[["year", "q", "tariff_g"]]


def specs(p: pd.DataFrame) -> list[tuple[str, list[str], list[str]]]:
    p["HDD_yoy_100"] = p["HDD_yoy"] / 100
    p["CDD_yoy_10"] = p["CDD_yoy"] / 10
    for k in (1, 2, 3, 4):
        p[f"TAVG_yoy_Q{k}"] = p["TAVG_yoy"] * (p["q"] == k)
    tq = [f"TAVG_yoy_Q{k}" for k in (1, 2, 3, 4)]
    gas_ctrl = ["tariff_g"] if "tariff_g" in p and p["tariff_g"].notna().any() else []
    return [("gas", ["HDD_yoy_100"], gas_ctrl),
            ("beverage", ["CDD_yoy_10"], []),
            ("beverage", ["HOT_yoy"], []),
            ("apparel", tq, []),
            ("retail", tq, [])]


def main(split_year: int = 2021) -> None:
    os.makedirs(C.OUTPUT, exist_ok=True)
    w = load("weather_daily")
    if w is None:
        stop("data/raw/weather_daily.csv 가 없습니다. run_collect.py weather 를 먼저 실행하세요.")
    wq = weather_panel(w)
    wq.to_csv(f"{C.PROCESSED}/weather_quarterly.csv", index=False, encoding="utf-8-sig")
    fw = panel.firm_weather(wq, C.FIRMS)

    fin = load("fin_quarterly", dtype={"code": str})
    if fin is not None:
        bad = quality.fin_checks(fin)
        if len(bad["nonpositive_revenue"]):
            stop(f"매출이 0 이하인 분기 {len(bad['nonpositive_revenue'])}건")
        fin = panel.fin_growth(fin)
    px, mkt = load("prices", dtype={"code": str}, parse_dates=["date"]), load("market", parse_dates=["date"])
    rets = panel.quarterly_returns(px, mkt) if px is not None and mkt is not None else None

    p = panel.build_panel(fw, fin, rets)
    t = tariff_control()
    if t is not None:
        p = p.merge(t, on=["year", "q"], how="left")
    p = p[(p["year"] >= C.ANALYSIS_START_YEAR) & (p["year"] <= C.ANALYSIS_END_YEAR)]
    p.to_csv(f"{C.PROCESSED}/panel.csv", index=False, encoding="utf-8-sig")

    outcomes = [y for y in ("rev_g", "op_chg", "xret", "xret_next") if y in p]
    rows, oos = [], []
    for group, xs, ctrl in specs(p):
        g = p[p["group"] == group]
        for y in outcomes:
            for yfe in (False, True):
                r = regress.panel_reg(g, y, list(xs), ctrl, year_dummies=yfe)
                if not r.empty:
                    r.insert(0, "group", group)
                    rows.append(r)
            o = regress.oos_gain(g, y, list(xs), split_year)
            o["group"] = group
            oos.append(o)
    res = regress.add_multiple_testing(pd.concat(rows, ignore_index=True)) if rows else pd.DataFrame()
    oos = pd.DataFrame(oos)
    res.to_csv(f"{C.OUTPUT}/results.csv", index=False, encoding="utf-8-sig")
    oos.to_csv(f"{C.OUTPUT}/oos.csv", index=False, encoding="utf-8-sig")
    s1 = None
    if not res.empty:
        if C.PREREG_STAGE1_METHOD == "perm":
            pp = {}
            for group, xs, ctrl in specs(p):
                sp = C.PREREG_STAGE1.get(group)
                if sp and sp["x"] in xs and group not in pp:
                    pp[group] = prereg.perm_pvalue(p[p["group"] == group], sp["y"], list(xs), sp["x"], ctrl, sp["sign"],
                                                   sp["year_fe"], C.PREREG_PERM_B, C.PREREG_PERM_SEED)["p_perm"]
            s1 = prereg.stage1(res, C.PREREG_STAGE1, C.PREREG_STAGE1_BH_Q, None, p_override=pp)
        else:
            s1 = prereg.stage1(res, C.PREREG_STAGE1, C.PREREG_STAGE1_BH_Q, C.PREREG_STAGE1_MIN_T)
        s1.to_csv(f"{C.OUTPUT}/prereg_stage1.csv", index=False, encoding="utf-8-sig")
        print(s1[["group", "x", "coef", "t", "p_one", "q_bh", "pass"]].round(3).to_string(index=False))
    gv = gas_volume_test(wq)
    write_report(res, oos, p, split_year, s1, gv)
    print("완료: output/results.csv, output/oos.csv, output/report.md")


def gas_volume_test(wq: pd.DataFrame) -> dict | None:
    """추가 사전 등록(config.PREREG_GAS_VOLUME): 지역 도시가스 판매량 ~ ΔHDD, 연도 순열검정."""
    from swi import volume
    sp = C.PREREG_GAS_VOLUME
    if not os.path.exists(sp["file"]):
        print(f"[판매량 검정] 건너뜀: {sp['file']} 없음")
        return None
    v = volume.read_kogas_wide(sp["file"])
    d = volume.region_panel(volume.quarterly_volume_growth(v), wq, sp["region_stn"])
    d = d[(d["year"] >= sp["start_year"]) & (d["year"] <= sp["end_year"])].dropna(subset=["vol_g", "HDD_yoy_100"])
    d.to_csv(f"{C.PROCESSED}/gas_volume_panel.csv", index=False, encoding="utf-8-sig")
    ref = regress.panel_reg(d, "vol_g", ["HDD_yoy_100"], [], year_dummies=sp["year_fe"])
    r = prereg.perm_pvalue(d, "vol_g", ["HDD_yoy_100"], "HDD_yoy_100", [], sp["sign"], sp["year_fe"],
                           C.PREREG_PERM_B, C.PREREG_PERM_SEED)
    out = dict(coef=r["coef"], t_DK=float(ref["t"].iloc[0]) if len(ref) else np.nan, p_perm=r["p_perm"],
               n=len(d), regions=d["region"].nunique(), years=f"{d['year'].min()}~{d['year'].max()}")
    out["pass"] = bool(out["p_perm"] < sp["alpha"] and np.sign(out["coef"]) == sp["sign"])
    pd.DataFrame([out]).to_csv(f"{C.OUTPUT}/prereg_gas_volume.csv", index=False, encoding="utf-8-sig")
    print(f"[판매량 검정] 계수 {out['coef']:.4f} (100도일당 로그성장률), 순열 p = {out['p_perm']:.3f}, "
          f"n={out['n']} ({out['regions']}개 지역, {out['years']}) → {'통과' if out['pass'] else '미통과'}")
    return out


def write_report(res: pd.DataFrame, oos: pd.DataFrame, p: pd.DataFrame, split_year: int, s1=None, gv=None) -> None:
    lines = ["# 날씨지수 분석 결과", "",
             f"- 기간 {C.ANALYSIS_START_YEAR}~{C.ANALYSIS_END_YEAR}, 기업 {p['code'].nunique()}개",
             "- 모형: 기업 고정효과 + 분기더미 (+연도더미 버전), Driscoll-Kraay 표준오차",
             "- 날씨 단위: HDD 100도일, CDD 10도일, 폭염 1일, 기온 1℃ (모두 전년 동분기 대비 변화)",
             "- p_bh: 전체 검정에 대한 Benjamini-Hochberg 보정 p값", ""]
    if C.EXCLUDED:
        lines += ["- 사전 규칙으로 제외: " + ", ".join(f"{c}({r})" for c, r in C.EXCLUDED.items())
                  + (" — 이번 실행은 SWI_INCLUDE_EXCLUDED=1 로 포함한 참고 버전" if len(C.FIRMS) == len(C.FIRMS_ALL) else ""), ""]
    if s1 is not None:
        lines += prereg.stage1_markdown(s1, C.PREREG_STAGE1_BH_Q, C.PREREG_STAGE1_MIN_T, C.PREREG_DATE)
    if gv is not None:
        lines += [f"## 추가 사전 등록 — 도시가스 판매량 ~ ΔHDD ({C.PREREG_GAS_VOLUME['date']}, 1차 미통과 뒤 추가)", "",
                  f"- 계수 {gv['coef']:.4f} (100도일당), DK t {gv['t_DK']:.2f}(참고), 연도 순열 단측 p = {gv['p_perm']:.3f}, "
                  f"n = {gv['n']} ({gv['regions']}개 지역, {gv['years']}) → **{'통과' if gv['pass'] else '미통과'}** (기준 p < 0.05)",
                  "- 1차 4개 업종이 모두 미통과한 뒤 추가한 검정입니다. 판매량 자료를 보기 전에 기준을 고정했습니다.", ""]
        lines += ["## 탐색 결과 (판정에 쓰지 않음)", ""]
    if not res.empty:
        show = res[["group", "y", "x", "year_fe", "coef", "se", "t", "p", "p_bh", "nobs", "n_firms"]].round(4)
        lines += ["## 회귀 결과", "", show.to_markdown(index=False), ""]
    if not oos.empty:
        lines += [f"## 표본외 검증 (학습 ~{split_year}, 검증 {split_year + 1}~)",
                  "oos_gain > 0 이면 날씨를 넣었을 때 표본외 예측 오차가 줄어든 것입니다.", "",
                  oos.round(4).to_markdown(index=False), ""]
    open(f"{C.OUTPUT}/report.md", "w", encoding="utf-8").write("\n".join(lines))


if __name__ == "__main__":
    main()
