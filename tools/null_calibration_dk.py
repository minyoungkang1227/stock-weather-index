"""귀무 보정 점검: 실제 실적·주가 + '무작위 합성 날씨'(실제 아님)로 1차 판정을 여러 번 → 거짓 통과율."""
import sys, time
import numpy as np, pandas as pd
import config as C
import run_analysis as A
from core import panel, regress
from swi import prereg

fin = panel.fin_growth(A.load("fin_quarterly", dtype={"code": str}))
tar = A.tariff_control()
d = pd.date_range("1985-01-01", "2025-12-31"); doy = d.dayofyear.values
N = int(sys.argv[1]); out = []
t0 = time.time()
for seed in range(N):
    rng = np.random.default_rng(1000 + seed)
    # 연 단위·계절 단위로 지속되는 이상기온(실제 날씨처럼 자기상관) + 지점 간 상관
    common = np.repeat(rng.normal(0, 1.5, len(d) // 30 + 1), 30)[:len(d)]
    rows = []
    for stn in C.STATIONS:
        t = 12.5 - 14 * np.cos(2 * np.pi * (doy - 15) / 365) + common + rng.normal(0, 2, len(d))
        rows.append(pd.DataFrame({"stn": stn, "date": d, "tavg": t, "tmin": t - 5, "tmax": t + 5, "rain": 0.0}))
    w = pd.concat(rows)
    wq = A.weather_panel(w)
    p = panel.build_panel(panel.firm_weather(wq, C.FIRMS), fin, None)
    if tar is not None:
        p = p.merge(tar, on=["year", "q"], how="left")
    p = p[(p["year"] >= C.ANALYSIS_START_YEAR) & (p["year"] <= C.ANALYSIS_END_YEAR)]
    rows = []
    for g, xs, ctrl in A.specs(p):
        sp = C.PREREG_STAGE1[g]
        if sp["x"] not in xs:
            continue
        r = regress.panel_reg(p[p["group"] == g], "rev_g", list(xs), ctrl, year_dummies=True)
        rows.append(r.assign(group=g))
    s1 = prereg.stage1(pd.concat(rows), C.PREREG_STAGE1, C.PREREG_STAGE1_BH_Q, C.PREREG_STAGE1_MIN_T)
    out.append(s1.assign(seed=seed)[["seed", "group", "t", "p_one", "pass"]])
r = pd.concat(out)
print(f"{N}회, {time.time()-t0:.0f}초 (합성 날씨 — 실제 시장 결과 아님)")
print("업종별 단측 p<0.05 비율(이론 0.05):"); print(r.assign(rej=r.p_one < 0.05).groupby("group").rej.mean().round(3).to_string())
print("업종별 통과율:"); print(r.groupby("group")["pass"].mean().round(3).to_string())
print("한 업종이라도 통과한 비율:", r.groupby("seed")["pass"].any().mean().round(3))
print("|t| 표준편차(이론 ~1):"); print(r.groupby("group").t.std().round(2).to_string())
