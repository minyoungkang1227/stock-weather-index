"""연도 순열검정의 귀무 보정 점검 (합성 날씨 — 실제 시장 결과 아님)."""
import sys, time
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests
import config as C, run_analysis as A
from core import panel
from swi import prereg
fin = panel.fin_growth(A.load("fin_quarterly", dtype={"code": str})); tar = A.tariff_control()
d = pd.date_range("1985-01-01", "2025-12-31"); doy = d.dayofyear.values
N, B = int(sys.argv[1]), int(sys.argv[2]); out = []; t0 = time.time()
for seed in range(N):
    rng = np.random.default_rng(1000 + seed)
    common = np.repeat(rng.normal(0, 1.5, len(d) // 30 + 1), 30)[:len(d)]
    w = pd.concat([pd.DataFrame({"stn": s, "date": d, "tavg": (t := 12.5 - 14*np.cos(2*np.pi*(doy-15)/365) + common + rng.normal(0, 2, len(d))),
                                 "tmin": t - 5, "tmax": t + 5, "rain": 0.0}) for s in C.STATIONS])
    p = panel.build_panel(panel.firm_weather(A.weather_panel(w), C.FIRMS), fin, None)
    if tar is not None: p = p.merge(tar, on=["year", "q"], how="left")
    p = p[(p.year >= C.ANALYSIS_START_YEAR) & (p.year <= C.ANALYSIS_END_YEAR)]
    ps = {}
    for g, xs, ctrl in A.specs(p):
        sp = C.PREREG_STAGE1[g]
        if sp["x"] in xs:
            r = prereg.perm_pvalue(p[p.group == g], "rev_g", list(xs), sp["x"], ctrl, sp["sign"], True, B, seed)
            ps[g] = r["p_perm"]
    q = multipletests(list(ps.values()), method="fdr_bh")[1]
    for (g, pv), qq in zip(ps.items(), q):
        out.append(dict(seed=seed, group=g, p=pv, q=qq))
r = pd.DataFrame(out)
print(f"{N}회 x B={B}, {time.time()-t0:.0f}초 (합성 날씨 — 실제 시장 결과 아님)")
print("업종별 p<0.05 비율(이론 0.05):"); print(r.assign(x=r.p < 0.05).groupby("group").x.mean().round(3).to_string())
print("업종별 p<0.10 비율(이론 0.10):"); print(r.assign(x=r.p < 0.10).groupby("group").x.mean().round(3).to_string())
print("한 업종이라도 q<=0.10:", r.groupby("seed").q.min().le(0.10).mean().round(3))
