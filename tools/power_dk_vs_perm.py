"""검정력 점검: 합성 날씨에 '심어 둔 효과'를 넣었을 때 두 방식이 얼마나 잡아내나 (합성 — 실제 시장 결과 아님).
음료 매출 성장률에 ΔCDD(10도일)당 +b 만큼 인위적으로 더함."""
import sys, time
import numpy as np, pandas as pd
import config as C, run_analysis as A
from core import panel, regress
from swi import prereg
fin = panel.fin_growth(A.load("fin_quarterly", dtype={"code": str}))
d = pd.date_range("1985-01-01", "2025-12-31"); doy = d.dayofyear.values
N, B, b = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]); out = []
for seed in range(N):
    rng = np.random.default_rng(5000 + seed)
    common = np.repeat(rng.normal(0, 1.5, len(d) // 30 + 1), 30)[:len(d)]
    w = pd.concat([pd.DataFrame({"stn": s, "date": d, "tavg": (t := 12.5 - 14*np.cos(2*np.pi*(doy-15)/365) + common + rng.normal(0, 2, len(d))),
                                 "tmin": t - 5, "tmax": t + 5, "rain": 0.0}) for s in C.STATIONS])
    p = panel.build_panel(panel.firm_weather(A.weather_panel(w), C.FIRMS), fin, None)
    p = p[(p.year >= C.ANALYSIS_START_YEAR) & (p.year <= C.ANALYSIS_END_YEAR)]
    xs = [x for g, xs, _ in A.specs(p) if g == "beverage" for x in xs][:1]
    g = p[p.group == "beverage"].copy()
    g["rev_g"] = g["rev_g"] + b * g["CDD_yoy_10"]
    r = regress.panel_reg(g, "rev_g", xs, [], year_dummies=True)
    pp = prereg.perm_pvalue(g, "rev_g", xs, xs[0], [], +1, True, B, seed)
    out.append(dict(t=float(r.t.iloc[0]), p_perm=pp["p_perm"]))
r = pd.DataFrame(out)
print(f"b={b}: DK t>=2 비율 {np.mean(r.t >= 2):.2f} / 순열 p<0.05 비율 {np.mean(r.p_perm < 0.05):.2f}  (N={N})")
