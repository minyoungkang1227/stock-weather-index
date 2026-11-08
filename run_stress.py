"""날씨 리스크 스트레스 테스트.
필요: data/raw/weather_daily.csv, data/raw/fin_quarterly.csv, output/results.csv(run_analysis.py 결과)
사용법: python run_stress.py        결과: output/stress_report.md, output/stress_*.csv"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import config as C
from core import indices, stress, weather_sim
from run_analysis import load, stop, weather_panel

EOK = 1e8  # 억원


def firm_series(sim: dict, stns: list, stn, index: str) -> np.ndarray:
    """[n_paths, 4] 기업이 쓰는 지점(또는 전국평균)의 분기 지수."""
    if stn == "NATIONAL":
        idx = [stns.index(s) for s in C.NATIONAL if s in stns]
        return sim[index][:, idx, :].mean(axis=1)
    return sim[index][:, stns.index(int(stn)), :]


def base_financials(fin: pd.DataFrame) -> dict:
    """기업별 가장 최근 '4개 분기가 다 있는' 연도의 분기 매출과 연간 영업이익."""
    out = {}
    for code, g in fin.groupby("code"):
        full = g.dropna(subset=["revenue", "op_income"]).groupby("year").filter(lambda x: len(x) == 4)
        if full.empty:
            continue
        y = full["year"].max()
        r = full[full["year"] == y].sort_values("q")
        out[code] = dict(year=int(y), rev=r["revenue"].values, op_annual=float(r["op_income"].sum()))
    return out


def simulate_impacts(sim, stns, W_exp, betas, base, rng) -> tuple[dict, dict]:
    imps, meta = {}, {}
    for code, (name, group, stn, _) in C.FIRMS.items():
        index = C.STRESS_SPEC.get(group)
        if code not in base or group not in betas or index is None:
            continue
        W = firm_series(sim, stns, stn, index)
        imps[code] = stress.firm_impacts(W, W_exp[code], base[code]["rev"], betas[group], index, rng)
        meta[code] = (name, group)
    return imps, meta


def metrics_table(imps, meta, base) -> pd.DataFrame:
    rows = []
    for code, imp in imps.items():
        name, group = meta[code]
        for kind in ("revenue", "op_income"):
            m = stress.risk_metrics(imp[kind])
            denom = base[code]["rev"].sum() if kind == "revenue" else base[code]["op_annual"]
            row = dict(code=code, name=name, group=group, kind=kind, base_year=base[code]["year"])
            for k, v in m.items():
                row[f"{k}_억"] = v / EOK
                row[f"{k}_%"] = 100 * v / denom if denom > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate(imps, meta, base) -> pd.DataFrame:
    """업종 합산(원)과 동일가중 포트폴리오(기업별 영업이익 대비 %의 평균)."""
    rows = []
    groups = sorted({g for _, g in meta.values()})
    for g in groups + ["ALL"]:
        codes = [c for c in imps if g == "ALL" or meta[c][1] == g]
        tot = sum(imps[c]["op_income"] for c in codes)
        m = stress.risk_metrics(tot)
        rows.append(dict(scope=g, kind="업종 합산 영업이익(억원)", **{k: v / EOK for k, v in m.items()}))
        pct = [imps[c]["op_income"] / base[c]["op_annual"] * 100 for c in codes if base[c]["op_annual"] > 0]
        if pct:
            m = stress.risk_metrics(np.mean(pct, axis=0))
            rows.append(dict(scope=g, kind="동일가중 포트폴리오 이익 변화(%)", **m))
    return pd.DataFrame(rows)


def historical_replay(wq: pd.DataFrame, betas, base) -> pd.DataFrame:
    """과거 각 연도의 '평년 대비 날씨 편차'를 지금의 실적 규모에 그대로 적용."""
    rows = []
    for code, (name, group, stn, _) in C.FIRMS.items():
        index = C.STRESS_SPEC.get(group)
        if code not in base or group not in betas:
            continue
        s = wq[wq["stn"].astype(str) == str(stn)]
        for y, g in s.groupby("year"):
            g = g.set_index("q").reindex([1, 2, 3, 4])
            anom = g[f"{index}_anom"].values
            if np.isnan(anom).any():
                continue
            W = anom[None, :]
            imp = stress.firm_impacts(W, np.zeros(4), base[code]["rev"], betas[group], index, None)
            rows.append(dict(year=int(y), code=code, group=group, op_억=imp["op_income"][0] / EOK,
                             rev_억=imp["revenue"][0] / EOK))
    h = pd.DataFrame(rows)
    if h.empty:
        return h
    piv = h.pivot_table(index="year", columns="group", values="op_억", aggfunc="sum")
    piv["합계"] = piv.sum(axis=1)
    return piv.sort_values("합계")


def validate_generator(w: pd.DataFrame) -> pd.DataFrame:
    """VALIDATION_SPLIT까지 자료로만 추정한 생성기가 이후 실제 분기 지수를 90% 구간에 담는지 확인."""
    train = w[w["date"].dt.year <= C.VALIDATION_SPLIT]
    params, corr, stns, corrA = weather_sim.fit_all(train)
    real = indices.quarterly(indices.daily_indices(w, C.BASE_TEMP, C.HEATWAVE_TMAX))
    last_full = int(w["date"].max().year) - (0 if w["date"].max().month == 12 and w["date"].max().day == 31 else 1)
    rows = []
    for y in range(C.VALIDATION_SPLIT + 1, last_full + 1):
      for model, qc in (("OU만", False), ("OU+분기충격", True)):
        sim = weather_sim.simulate_year(params, corr, stns, y, n_paths=1000, seed=y,
                                        base=C.BASE_TEMP, hot=C.HEATWAVE_TMAX, corrA=corrA,
                                        quarter_component=qc)
        for index in ("HDD", "CDD", "TAVG", "HOT"):
            for j, s in enumerate(stns):
                r = real[(real["stn"] == s) & (real["year"] == y)].set_index("q")[index].reindex([1, 2, 3, 4]).values
                for q in range(4):
                    if index == "HDD" and q == 2 or index in ("CDD", "HOT") and q in (0, 3):
                        continue  # 값이 거의 항상 0인 계절은 제외
                    x, n = stress.coverage(np.array([r[q]]), sim[index][:, j, q][:, None])
                    rows.append(dict(model=model, index=index, year=y, stn=s, q=q + 1, out=x, n=n))
    v = pd.DataFrame(rows)
    res = []
    for (model, index), g in v.groupby(["model", "index"]):
        n, x = int(g["n"].sum()), int(g["out"].sum())
        lr, p = stress.kupiec(n, x, 0.10)
        res.append(dict(model=model, index=index, n=n, outside_90=x, rate=x / n if n else np.nan, expected=0.10,
                        kupiec_LR=lr, kupiec_p=p))
    return pd.DataFrame(res)


def main() -> None:
    os.makedirs(C.OUTPUT, exist_ok=True)
    w = load("weather_daily")
    fin = load("fin_quarterly", dtype={"code": str})
    res_path = f"{C.OUTPUT}/results.csv"
    if w is None or fin is None or not os.path.exists(res_path):
        stop("weather_daily.csv, fin_quarterly.csv, output/results.csv 가 모두 필요합니다 (run_analysis.py 먼저).")
    w["date"] = pd.to_datetime(w["date"])
    results = pd.read_csv(res_path)
    betas = stress.load_betas(results, C.STRESS_SPEC)
    base = base_financials(fin)

    print("[1] 기온 모형 추정")
    params, corr, stns, corrA = weather_sim.fit_all(w)
    par = pd.DataFrame([dict(stn=s, phi=params[s]["phi"], kappa=(-np.log(params[s]["phi"]) if params[s]["phi"] > 0 else np.nan),
                             trend_per_decade=params[s]["beta"][1] * 10,
                             tau_Q1=params[s]["tau"][0], tau_Q2=params[s]["tau"][1],
                             tau_Q3=params[s]["tau"][2], tau_Q4=params[s]["tau"][3],
                             rho_quarter=params[s]["rho"]) for s in stns])

    print(f"[2] {C.STRESS_YEAR}년 몬테카를로 {C.STRESS_PATHS:,}회")
    sim = weather_sim.simulate_year(params, corr, stns, C.STRESS_YEAR, C.STRESS_PATHS, C.STRESS_SEED,
                                    base=C.BASE_TEMP, hot=C.HEATWAVE_TMAX, corrA=corrA)
    W_exp = {}
    for code, (name, group, stn, _) in C.FIRMS.items():
        index = C.STRESS_SPEC.get(group)
        if index:
            W_exp[code] = firm_series(sim, stns, stn, index).mean(axis=0)
    rng = np.random.default_rng(C.STRESS_SEED) if C.BETA_UNCERTAINTY else None
    imps, meta = simulate_impacts(sim, stns, W_exp, betas, base, rng)
    firm_tab = metrics_table(imps, meta, base)
    agg_tab = aggregate(imps, meta, base)

    print("[3] 기후 이동 시나리오")
    shift_rows = []
    for sh in C.CLIMATE_SHIFTS:
        s2 = weather_sim.simulate_year(params, corr, stns, C.STRESS_YEAR, C.STRESS_PATHS, C.STRESS_SEED + 1,
                                       shift=sh, base=C.BASE_TEMP, hot=C.HEATWAVE_TMAX, corrA=corrA)
        rng2 = np.random.default_rng(C.STRESS_SEED + 1) if C.BETA_UNCERTAINTY else None
        i2, m2 = simulate_impacts(s2, stns, W_exp, betas, base, rng2)  # 기대 날씨는 기준 시나리오 그대로
        a2 = aggregate(i2, m2, base)
        a2.insert(0, "shift_C", sh)
        shift_rows.append(a2)

    print("[4] 과거 날씨 재현")
    wq = weather_panel(w.copy())
    hist = historical_replay(wq, betas, base)

    print("[5] 생성기 표본외 검증")
    val = validate_generator(w)

    firm_tab.to_csv(f"{C.OUTPUT}/stress_firms.csv", index=False, encoding="utf-8-sig")
    agg_tab.to_csv(f"{C.OUTPUT}/stress_aggregate.csv", index=False, encoding="utf-8-sig")
    hist.to_csv(f"{C.OUTPUT}/stress_historical.csv", encoding="utf-8-sig")
    val.to_csv(f"{C.OUTPUT}/stress_validation.csv", index=False, encoding="utf-8-sig")
    used = results[results["year_fe"].astype(str).isin(["False", "false", "0"])]
    used = used[used.apply(lambda r: stress.parse_x(r["x"])[0] == C.STRESS_SPEC.get(r["group"]), axis=1)]
    used = used[used["y"].isin(["rev_g", "op_chg"])][["group", "y", "x", "coef", "se", "t", "p_bh", "nobs"]]
    write_report(par, firm_tab, agg_tab, pd.concat(shift_rows) if shift_rows else pd.DataFrame(), hist, val, used)
    print("완료: output/stress_report.md")


def write_report(par, firm_tab, agg_tab, shift_tab, hist, val, used) -> None:
    r = lambda d: d.round(3).to_markdown(index=False)
    L = ["# 날씨 리스크 스트레스 테스트", "",
         f"- 대상 연도 {C.STRESS_YEAR}, 몬테카를로 {C.STRESS_PATHS:,}회, 계수 추정오차 반영: {C.BETA_UNCERTAINTY}",
         "- 기준: 사업계획이 '기대 날씨(시뮬레이션 평균)'를 가정했을 때 대비 실적 차이",
         "- 손실 기준 VaR·ES (양수 = 손실). %는 기준연도 연간 매출·영업이익 대비", "",
         "## 1. 기온 모형 모수",
         "phi: 하루 뒤 기온 편차가 남는 비율(평균회귀 속도 kappa = −ln phi), trend: 10년당 ℃,",
         "tau_Qk: 분기 전체가 평년보다 덥거나 추운 정도의 표준편차(℃), rho_quarter: 그 편차가 다음 분기로 이어지는 정도", "",
         r(par), "",
         "## 2-0. 사용한 민감도 계수 (run_analysis.py 결과, p_bh가 크면 그 업종 결과는 참고만)", "", r(used), "",
         "## 2. 기업별 날씨 VaR", "", r(firm_tab[firm_tab["kind"] == "op_income"].drop(columns="kind")), "",
         "### 매출", "", r(firm_tab[firm_tab["kind"] == "revenue"].drop(columns="kind")), "",
         "## 3. 업종 합산·포트폴리오", "", r(agg_tab), ""]
    if not shift_tab.empty:
        L += ["## 4. 기후 이동 시나리오 (기대 날씨는 현재 기준 그대로)", "", r(shift_tab), ""]
    if not hist.empty:
        L += ["## 5. 과거 날씨 재현: 영업이익 영향이 가장 컸던 해 (억원)", "", hist.head(8).round(1).to_markdown(), ""]
    L += ["## 6. 기온 생성기 표본외 검증",
          f"{C.VALIDATION_SPLIT}년까지로 추정해 이후 실제 분기 지수가 90% 구간 밖으로 나간 비율. 10%에 가까워야 하며 kupiec_p < 0.05면 기각.", "",
          r(val), "",
          "## 해석 시 주의",
          "- 민감도 계수는 과거 관계입니다. 요금제·사업구조가 바뀌면 그대로 적용되지 않습니다.",
          "- 여러 지점·분기를 함께 세어 검증 표본이 서로 독립이 아니므로 Kupiec p값은 다소 낙관적입니다.",
          "- 최고기온은 평균기온에 과거 '최고−평균' 차이를 더해 만들므로, 폭염일 꼬리를 과소평가할 수 있습니다(검증표의 HOT 행 확인)."]
    open(f"{C.OUTPUT}/stress_report.md", "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
