"""Stock Weather Index — 아키텍처(기상·기업·금융 → AI 엔진 → 종목별 0~100 지수 → 투자 의사결정) 시제품.
필요: data/raw/weather_daily.csv, fin_quarterly.csv, prices.csv(거래량 포함), market.csv
사용법: python run_swi.py      결과: output/swi_report.md, output/swi_monthly.csv, output/swi_latest.csv"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import config as C
from core import panel
from run_analysis import load, stop
from swi import decision, engine, event, index, inputs, volume


def make_sens(fin_p):
    """(업종, 그 시점에 확정된 분기) -> 종목별 민감도. 같은 조건이면 다시 계산하지 않음."""
    cache: dict = {}

    def get(group: str, kq: tuple[int, int]) -> dict:
        ck = (group, kq)
        if ck not in cache:
            sub = fin_p[(fin_p["group"] == group) &
                        ((fin_p["year"] < kq[0]) | ((fin_p["year"] == kq[0]) & (fin_p["quarter"] <= kq[1])))]
            cache[ck] = engine.sensitivities(sub, C.STRESS_SPEC[group], C.MIN_FIRM_OBS, y="y_sens")
        return cache[ck]
    return get


def build_rows(e, fin_p, mp, get_sens) -> pd.DataFrame:
    """종목×월말 행: 날씨 기대·민감도(그 시점 공시 기준)·수혜·신뢰도·주가 특징."""
    src = fin_p.groupby("code")["y_src"].agg(lambda x: x.mode().iat[0] if len(x.dropna()) else "매출").to_dict()
    rows = []
    e_idx = e.set_index(["stn", "year", "month"])
    for code, (name, group, stn, _) in C.FIRMS.items():
        I = C.STRESS_SPEC.get(group)
        sc = engine.SCALE[I]
        g = mp[mp["code"] == code]
        for _, r in g.iterrows():
            y, mth = int(r["year"]), int(r["month"])
            if y < C.ANALYSIS_START_YEAR or y > C.ANALYSIS_END_YEAR:
                continue
            key = (str(stn), y, mth)
            if key not in e_idx.index:
                continue
            w = e_idx.loc[key]
            t = pd.Timestamp(year=y, month=mth, day=1) + pd.offsets.MonthEnd(0)
            kq = engine.known_quarter(t, C.DISCLOSURE_LAG_DAYS)
            s = get_sens(group, kq).get(code)
            q = int(w["quarter"])
            b, se = (s[0][q - 1], s[1][q - 1]) if s else (np.nan, np.nan)
            dW = (w[f"{I}_exp"] - w[f"{I}_ly"]) / sc
            sd = w[f"{I}_sd"] / sc
            ben, ben_sd, conf = engine.benefit_and_confidence(b, se, dW, sd)
            z = lambda v: (w[f"{v}_exp"] - w[f"{v}_norm"]) / w[f"{v}_qsd"] if w[f"{v}_qsd"] > 0 else 0.0
            rows.append(dict(code=code, name=name, group=group, y_src=src.get(code, "매출"),
                             ym=r["ym"], year=y, month=mth, quarter=q,
                             obs_share=w["obs_share"], beta=b, beta_se=se, beta_t=b / se if se and se > 0 else np.nan,
                             dW=dW, dW_sd=sd, benefit=float(ben), benefit_sd=float(ben_sd), C=float(conf),
                             anom_z=z(I), rain_z=z("RAIN"), cold_z=z("COLD"),
                             mom1=r["mom1"], mom3=r["mom3"], vol_chg=r["vol_chg"], fwd_xret=r["fwd_xret"]))
    d = pd.DataFrame(rows)
    d["up"] = np.where(d["fwd_xret"].isna(), np.nan, (d["fwd_xret"] > 0).astype(float))
    for gname in sorted(d["group"].unique())[1:]:
        d[f"g_{gname}"] = (d["group"] == gname).astype(float)
    return d


def prepare() -> dict:
    """원자료를 읽어 1층 입력과 민감도 계산기를 준비 (월별 백테스트와 일별 산출이 함께 씀)."""
    w, fin = load("weather_daily"), load("fin_quarterly", dtype={"code": str})
    px, mkt = load("prices", dtype={"code": str}), load("market")
    if any(x is None for x in (w, fin, px, mkt)):
        stop("weather_daily, fin_quarterly, prices, market 자료가 모두 필요합니다 (run_collect.py).")
    m = inputs.monthly_weather(w, C.BASE_TEMP, C.HEATWAVE_TMAX, C.COLD_TMIN)
    e = inputs.add_national(inputs.expected_quarter(m, C.NORMAL_YEARS, C.MIN_NORMAL_YEARS), C.NATIONAL)
    rq = inputs.realized_quarters(e)
    fg = panel.fin_growth(fin).rename(columns={"q": "quarter"})
    fmap = pd.DataFrame([dict(code=c, group=v[1], stn=str(v[2])) for c, v in C.FIRMS.items()])
    fin_p = fg.merge(fmap, on="code").merge(rq, on=["stn", "year", "quarter"], how="left")
    vraw = load("citygas_volume")
    vg = volume.quarterly_volume_growth(vraw) if vraw is not None else None
    fin_p = volume.attach(fin_p, vg, C.GAS_REGION, C.SWI_GAS_USE_VOLUME)
    return dict(w=w, fin=fin, px=px, mkt=mkt, e=e, fin_p=fin_p, get_sens=make_sens(fin_p),
                mp=inputs.month_end_prices(px, mkt))


WX_FEATS = ["benefit", "C", "beta_t", "anom_z", "rain_z", "cold_z", "obs_share"]
MKT_FEATS = ["mom1", "mom3", "vol_chg"]


def main() -> None:
    os.makedirs(C.OUTPUT, exist_ok=True)
    print("[1층] 입력 데이터 정리")
    P_ = prepare()
    e, fin_p, get_sens, mp, px, mkt = P_["e"], P_["fin_p"], P_["get_sens"], P_["mp"], P_["px"], P_["mkt"]

    print("[2층] 민감도(공시 시점 기준) · 수혜 · 신뢰도")
    d = build_rows(e, fin_p, mp, get_sens)
    d["B"] = index.benefit_score(d, by="group" if C.SWI_WITHIN_GROUP else None)

    print("[2층] 상승확률 모델 워크포워드 (로지스틱 vs GBM, 날씨 변수 유무)")
    gcols = [c for c in d.columns if c.startswith("g_")]
    mkt_feats = [f for f in MKT_FEATS if d[f].notna().any()]
    wx_feats = list(WX_FEATS)
    d[wx_feats] = d[wx_feats].fillna(0.0)
    sets = {"weather": wx_feats + mkt_feats + gcols, "base": mkt_feats + gcols}
    for kind in ("logit", "gbm"):
        for sname, feats in sets.items():
            d[f"p_{kind}_{sname}"] = engine.walk_forward(d, feats, kind, C.SWI_MIN_TRAIN_MONTHS, C.SWI_REFIT_EVERY)

    oos_months = sorted(d.loc[d["p_logit_weather"].notna(), "ym"].unique())
    if not oos_months:
        stop("표본외 예측 기간이 없습니다. 자료 기간을 늘리거나 SWI_MIN_TRAIN_MONTHS를 줄이세요.")
    sel_end = oos_months[min(C.SWI_SELECT_MONTHS, len(oos_months)) - 1]
    sel, test = d[d["ym"] <= sel_end], d[(d["ym"] > sel_end) & d["ym"].isin(oos_months)]
    evals = []
    for kind in ("logit", "gbm"):
        for sname in sets:
            col = f"p_{kind}_{sname}"
            for wname, part in (("선택구간", sel), ("평가구간", test)):
                evals.append(dict(model=kind, features=sname, window=wname, **engine.evaluate(part["up"].values, part[col].values)))
    ev = pd.DataFrame(evals)
    s_ll = ev[(ev["window"] == "선택구간") & (ev["features"] == "weather")].set_index("model")["logloss"]
    chosen = s_ll.idxmin() if s_ll.notna().any() else "logit"
    d["P"] = 100 * d[f"p_{chosen}_weather"]
    aucd = engine.auc_diff_bootstrap(test, f"p_{chosen}_weather", f"p_{chosen}_base") if len(test) else (np.nan,) * 3

    print("[3층] Stock Weather Index")
    d["SWI"] = index.combine(d["B"], d["P"], d["C"], C.SWI_WEIGHTS["benefit"], C.SWI_WEIGHTS["prob"])

    test = d[(d["ym"] > sel_end) & d["ym"].isin(oos_months)]  # P·SWI가 붙은 뒤 다시 자름

    print("[4층] 의사결정 지원")
    latest = decision.flags(d[d["ym"] == d["ym"].max()][["code", "name", "group", "ym", "B", "P", "C", "SWI",
                                                         "beta", "dW", "benefit"]], C.SWI_WATCH, C.SWI_WARN)
    bts = []
    for label, part in (("표본외 전체", d[d["ym"].isin(oos_months)]), ("평가구간", test)):
        for sc in ("B", "P", "SWI"):
            for how in ("업종 내", "전 종목"):
                fn = decision.long_short_within if how == "업종 내" else decision.long_short
                r = decision.summarize(fn(part, sc, cost_bps=C.COST_BPS_ROUNDTRIP), sc)
                r["window"], r["ranking"] = label, how
                bts.append(r)
    bt = pd.DataFrame(bts)
    per_group = []
    for gname, g in test.groupby("group"):
        for sc in ("B", "SWI"):
            r = decision.summarize(decision.long_short(g, sc, cost_bps=C.COST_BPS_ROUNDTRIP, min_n=3), sc)
            r["group"] = gname
            per_group.append(r)
    per_group = pd.DataFrame(per_group)
    main_rank = "업종 내" if C.SWI_WITHIN_GROUP else "전 종목"
    bt["main"] = bt["ranking"] == main_rank

    print("[검증] 공시일 이벤트 스터디")
    ev_tab, ev_verdict, ev_info = run_event_study(e, fin_p, px, mkt, get_sens)

    latest["ym"] = latest["ym"].astype(str)
    save_model(d, sets["weather"], chosen)
    ev.to_csv(f"{C.OUTPUT}/swi_model_eval.csv", index=False, encoding="utf-8-sig")
    bt.to_csv(f"{C.OUTPUT}/swi_backtest.csv", index=False, encoding="utf-8-sig")
    per_group.to_csv(f"{C.OUTPUT}/swi_backtest_by_group.csv", index=False, encoding="utf-8-sig")
    ev_tab.assign(verdict=ev_verdict, sources=str(ev_info)).to_csv(f"{C.OUTPUT}/swi_event_study.csv", index=False,
                                                                   encoding="utf-8-sig")
    d.to_csv(f"{C.OUTPUT}/swi_monthly.csv", index=False, encoding="utf-8-sig")
    latest.to_csv(f"{C.OUTPUT}/swi_latest.csv", index=False, encoding="utf-8-sig")
    write_report(d, ev, chosen, aucd, latest, bt, sel_end, oos_months, ev_tab, ev_verdict, ev_info, per_group)
    print("완료: output/swi_report.md")


def save_model(d: pd.DataFrame, feats: list[str], kind: str) -> None:
    """일별 산출용: 정답이 확정된 전체 기간으로 선택된 모델을 다시 학습해 저장하고, 업종별 수혜 규모도 저장."""
    import json

    import joblib
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    tr = d.dropna(subset=feats + ["up"])
    model = engine.make_model(kind).fit(tr[feats].values, tr["up"].values)
    joblib.dump(dict(model=model, features=feats, kind=kind, trained_through=str(tr["ym"].max()),
                     groups=sorted(d["group"].unique())), f"{C.MODEL_DIR}/p_model.joblib")
    scales = d.dropna(subset=["benefit"]).groupby("group")["benefit"].std().to_dict()
    scales["_all"] = float(d["benefit"].std())
    json.dump(scales, open(f"{C.MODEL_DIR}/b_scale.json", "w"))


def run_event_study(e, fin_p, px, mkt, get_sens):
    disc = load("disclosures", parse_dates=["date"])
    keys = fin_p.rename(columns={"quarter": "q"})
    keys = keys[(keys["year"] >= C.ANALYSIS_START_YEAR) & (keys["year"] <= C.ANALYSIS_END_YEAR)]
    ev = event.event_dates(keys, disc, C.EVENT_FALLBACK_DAYS)
    e_idx = e.set_index(["stn", "year", "month"])
    sur = []
    for _, r in ev.iterrows():
        name, group, stn, _ = C.FIRMS[r["code"]]
        I = C.STRESS_SPEC[group]
        key = (str(stn), int(r["year"]), int(3 * r["q"]))
        if key not in e_idx.index:
            sur.append(np.nan)
            continue
        w = e_idx.loc[key]
        dW = (w[f"{I}_act"] - w[f"{I}_ly"]) / engine.SCALE[I]
        s = get_sens(group, engine.known_quarter(r["date"], C.DISCLOSURE_LAG_DAYS)).get(r["code"])
        sur.append(s[0][int(r["q"]) - 1] * dW if s else np.nan)
    ev["surprise"] = sur
    ev["group"] = ev["code"].map(lambda c: C.FIRMS[c][1])
    ev["surprise_z"] = ev["surprise"] / ev.groupby("group")["surprise"].transform("std")
    ar = event.daily_abnormal(px, mkt)
    car = event.cars(ev, ar, C.EVENT_WINDOWS)
    car = car.merge(ev[["code", "year", "q", "group", "surprise_z"]], on=["code", "year", "q"])
    car["cluster"] = car["year"] * 10 + car["q"]
    tab = pd.DataFrame([event.regress(car, w) for w in C.EVENT_WINDOWS])
    names = list(C.EVENT_WINDOWS)
    info = ev["kind"].value_counts().to_dict()
    car.to_csv(f"{C.OUTPUT}/swi_event_cars.csv", index=False, encoding="utf-8-sig")
    return tab, event.verdict(tab, names[0], names[1:]), info


def write_report(d, ev, chosen, aucd, latest, bt, sel_end, oos_months, ev_tab=None, ev_verdict="", ev_info=None,
                 per_group=None) -> None:
    r = lambda x, k=3: x.round(k).to_markdown(index=False)
    L = ["# Stock Weather Index 시제품 결과", "",
         "## 아키텍처 대응", "",
         "| 구조도 단계 | 구현 내용 |", "|---|---|",
         "| 기상정보 (기온·강수·한파) | 관측소 8곳 일자료 → 난방·냉방도일, 폭염일, 한파일(최저 −12℃ 이하), 강수량. 월말마다 '관측된 달 + 남은 달 평년'으로 이번 분기 예상 |",
         "| 기업정보 (제품·지역·매출) | 업종(제품군), 공급 지역 관측소 매칭, DART 분기 매출·영업이익 (공시 시차 반영) |",
         "| 금융정보 (주가·거래량) | 월말 종가, 시장 대비 초과수익률, 1·3개월 모멘텀, 거래량 변화 |",
         "| AI 엔진: 기상 민감도 | 종목별 회귀계수를 업종 계수 쪽으로 부분 풀링 (그 시점까지 공시된 실적만 사용) |",
         "| AI 엔진: 예보 변화 | 이번 분기 예상 날씨 − 작년 같은 분기 (예보 자료를 쌓으면 '남은 달 평년' 자리에 대체) |",
         "| AI 엔진: 수익률 학습 | 로지스틱 vs GBM 월별 워크포워드, 표본외 앞부분에서 선택하고 뒷부분에서 평가 |",
         "| Stock Weather Index | 기상수혜도 B, 상승확률 P, 신뢰도 C, 종합 SWI (모두 0~100) |",
         "| 투자 의사결정 지원 | 관심종목·위험경고 목록, 점수별 상·하위 포트폴리오 백테스트 |", "",
         f"## 2층 모델 평가 (선택 모델: **{chosen}**)",
         f"표본외 {len(oos_months)}개월 중 {sel_end}까지 선택구간, 이후 평가구간. "
         "weather = 날씨 변수 포함, base = 주가·거래량만. logloss가 logloss_base(항상 평균확률)보다 낮아야 예측력이 있다는 뜻입니다.", "",
         r(ev), "",
         f"평가구간 AUC 차이(날씨 포함 − 미포함): {aucd[0]:.3f}, 월 단위 부트스트랩 90% 구간 [{aucd[1]:.3f}, {aucd[2]:.3f}] "
         "— 구간이 0을 포함하면 날씨 변수가 상승확률 예측을 개선했다고 말할 수 없습니다.", "",
         f"## 3층 최신 지수 ({d['ym'].max()})", "",
         r(latest, 2), "",
         "## 4층 포트폴리오 백테스트 (상위 1/3 − 하위 1/3, 다음 달 초과수익률, 거래비용 반영)",
         "업종 내: 업종마다 따로 상·하위를 만들고 평균(종목 선택 능력). 전 종목: 이전 방식(계절별 업종 갈아타기가 섞임). "
         "main=True 가 현재 설정. spread_t_NW는 Newey-West t값.", "",
         r(bt), "",
         "### 업종별 (평가구간) — 업종 내 평균은 효과 없는 업종이 섞이면 희석되므로 업종별로도 확인", "",
         (r(per_group) if per_group is not None and not per_group.empty else "(계산 불가)"), "",
         "## 날씨 수혜가 주가에 언제 반영되는가 (공시일 이벤트 스터디)",
         f"공시일 출처: {ev_info}. coef_pct = 날씨 수혜 1표준편차당 누적 초과수익률(%p), 분기 클러스터 표준오차.", "",
         (r(ev_tab) if ev_tab is not None and not ev_tab.empty else "(계산 불가)"), "",
         f"**판정:** {ev_verdict}", "",
         "## 민감도 결과변수", "",
         d.groupby(["code", "name", "group"])["y_src"].first().reset_index().to_markdown(index=False), "",
         "## 해석 시 주의",
         "- '남은 달'은 평년값으로 채웠습니다. 실제 기상청 예보(1·3개월 전망)를 날짜별로 쌓아야 진짜 '예보 기반'이 됩니다.",
         "- 종합 SWI의 가중치는 설계상 선택입니다. 백테스트에서 B·P·SWI를 따로 보는 이유입니다.",
         "- 상승확률 모델은 평가구간 결과만 믿을 수 있습니다(선택구간은 모델을 고르는 데 이미 썼음).",
         "- 실적 공시 시차를 90일로 보수적으로 두었습니다. 실제 공시일을 쓰면 정보가 조금 더 빨라집니다."]
    open(f"{C.OUTPUT}/swi_report.md", "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
