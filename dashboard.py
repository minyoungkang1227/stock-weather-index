"""Stock Weather Index 대시보드.  실행: python swi_system.py dashboard  (또는 streamlit run dashboard.py)"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

import config as C
from system import db

st.set_page_config(page_title="Stock Weather Index", layout="wide")
GROUP_KO = {"gas": "도시가스·난방", "beverage": "음료·빙과", "apparel": "의류", "retail": "유통"}


@st.cache_data(ttl=300)
def t(name: str) -> pd.DataFrame | None:
    if not os.path.exists(C.DB_PATH):
        return None
    df = db.read(C.DB_PATH, name)
    if df is not None and "code" in df:
        df["code"] = df["code"].astype(str).str.zfill(6)
    if df is not None and "group" in df:
        df["group"] = df["group"].map(lambda g: GROUP_KO.get(g, g))
    return df


st.title("Stock Weather Index")
st.caption("이번 분기 날씨 짐작(관측·기상청 예보·평년)과 그것이 업종 물량에 주는 영향을 봅니다. "
           "확인된 것과 확인되지 않은 것은 '검증' 탭 맨 위 표에 있습니다. 투자 권유가 아닙니다.")

if not os.path.exists(C.DB_PATH):
    st.error(f"DB가 없습니다({C.DB_PATH}). `python swi_system.py init` → `load`(이미 수집했을 때) 또는 `backfill` 순서로 실행하세요.")
    st.stop()

daily, runs = t("swi_daily"), t("runs")
tab0, tab1, tab2, tab3, tab4 = st.tabs(["이번 분기 날씨", "종목 지수 (참고)", "종목 상세", "검증", "시스템 상태"])

REGION_OF = {108: "서울", 112: "인천", 119: "경기(수원)", 133: "대전", 143: "대구", 152: "울산", 156: "광주", 159: "부산"}


def out_csv(name: str) -> pd.DataFrame | None:
    f = os.path.join(C.OUTPUT, f"{name}.csv")
    return pd.read_csv(f) if os.path.exists(f) else None


def volume_coefs() -> dict:
    """검정에서 나온 '100도일당 물량 변화' 계수 (통과한 것만). 없으면 빈 dict."""
    out = {}
    g = out_csv("prereg_gas_volume")
    if g is not None and str(g.iloc[0]["pass"]).lower() == "true":
        out["도시가스 판매량 (지역)"] = ("HDD", float(g.iloc[0]["coef"]))
    sv = out_csv("prereg_sector_volume")
    if sv is not None:
        for _, r in sv.iterrows():
            if str(r["pass"]).lower() != "true":
                continue
            v = "CDD" if "냉방" in r["name"] else "HDD"
            label = {"전력": "전력 판매량 (전국)", "비알코올": "비알코올음료 출하 (전국)", "빙과": "낙농·빙과 출하 (전국)"}
            key = next((lab for k, lab in label.items() if k in r["name"]), r["name"])
            out[f"{key} · {'더위' if v == 'CDD' else '추위'}"] = (v, float(r["coef"]))
    return out


with tab0:
    nc = t("nowcast_daily")
    if nc is None or nc.empty:
        st.info("아직 날씨 짐작이 없습니다. `python swi_system.py daily` 를 실행하세요.")
    else:
        asof = nc["asof"].max()
        cur = nc[nc["asof"] == asof].copy()
        qn = int(cur["quarter"].iloc[0])
        main = "HDD" if qn in (1, 4) else ("CDD" if qn == 3 else "HDD")
        word = {"HDD": "난방도일 (추위)", "CDD": "냉방도일 (더위)"}
        a, b, c = st.columns(3)
        a.metric("기준일", asof)
        a.caption(f"{int(cur['year'].iloc[0])}년 {qn}분기")
        r0 = cur.iloc[0]
        b.metric("관측 비율", f"{100 * r0['obs_share']:.0f}%")
        b.caption(f"관측 {int(r0['n_obs'])}일 · 예보 {int(r0['n_fc'])}일 · 평년 {int(r0['n_clim'])}일")
        nat = cur[cur["stn"].astype(str) == "NATIONAL"]
        if len(nat):
            n0 = nat.iloc[0]
            c.metric(f"전국 {word[main]}: 작년 대비", f"{n0[main + '_exp'] - n0[main + '_ly']:+.0f}도일",
                     help="이번 분기 짐작(관측+예보+평년) − 작년 같은 분기 실제")
        st.subheader(f"지역별 이번 분기 {word[main]} 짐작")
        rows = []
        for _, r in cur.iterrows():
            stn = str(r["stn"])
            name = "전국 (6개 도시 평균)" if stn == "NATIONAL" else REGION_OF.get(int(stn), stn)
            rows.append({"지역": name, "짐작": r[main + "_exp"], "± (1표준편차)": r[main + "_sd"],
                         "작년 같은 분기": r[main + "_ly"], "작년 대비": r[main + "_exp"] - r[main + "_ly"],
                         "평년": r[main + "_norm"], "평년 대비": r[main + "_exp"] - r[main + "_norm"]})
        st.dataframe(pd.DataFrame(rows).round(0), hide_index=True, width="stretch")
        st.caption("짐작 = 어제까지 관측 + 기상청 중기예보(편향 보정) + 평년(직전 30년). "
                   "예보를 쓰면 평년만 쓸 때보다 분기 난방도일 짐작 오차가 9.2% 줄어듭니다(2015~2026 검증, p = 0.0006).")

        coefs = volume_coefs()
        if coefs:
            st.subheader("이 날씨가 업종 물량에 주는 몫 (참고)")
            st.warning("참고치입니다. 날씨가 물량을 움직인다는 것은 확인됐지만, 2021~2025 표본외 검증에서 "
                       "이 몫으로 물량을 미리 맞히는 정확도는 개선되지 않았습니다(가격·추세 충격이 더 큼). "
                       "물량 예측치로 쓰지 마세요.")
            vr = []
            for label, (v, b_) in coefs.items():
                if v != main and qn != 2:
                    continue
                if "지역" in label:
                    for _, r in cur[cur["stn"].astype(str) != "NATIONAL"].iterrows():
                        if int(r["stn"]) == 152:      # 울산은 판매량 검정에서 규칙으로 제외
                            continue
                        d = (r[v + "_exp"] - r[v + "_ly"]) / 100
                        vr.append({"물량": f"도시가스 판매량 · {REGION_OF.get(int(r['stn']))}", "100도일당": b_,
                                   "작년 대비 날씨 몫(%p)": 100 * b_ * d, "± 날씨 불확실성(%p)": 100 * b_ * r[v + "_sd"] / 100})
                elif len(nat):
                    d = (n0[v + "_exp"] - n0[v + "_ly"]) / 100
                    vr.append({"물량": label, "100도일당": b_, "작년 대비 날씨 몫(%p)": 100 * b_ * d,
                               "± 날씨 불확실성(%p)": 100 * b_ * n0[v + "_sd"] / 100})
            if vr:
                st.dataframe(pd.DataFrame(vr).round(3), hide_index=True, width="stretch")
                st.caption("날씨 몫 = 검정에서 나온 계수 × (이번 분기 짐작 − 작년 같은 분기). 계수 자체의 추정오차는 포함하지 않았습니다.")
            else:
                st.caption("이번 분기에 해당하는 계절 지표의 검정 통과 물량이 없습니다.")

with tab1:
    if daily is None or daily.empty:
        st.info("아직 일별 지수가 없습니다. `python swi_system.py daily` 를 실행하세요.")
        last_run = runs.sort_values("finished").iloc[-1] if runs is not None and len(runs) else None
        if last_run is not None and last_run["status"] != "ok":
            st.warning(f"최근 실행 상태: {last_run['job']} / {last_run['status']} — {last_run['message'][:300]}")
    else:
        asof = daily["asof"].max()
        cur = daily[daily["asof"] == asof].copy()
        last_daily = runs[runs["job"] == "daily"].sort_values("finished").iloc[-1] if runs is not None else None
        if last_daily is not None and last_daily["status"] != "ok":
            st.warning(f"가장 최근 일별 실행이 '{last_daily['status']}' 상태입니다. 아래는 {asof} 기준 마지막 정상 결과입니다. "
                       f"사유: {last_daily['message'][:300]}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("기준일", asof)
        c2.metric("관심종목", int(cur["관심종목"].astype(int).sum()))
        c3.metric("위험경고", int(cur["위험경고"].astype(int).sum()))
        c4.metric("상승확률 모델", str(cur["model"].iloc[0]) if "model" in cur else "없음")
        groups = st.multiselect("업종", sorted(cur["group"].unique()), default=sorted(cur["group"].unique()))
        view = cur[cur["group"].isin(groups)].sort_values("SWI", ascending=False)
        st.dataframe(view[["name", "group", "SWI", "B", "P", "C", "관심종목", "위험경고", "obs_share"]]
                     .rename(columns={"name": "종목", "group": "업종", "B": "기상수혜도", "P": "상승확률", "C": "신뢰도",
                                      "obs_share": "분기 관측비율"}).round(1),
                     hide_index=True, width="stretch")
        st.caption("기상수혜도 50 = 날씨 영향 없음. 신뢰도가 낮으면 종합 SWI는 50 쪽으로 당겨집니다.")
        st.warning("근거 범위: 실데이터 검증에서 확인된 것은 '날씨 → 물량(도시가스·전력·비알코올음료·빙과)'과 '예보가 분기 날씨 짐작을 개선'까지이고, 날씨만으로 물량을 미리 맞히는 것은 표본외 검증에서 실패했습니다. "
                   "상승확률(P)과 이를 포함한 SWI 는 주가 근거가 확인되지 않았습니다 — 검증 탭 맨 위 '사전 등록 판정' 참고.")

with tab2:
    if daily is None or daily.empty:
        st.info("일별 지수가 쌓이면 표시됩니다.")
    else:
        names = daily.drop_duplicates("code").set_index("code")["name"].to_dict()
        code = st.selectbox("종목", list(names), format_func=lambda c: f"{names[c]} ({c})")
        h = daily[daily["code"] == code].sort_values("asof")
        last = h.iloc[-1]
        a, b, c = st.columns(3)
        a.metric("SWI", f"{last['SWI']:.1f}")
        b.metric("기상수혜도 / 신뢰도", f"{last['B']:.1f} / {last['C']:.1f}")
        c.metric("상승확률", "-" if pd.isna(last.get("P")) else f"{last['P']:.1f}")
        st.subheader("점수 추이 (일별)")
        st.line_chart(h.set_index("asof")[["SWI", "B", "C"] + (["P"] if h["P"].notna().any() else [])])
        st.subheader(f"이번 분기 날씨 구성 — 지표: {last['index_var']}")
        parts = pd.DataFrame({"값": [last["w_obs"], last["w_fc"], last["w_clim"]]},
                             index=[f"관측 {int(last['n_obs'])}일", f"예보 {int(last['n_fc'])}일", f"평년 {int(last['n_clim'])}일"])
        st.bar_chart(parts)
        st.write(f"이번 분기 예상 = {last['w_obs'] + last['w_fc'] + last['w_clim']:.1f}, "
                 f"작년 같은 분기 = {last['w_ly']:.1f}, 평년 = {last['w_norm']:.1f}")
        st.write(f"민감도(β) = {last['beta']:.4f} (표준오차 {last['beta_se']:.4f}), 예상 날씨 변화 = {last['dW']:.2f} "
                 f"→ 날씨 수혜(민감도 결과변수 성장률 기여) = {100 * last['benefit']:.2f}%p")
        m = t("swi_monthly")
        if m is not None and "y_src" in m:
            src = m[m["code"] == code]["y_src"].dropna()
            if len(src):
                st.caption(f"민감도 추정에 쓴 결과변수: {src.iloc[-1]}")

with tab3:
    st.subheader(f"사전 등록 판정 — 결과를 보기 전에 정한 기준 (등록 {C.PREREG_DATE})")
    st.caption("p값은 모두 연도 순열검정 999회. 이 절이 결론이고, 아래 표들은 참고 자료입니다. 자세한 근거: docs/결론_말할수있는것_없는것.md")
    rows = []
    s1 = out_csv("prereg_stage1")
    if s1 is not None:
        for _, r in s1.iterrows():
            rows.append(dict(검정=f"날씨 → 매출 ({GROUP_KO.get(r['group'], r['group'])}, {r['x']})", 계수=r["coef"],
                             p값=r["p_one"], 판정="통과" if str(r["pass"]).lower() == "true" else "미통과"))
    for name, label in (("prereg_gas_volume", "날씨 → 도시가스 판매량 (7개 시도)"),
                        ("prereg_gas_opinc", "날씨 → 도시가스 5사 영업이익")):
        x = out_csv(name)
        if x is not None:
            r = x.iloc[0]
            rows.append(dict(검정=label, 계수=r["coef"], p값=r["p_perm"], 판정="통과" if str(r["pass"]).lower() == "true" else "미통과"))
    sv = out_csv("prereg_sector_volume")
    if sv is not None:
        for _, r in sv.iterrows():
            rows.append(dict(검정=f"업종 물량: {r['name']}", 계수=r["coef"], p값=r["p_one"],
                             판정=("자료 없음" if str(r.get("note", "")) == "자료 없음" else
                                   "통과" if str(r["pass"]).lower() == "true" else "미통과")))
    ov = out_csv("prereg_nowcast_oos_verdict")
    if ov is not None:
        r = ov.iloc[0]
        rows.append(dict(검정=f"표본외: 날씨로 물량 미리 짐작 ({int(r['quarters_better'])}/{int(r['n_quarters'])}개 분기 개선)",
                         계수=r["mean_gain"], p값=r["p_one"], 판정="통과" if str(r["passed"]).lower() == "true" else "미통과"))
    fvx = out_csv("prereg_forecast_value")
    if fvx is not None:
        r = fvx.iloc[0]
        rows.append(dict(검정=f"예보 → 분기 난방도일 짐작 (RMSE {r['rmse_A']:.1f}→{r['rmse_B']:.1f}도일)",
                         계수=-r["rmse_reduction"], p값=r["p_one"], 판정="통과" if str(r["passed"]).lower() == "true" else "미통과"))
    s2 = out_csv("prereg_stage2")
    if s2 is not None:
        rows.append(dict(검정="2차: SWI 업종 내 롱숏 (주가)", 계수=float("nan"), p값=float("nan"), 판정=str(s2.iloc[0]["verdict"])))
    if rows:
        st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch", height=36 * (len(rows) + 1) + 4)
    else:
        st.info("`python run_analysis.py` 를 실행하면 판정이 표시됩니다.")
    prof = out_csv("forecast_value_by_days_left")
    if prof is not None:
        st.caption("예보의 값어치 — 분기 남은 일수별 짐작 오차 감소율")
        st.dataframe(prof.rename(columns={"bucket": "남은 일수", "rmse_A": "평년만", "rmse_B": "예보 포함", "reduction": "감소율"})
                     .round(3), hide_index=True)

    es, ev, bt, bg = t("swi_event_study"), t("swi_model_eval"), t("swi_backtest"), t("swi_backtest_by_group")
    if es is None:
        st.info("월간 작업(`python swi_system.py monthly`)을 실행하면 검증 결과가 표시됩니다.")
    else:
        rid = es["run_id"].iloc[-1]
        st.divider()
        st.caption("아래는 참고 자료입니다. 1차(날씨→실적)를 통과한 업종이 없으므로 주가 관련 수치는 근거로 쓰지 않습니다.")
        st.subheader("날씨가 주가에 언제 반영되는가 (공시일 이벤트 스터디)")
        st.success(es[es["run_id"] == rid]["verdict"].iloc[0])
        st.dataframe(es[es["run_id"] == rid][["window", "n", "coef_pct", "t", "p"]].round(3), hide_index=True)
        st.subheader("상승확률 모델 (표본외)")
        st.dataframe(ev[ev["run_id"] == ev["run_id"].iloc[-1]].drop(columns="run_id").round(3), hide_index=True)
        st.subheader("포트폴리오 검증 (상위 1/3 − 하위 1/3, 다음 달 초과수익률, 거래비용 반영)")
        st.dataframe(bt[bt["run_id"] == bt["run_id"].iloc[-1]].drop(columns="run_id").round(3), hide_index=True)
        if bg is not None:
            st.caption("업종별")
            st.dataframe(bg[bg["run_id"] == bg["run_id"].iloc[-1]].drop(columns="run_id").round(3), hide_index=True)

with tab4:
    st.subheader("실행 이력")
    if runs is not None:
        st.dataframe(runs.sort_values("finished", ascending=False).head(20)[["finished", "job", "status", "asof", "message"]],
                     hide_index=True, width="stretch")
    st.subheader("데이터 신선도")
    w = t("weather_daily")
    if w is not None:
        st.dataframe(w.dropna(subset=["tavg"]).groupby("stn")["date"].max().rename("마지막 관측일").reset_index(), hide_index=True)
    p = t("prices")
    if p is not None:
        st.write(f"주가 마지막 날짜: {p['date'].max()}")
    f = t("forecast_mid")
    if f is not None and len(f):
        iss = f["issued_at"].astype(str)
        st.write(f"예보 이력: {iss.nunique():,}회 발표분 저장 ({iss.min()[:8]} ~ {iss.max()[:8]})")
    else:
        st.write("예보 스냅샷: 아직 없음")
