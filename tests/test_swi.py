"""Stock Weather Index 스모크테스트 (손계산 대조·미래정보 차단). 합성 데이터는 실제 자료가 아닙니다."""
import calendar
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swi import decision, engine, index, inputs  # noqa: E402


def _const_station(start=1980, end=2023, tavg=10.0):
    dates = pd.date_range(f"{start}-01-01", f"{end}-12-31")
    return pd.DataFrame({"stn": 108, "date": dates, "tavg": tavg, "tmin": tavg - 5, "tmax": tavg + 5, "rain": np.nan})


def test_monthly_weather():
    w = pd.DataFrame({"stn": 108, "date": pd.date_range("2021-01-01", "2021-01-31"),
                      "tavg": 8.0, "tmin": -13.0, "tmax": 12.0, "rain": [5.0] + [np.nan] * 30})
    m = inputs.monthly_weather(w).iloc[0]
    assert m["HDD"] == 310 and m["COLD"] == 31 and m["RAIN"] == 5 and m["CDD"] == 0


def test_expected_quarter_hand_calc_and_no_lookahead():
    w = _const_station()
    w.loc[(w["date"].dt.year == 2022) & (w["date"].dt.month == 1), "tavg"] = 0.0  # 2022년 1월만 0℃
    e = inputs.expected_quarter(inputs.monthly_weather(w)).set_index(["stn", "year", "month"])
    r = e.loc[("108", 2022, 1)]
    feb = np.mean([8 * (29 if calendar.isleap(y) else 28) for y in range(1992, 2022)])
    assert abs(r["HDD_exp"] - (18 * 31 + feb + 8 * 31)) < 1e-9     # 관측 1월 + 평년 2·3월
    assert abs(r["HDD_ly"] - 8 * 90) < 1e-9                       # 2021년 1분기 실제 (90일)
    assert abs(r["obs_share"] - 1 / 3) < 1e-12
    # 2023년 자료를 바꿔도 2022년 1월 시점 값은 그대로
    w2 = w.copy()
    w2.loc[w2["date"].dt.year == 2023, "tavg"] = 40.0
    e2 = inputs.expected_quarter(inputs.monthly_weather(w2)).set_index(["stn", "year", "month"])
    assert abs(e2.loc[("108", 2022, 1), "HDD_exp"] - r["HDD_exp"]) < 1e-9
    assert abs(e2.loc[("108", 2022, 1), "HDD_norm"] - r["HDD_norm"]) < 1e-9


def test_known_quarter():
    assert engine.known_quarter(pd.Timestamp("2024-03-31"), 90) == (2023, 4)
    assert engine.known_quarter(pd.Timestamp("2024-06-30"), 90) == (2024, 1)
    assert engine.known_quarter(pd.Timestamp("2024-06-28"), 90) == (2023, 4)  # 6/28-90일 = 3/30 → 1분기 미확정


def test_sensitivities_noiseless_and_pooling():
    rows = []
    rng = np.random.default_rng(0)
    for c, b in (("A", 0.1), ("B", 0.2), ("C", 0.3)):
        for y in range(2010, 2022):
            for q in (1, 2, 3, 4):
                x = rng.normal(0, 100)
                rows.append(dict(code=c, year=y, quarter=q, HDD_yoy=x, rev_g=0.01 * q + b * x / 100))
    d = pd.DataFrame(rows)
    s = engine.sensitivities(d, "HDD", min_obs=12)
    for c, b in (("A", 0.1), ("B", 0.2), ("C", 0.3)):
        assert np.allclose(s[c][0], b, atol=1e-8)   # 오차 0이면 종목 계수 그대로(w=1)
    # 표본이 적은 종목은 업종 계수로
    s2 = engine.sensitivities(d, "HDD", min_obs=100)
    assert np.allclose(s2["A"][0], s2["C"][0])


def test_benefit_confidence_hand_calc():
    ben, sd, conf = engine.benefit_and_confidence(0.04, 0.01, 2.0, 1.0)
    var = 0.04 ** 2 * 1 + 2 ** 2 * 0.01 ** 2 + 0.01 ** 2 * 1   # 0.0021
    assert abs(ben - 0.08) < 1e-12 and abs(sd - np.sqrt(var)) < 1e-12
    assert abs(conf - 100 * (2 * stats.norm.cdf(0.08 / np.sqrt(var)) - 1)) < 1e-9


def test_combine():
    assert abs(index.combine([80], [60], [100])[0] - 70) < 1e-12
    assert abs(index.combine([80], [60], [0])[0] - 60) < 1e-12
    assert abs(index.combine([80], [np.nan], [100])[0] - 65) < 1e-12  # 확률 없으면 50 취급


def test_walk_forward_no_peek():
    rng = np.random.default_rng(1)
    months = pd.period_range("2015-01", "2020-12", freq="M")
    df = pd.DataFrame([dict(ym=m, x=rng.normal()) for m in months for _ in range(10)])
    df["up"] = (df["x"] + rng.normal(0, 1, len(df)) > 0).astype(float)
    p1 = engine.walk_forward(df, ["x"], "logit", 24, 1)
    cut = pd.Period("2019-06", "M")
    df2 = df.copy()
    df2.loc[df2["ym"] >= cut, "up"] = 1 - df2.loc[df2["ym"] >= cut, "up"]   # 미래 정답 뒤집기
    p2 = engine.walk_forward(df2, ["x"], "logit", 24, 1)
    m = df["ym"] <= cut
    assert np.allclose(p1[m].fillna(-1), p2[m].fillna(-1))  # cut 달까지의 예측은 변하지 않아야 함
    assert p1[df["ym"] < months[24]].isna().all()            # 최소 학습기간 전에는 예측 없음


def test_long_short_hand_calc():
    df = pd.DataFrame({"ym": pd.Period("2020-01", "M"), "code": list("ABCDEF"),
                       "s": [1, 2, 3, 4, 5, 6], "fwd_xret": [-0.02, -0.01, 0, 0, 0.01, 0.03]})
    bt = decision.long_short(df, "s", cost_bps=30)
    assert abs(bt.loc[0, "spread"] - ((0.02 - (-0.015)) - 2 * 0.003)) < 1e-12


def test_newey_west_lag0():
    x = np.array([0.01, 0.02, -0.01, 0.03, 0.0, 0.02, 0.01, -0.02, 0.04, 0.01, 0.0])
    t = decision.newey_west_t(x, lags=0)
    assert abs(t - x.mean() / np.sqrt(x.var() / len(x))) < 1e-12


# ---------- 개선 1~3 ----------
from swi import event, volume  # noqa: E402


def test_volume_growth_hand_calc():
    v = pd.DataFrame({"region": "서울", "ym": [f"{y}-0{m}" for y in (2021, 2022) for m in (1, 2, 3)] + ["2022-04"],
                      "volume": [100, 100, 100, 110, 110, 110, 50]})
    g = volume.quarterly_volume_growth(v).set_index(["region", "year", "quarter"])
    assert abs(g.loc[("서울", 2022, 1), "vol_g"] - np.log(1.1)) < 1e-12
    assert ("서울", 2022, 2) not in g.index          # 4월만 있는 분기는 계산하지 않음


def test_volume_attach_switch():
    fin = pd.DataFrame({"code": ["017390"] * 10 + ["071320"] * 10, "year": list(range(2010, 2020)) * 2,
                        "quarter": 1, "rev_g": 0.5})
    vg = pd.DataFrame({"region": "서울", "year": range(2010, 2020), "quarter": 1, "vol_g": 0.1})
    f = volume.attach(fin, vg, {"017390": "서울"}, True)
    assert (f.loc[f["code"] == "017390", "y_sens"] == 0.1).all() and (f.loc[f["code"] == "017390", "y_src"] == "판매량").all()
    assert (f.loc[f["code"] == "071320", "y_sens"] == 0.5).all() and (f.loc[f["code"] == "071320", "y_src"] == "매출").all()
    off = volume.attach(fin, vg, {"017390": "서울"}, False)       # 끄면 이전 동작(매출)
    assert (off["y_sens"] == 0.5).all()


def test_event_dates_fallback_and_override():
    keys = pd.DataFrame({"code": ["A", "A"], "year": [2022, 2022], "q": [1, 2]})
    disc = pd.DataFrame({"code": ["A"], "year": [2022], "q": [2], "date": [pd.Timestamp("2022-07-28")], "kind": ["잠정실적"]})
    ev = event.event_dates(keys, disc, 45).set_index("q")
    assert ev.loc[1, "date"] == pd.Timestamp("2022-05-15") and ev.loc[1, "kind"].startswith("가정")
    assert ev.loc[2, "date"] == pd.Timestamp("2022-07-28") and ev.loc[2, "kind"] == "잠정실적"
    assert ev.loc[1, "qs"] == pd.Timestamp("2022-01-01") and ev.loc[1, "qe"] == pd.Timestamp("2022-03-31")


def test_cars_hand_calc():
    dates = pd.bdate_range("2022-01-03", "2022-06-30")
    ar = pd.DataFrame({"code": "A", "date": dates, "ar": 0.01})
    ev = pd.DataFrame({"code": ["A"], "year": [2022], "q": [1], "kind": ["x"], "date": [pd.Timestamp("2022-05-02")],
                       "qs": [pd.Timestamp("2022-01-01")], "qe": [pd.Timestamp("2022-03-31")]})
    c = event.cars(ev, ar, {"pre": None, "ann": (-1, 1), "post": (2, 40)}).iloc[0]
    n_q1 = ((dates >= "2022-01-01") & (dates <= "2022-03-31")).sum()
    assert abs(c["pre"] - 0.01 * n_q1) < 1e-12 and abs(c["ann"] - 0.03) < 1e-12 and abs(c["post"] - 0.39) < 1e-12


def test_event_regress_recovers_slope():
    """합성(실제 자료 아님): CAR = 0.02 × 수혜z + 잡음 -> 계수 2%p 근처."""
    rng = np.random.default_rng(5)
    n = 400
    d = pd.DataFrame({"group": rng.choice(["gas", "bev"], n), "surprise_z": rng.normal(0, 1, n),
                      "cluster": rng.integers(0, 40, n)})
    d["ann"] = 0.02 * d["surprise_z"] + rng.normal(0, 0.02, n)
    r = event.regress(d, "ann")
    assert abs(r["coef_pct"] - 2.0) < 0.3 and r["t"] > 5


def test_verdict_rules():
    mk = lambda tp, ta, tq: pd.DataFrame({"window": ["pre", "ann", "post"], "t": [tp, ta, tq]})
    assert event.verdict(mk(3, 0, 0), "pre", ["ann", "post"]).startswith("분기 중에 이미")
    assert event.verdict(mk(0, 2.5, 0), "pre", ["ann", "post"]).startswith("공시 전후")
    assert event.verdict(mk(1, 1, 1), "pre", ["ann", "post"]).startswith("어느 구간")


def test_long_short_within_hand_calc():
    df = pd.DataFrame({"ym": pd.Period("2020-01", "M"), "group": list("AAABBB"), "code": list("123456"),
                       "s": [1, 2, 3, 1, 2, 3], "fwd_xret": [0.0, 0.01, 0.02, -0.01, 0.0, 0.05]})
    bt = decision.long_short_within(df, "s", cost_bps=0)
    # 업종 A: 0.02 - 0.00 = 0.02, 업종 B: 0.05 - (-0.01) = 0.06 -> 평균 0.04
    assert abs(bt.loc[0, "spread"] - 0.04) < 1e-12


def test_benefit_score_by_group_independent():
    months = pd.period_range("2018-01", "2020-12", freq="M")
    rng = np.random.default_rng(2)
    df = pd.DataFrame([dict(ym=m, group=g, benefit=rng.normal(0, 0.01 if g == "A" else 0.05))
                       for m in months for g in ("A", "B") for _ in range(2)])
    s1 = index.benefit_score(df, by="group")
    df2 = df.copy()
    df2.loc[df2["group"] == "B", "benefit"] *= 10                  # B 업종만 크기를 바꿔도
    s2 = index.benefit_score(df2, by="group")
    a = df["group"] == "A"
    assert np.allclose(s1[a], s2[a])                                # A 업종 점수는 그대로
