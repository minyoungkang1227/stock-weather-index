"""운영 시스템 스모크테스트 (손계산 대조). 합성 데이터는 실제 자료가 아닙니다."""
import os
import sys
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from collect import forecast  # noqa: E402
from system import db, nowcast, today  # noqa: E402


def test_db_upsert_replace_and_new_column():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.db")
        db.upsert(p, "market", pd.DataFrame({"date": ["2024-01-02", "2024-01-03"], "close": [1.0, 2.0]}))
        db.upsert(p, "market", pd.DataFrame({"date": ["2024-01-03"], "close": [5.0], "volume": [10]}))
        r = db.read(p, "market").sort_values("date")
        assert r["close"].tolist() == [1.0, 5.0]          # 같은 키는 교체, 중복 없음
        assert "volume" in r and pd.isna(r["volume"].iloc[0])
        db.upsert(p, "market", pd.DataFrame({"date": ["2024-01-03"], "close": [5.0], "volume": [10]}))
        assert len(db.read(p, "market")) == 2             # 같은 내용을 다시 넣어도 그대로


def test_latest_tmfc():
    assert forecast.latest_tmfc(datetime(2026, 11, 9, 7, 30)) == "202611090600"
    assert forecast.latest_tmfc(datetime(2026, 11, 9, 19, 0)) == "202611091800"
    assert forecast.latest_tmfc(datetime(2026, 11, 9, 5, 0)) == "202611081800"


def test_parse_item():
    item = {"regId": "11B10101", "taMin4": "1", "taMax4": "9", "taMin5": "-2", "taMax5": "6", "taMin6": "", "taMax6": ""}
    f = forecast.parse_item(item, 108, "202611090600").set_index("lead_day")
    assert list(f.index) == [4, 5]                       # 빈 값은 건너뜀
    assert f.loc[4, "target_date"] == "2026-11-13" and f.loc[5, "tmin"] == -2.0


def _cfg(**over):
    """config 를 복사해 일부 값만 바꾼 설정 (PC에 오차 모델 파일이 생겨도 테스트가 흔들리지 않게)."""
    from types import SimpleNamespace
    return SimpleNamespace(**{**{k: getattr(C, k) for k in dir(C) if k.isupper()}, **over})


def test_nowcast_hand_calc():
    dates = pd.date_range("1980-01-01", "2023-02-13")
    w = pd.DataFrame({"stn": 108, "date": dates, "tavg": 10.0, "tmin": 5.0, "tmax": 15.0, "rain": np.nan})
    fc = pd.DataFrame({"issued_at": "202302140600", "stn": 108, "lead_day": range(3, 11),
                       "target_date": pd.date_range("2023-02-17", "2023-02-24").strftime("%Y-%m-%d"),
                       "tmin": 0.0, "tmax": 10.0})
    nc = nowcast.nowcast(w, fc, pd.Timestamp("2023-02-14"), _cfg(FORECAST_ERR_MODEL_PATH=None)).set_index("stn").loc["108"]
    assert nc["n_obs"] == 44 and nc["n_fc"] == 8 and nc["n_clim"] == 38
    assert abs(nc["HDD_obs"] - 8 * 44) < 1e-9          # 관측: 평균 10℃ -> 하루 8
    assert abs(nc["HDD_fc"] - 13 * 8) < 1e-9           # 예보: (0+10)/2=5℃ -> 하루 13
    assert abs(nc["HDD_clim"] - 8 * 38) < 1e-9         # 평년: 하루 8
    assert abs(nc["HDD_exp"] - 760) < 1e-9 and abs(nc["HDD_ly"] - 720) < 1e-9
    assert abs(nc["HDD_sd"] - np.sqrt(8 * C.FORECAST_ERR_SD ** 2)) < 1e-9   # 평년 구간 변동 0 + 예보 오차
    assert abs(nc["obs_share"] - 44 / 90) < 1e-12


def test_gate_stops_on_stale_and_outlier():
    asof = pd.Timestamp("2024-06-30")
    px = pd.DataFrame({"date": ["2024-06-28"], "code": ["000001"], "close": [1.0]})
    w = pd.DataFrame({"stn": 108, "date": pd.date_range("2024-01-01", "2024-06-10"), "tavg": 10.0, "tmin": 5.0, "tmax": 15.0})
    with pytest.raises(today.GateError):
        today.gate(w, px, asof)                          # 관측이 20일 멈춤
    w2 = pd.DataFrame({"stn": 108, "date": pd.date_range("2024-01-01", "2024-06-29"), "tavg": 10.0, "tmin": 5.0, "tmax": 15.0})
    w2.loc[10, "tmin"] = 20.0                            # 최저 > 최고
    with pytest.raises(today.GateError):
        today.gate(w2, px, asof)
    w2.loc[10, "tmin"] = 5.0
    today.gate(w2, px, asof)                             # 정상이면 통과


# ---------- 기상청 API허브 (실제 응답 샘플로 손계산 대조) ----------
from collect import kma_apihub  # noqa: E402

FX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def test_apihub_asos_parse():
    d = kma_apihub.asos_standard(kma_apihub.parse_typ01(open(os.path.join(FX, "kma_asos_sample.txt"), encoding="utf-8").read()))
    seoul = d[d["stn"] == 108].iloc[0]
    assert seoul["date"] == pd.Timestamp("2024-01-01")
    assert (seoul["tavg"], seoul["tmax"], seoul["tmin"], seoul["rain"]) == (3.3, 7.3, -0.3, 0.0)


def test_apihub_missing_values_become_nan():
    txt = "#START7777\n# 1. TM : a\n# 2. STN : b\n# 3. TA_AVG : c\n# 4. TA_MAX : d\n# 5. TA_MIN : e\n# 6. RN_DAY : f\n" \
          "20240101 108 3.3 -99.0 -0.3 -9.0\n#7777END"
    d = kma_apihub.asos_standard(kma_apihub.parse_typ01(txt)).iloc[0]
    assert np.isnan(d["tmax"]) and np.isnan(d["rain"]) and d["tavg"] == 3.3


def test_apihub_forecast_parse():
    raw = kma_apihub.parse_typ01(open(os.path.join(FX, "kma_wc_sample.txt"), encoding="utf-8").read())
    f = kma_apihub.forecast_standard(raw, {v: int(k) for k, v in C.FORECAST_REG.items()}).set_index(["stn", "target_date"])
    r = f.loc[(108, "2025-11-05")]          # 서울 11B10101, 2025-11-01 06시 발표, 4일 뒤
    assert r["issued_at"] == "202511010600" and r["lead_day"] == 4 and (r["tmin"], r["tmax"]) == (6, 17)
    assert 109 not in f.index.get_level_values(0)   # 우리 지점이 아닌 구역(백령도)은 빠짐


def test_forecast_region_codes_match_kma_list():
    """config 의 예보구역 코드가 기상청 예보구역 목록의 도시 이름과 맞는지 (광주는 경기 광주와 구별)."""
    reg = {}
    for ln in open(os.path.join(FX, "kma_reg_sample.txt"), encoding="utf-8"):
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.split()
        reg[p[0]] = p[-1]
    expect = {108: "서울", 112: "인천", 119: "수원", 133: "대전", 143: "대구", 152: "울산", 156: "광주", 159: "부산"}
    for stn, code in C.FORECAST_REG.items():
        assert reg[code] == expect[stn]
    assert C.FORECAST_REG[156] == "11F20501"   # 전남 광주 (11B20702는 경기 광주)


def test_forecast_skill_hand_calc():
    fc = pd.DataFrame({"issued_at": "202401010600", "stn": 108, "lead_day": [4, 4], "target_date": ["2024-01-05", "2024-01-06"],
                       "tmin": [0.0, 2.0], "tmax": [10.0, 8.0]})                     # 예보 평균 5, 5
    obs = pd.DataFrame({"stn": 108, "date": pd.to_datetime(["2024-01-05", "2024-01-06"]), "tavg": [4.0, 7.0]})
    sk = kma_apihub.forecast_skill(fc, obs).iloc[0]
    assert sk["n"] == 2 and abs(sk["bias"] - (-0.5)) < 1e-12 and abs(sk["mae"] - 1.5) < 1e-12   # 오차 +1, -2


def test_apihub_extra_trailing_column_and_misalignment_guard():
    """값이 설명보다 1개 많으면 뒤에 EXTRA_1 로 붙이고, 열이 밀린 경우는 멈춘다."""
    head = "#START7777\n# 1. TM : a\n# 2. STN : b\n# 3. TA_AVG : c\n# 4. TA_MAX : d\n# 5. TA_MAX_TM : t\n# 6. TA_MIN : e\n# 7. RN_DAY : f\n"
    d = kma_apihub.asos_standard(kma_apihub.parse_typ01(head + "20240101 108 3.3 7.3 1430 -0.3 0.0 =\n20240101 108 3.3 7.3 1430 -0.3 0.0 9\n#7777END"))
    assert list(d["tavg"]) == [3.3, 3.3]
    shifted = head + "20240101 3.3 108 7.3 1430 -0.3 0.0 9\n#7777END"   # STN 자리에 3.3 → 정렬 깨짐
    with pytest.raises(ValueError):
        kma_apihub.asos_standard(kma_apihub.parse_typ01(shifted))


def test_apihub_asos_comma_format_trailing_comma():
    """실제 stn=0 응답은 쉼표 구분 + 줄 끝 쉼표 → 끝 빈칸을 버리면 정확히 56열."""
    raw = kma_apihub.parse_typ01(open(os.path.join(FX, "kma_asos_comma_sample.txt"), encoding="utf-8").read())
    assert raw.shape[1] == 56 and not any(c.startswith("EXTRA_") for c in raw.columns)
    d = kma_apihub.asos_standard(raw).set_index("stn")
    assert (d.loc[108, "tavg"], d.loc[108, "tmax"], d.loc[108, "tmin"], d.loc[108, "rain"]) == (3.3, 7.3, -0.3, 0.0)
    assert (d.loc[112, "tavg"], d.loc[112, "tmax"], d.loc[112, "tmin"]) == (2.9, 5.2, 0.2)



# ---------- 예보 오차 모델 (리드 간 상관) ----------
def test_forecast_error_model_hand_calc():
    """발표 A: 4일 뒤 +1, 5일 뒤 +2 / 발표 B: -1, -2 (관측 0℃).
    편향 0, 분산 l4 = 2, l5 = 8 (n-1로 나눔), 공분산 (1·2 + (-1)(-2))/1 = 4 → 상관 1."""
    fc = pd.DataFrame({"issued_at": ["A", "A", "B", "B"], "stn": 108, "lead_day": [4, 5, 4, 5],
                       "target_date": ["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"],
                       "tmin": [0.0, 1.0, -2.0, -3.0], "tmax": [2.0, 3.0, 0.0, -1.0]})   # 평균 1, 2, -1, -2
    obs = pd.DataFrame({"stn": 108, "date": pd.date_range("2024-01-05", "2024-01-08"), "tavg": 0.0})
    m = kma_apihub.forecast_error_model(fc, obs, min_pairs=2)
    assert m.loc[4, "bias"] == 0 and m.loc[5, "bias"] == 0
    assert abs(m.loc[4, "lead_4"] - 2) < 1e-12 and abs(m.loc[5, "lead_5"] - 8) < 1e-12 and abs(m.loc[4, "lead_5"] - 4) < 1e-12


def test_forecast_sum_var_hand_calc():
    m = pd.DataFrame({"bias": [0.0, 0.0], "lead_4": [2.0, 4.0], "lead_5": [4.0, 8.0]}, index=pd.Index([4, 5], name="lead_day"))
    assert nowcast.forecast_sum_var([4, 5], m, 2.0) == 2 + 8 + 4 + 4      # 1'Σ1 = 18
    assert nowcast.forecast_sum_var([4, 5], None, 2.0) == 2 * 4            # 독립 가정 = 8 (과소)
    assert nowcast.forecast_sum_var([3, 4], m, 2.0) == 4 + 2               # 모델에 없는 리드 3: 대각 sd², 상관 0


def test_nowcast_uses_error_model():
    """리드 3~10 전부 분산 1·공분산 0.5, 편향 +1℃ → 예보 평균 5℃에서 1℃ 빼 4℃: HDD 하루 14 × 8일 = 112,
    분산 = 8×1 + 56×0.5 = 36 → sd 6 (독립 가정이면 sqrt(8))."""
    leads = list(range(3, 11))
    m = pd.DataFrame({"n": 100, "bias": 1.0, **{f"lead_{k}": [1.0 if j == k else 0.5 for j in leads] for k in leads}},
                     index=pd.Index(leads, name="lead_day"))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "em.csv")
        m.to_csv(path)
        dates = pd.date_range("1980-01-01", "2023-02-13")
        w = pd.DataFrame({"stn": 108, "date": dates, "tavg": 10.0, "tmin": 5.0, "tmax": 15.0, "rain": np.nan})
        fc = pd.DataFrame({"issued_at": "202302140600", "stn": 108, "lead_day": leads,
                           "target_date": pd.date_range("2023-02-17", "2023-02-24").strftime("%Y-%m-%d"),
                           "tmin": 0.0, "tmax": 10.0})
        nc = nowcast.nowcast(w, fc, pd.Timestamp("2023-02-14"), _cfg(FORECAST_ERR_MODEL_PATH=path)).set_index("stn").loc["108"]
    assert abs(nc["HDD_fc"] - 112) < 1e-9 and abs(nc["HDD_sd"] - 6.0) < 1e-9


# ---------- 사전 등록 ----------
from swi import prereg  # noqa: E402


def test_exclusion_rule():
    assert "081660" in C.FIRMS_ALL and "081660" not in C.FIRMS and len(C.FIRMS) == len(C.FIRMS_ALL) - 1
    assert "081660" in C.EXCLUDED


def test_prereg_stage1_hand_calc():
    from scipy import stats
    spec = C.PREREG_STAGE1
    res = pd.DataFrame([
        dict(group="gas", y="rev_g", x="HDD_yoy_100", year_fe=True, coef=0.3, se=0.1, t=3.0, p=0.0, nobs=200, n_periods=40),
        dict(group="beverage", y="rev_g", x="CDD_yoy_10", year_fe=True, coef=0.21, se=0.1, t=2.1, p=0.0, nobs=150, n_periods=40),
        dict(group="apparel", y="rev_g", x="TAVG_yoy_Q4", year_fe=True, coef=-0.25, se=0.1, t=-2.5, p=0.0, nobs=150, n_periods=40),
        dict(group="retail", y="rev_g", x="TAVG_yoy_Q4", year_fe=True, coef=0.25, se=0.1, t=2.5, p=0.0, nobs=150, n_periods=40),
        # 미끼: 사전 지정이 아닌 모형(연도더미 없음)은 t=10이어도 판정에 쓰이면 안 됨
        dict(group="retail", y="rev_g", x="TAVG_yoy_Q4", year_fe=False, coef=-1.0, se=0.1, t=-10.0, p=0.0, nobs=150, n_periods=40),
        dict(group="retail", y="rev_g", x="TAVG_yoy_Q1", year_fe=True, coef=-1.0, se=0.1, t=-10.0, p=0.0, nobs=150, n_periods=40),
    ])
    s = prereg.stage1(res, spec, 0.10, 2.0).set_index("group")
    p = {g: stats.t.sf(t * spec[g]["sign"], 39) for g, t in [("gas", 3.0), ("beverage", 2.1), ("apparel", -2.5), ("retail", 2.5)]}
    # BH 손계산: 작은 순서 gas(1) < apparel(2) < beverage(3) < retail(4), q_i = min_{j>=i} p_(j)·4/j
    q_ret = p["retail"]
    q_bev = min(p["beverage"] * 4 / 3, q_ret)
    q_app = min(p["apparel"] * 4 / 2, q_bev)
    q_gas = min(p["gas"] * 4 / 1, q_app)
    for g, q in [("gas", q_gas), ("apparel", q_app), ("beverage", q_bev), ("retail", q_ret)]:
        assert abs(s.loc[g, "q_bh"] - q) < 1e-12 and abs(s.loc[g, "p_one"] - p[g]) < 1e-12
    assert s.loc["gas", "pass"] and s.loc["beverage", "pass"] and s.loc["apparel", "pass"]
    assert not s.loc["retail", "pass"] and not s.loc["retail", "sign_ok"]   # 부호 반대 → 미통과 (미끼 무시)


def test_prereg_stage2_only_passed_groups():
    from swi import decision
    rng = np.random.default_rng(0)
    rows = []
    for ym in pd.period_range("2022-01", "2023-12", freq="M"):
        for g in ("gas", "retail"):
            for i in range(6):
                sc = float(i)
                # gas: 점수 높은 종목이 매달 1%p 더 오름 / retail: 점수와 무관한 잡음
                r = (0.01 * (i >= 4) - 0.01 * (i < 2)) if g == "gas" else rng.normal(0, 0.05)
                rows.append(dict(ym=str(ym), code=f"{g}{i}", group=g, SWI=sc, fwd_xret=r + rng.normal(0, 0.001)))
    d = pd.DataFrame(rows)
    s2 = prereg.stage2(d, ["gas"], C.PREREG_STAGE2, decision.long_short_within, decision.summarize, 30)
    assert s2["verdict"] == "통과" and s2["groups"] == ["gas"] and s2["months"] == 24
    none = prereg.stage2(d, [], C.PREREG_STAGE2, decision.long_short_within, decision.summarize, 30)
    assert none["verdict"] == "해당 없음"
