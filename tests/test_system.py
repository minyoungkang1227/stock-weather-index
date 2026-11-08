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


def test_nowcast_hand_calc():
    dates = pd.date_range("1980-01-01", "2023-02-13")
    w = pd.DataFrame({"stn": 108, "date": dates, "tavg": 10.0, "tmin": 5.0, "tmax": 15.0, "rain": np.nan})
    fc = pd.DataFrame({"issued_at": "202302140600", "stn": 108, "lead_day": range(3, 11),
                       "target_date": pd.date_range("2023-02-17", "2023-02-24").strftime("%Y-%m-%d"),
                       "tmin": 0.0, "tmax": 10.0})
    nc = nowcast.nowcast(w, fc, pd.Timestamp("2023-02-14"), C).set_index("stn").loc["108"]
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
