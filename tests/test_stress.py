"""스트레스 테스트 모듈 스모크테스트 (손계산 대조, 합성 데이터는 실제 자료 아님)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import stress, weather_sim  # noqa: E402


def _synthetic_station(phi=0.7, sigma=2.0, years=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1990-01-01", periods=365 * years, freq="D")
    doy = dates.dayofyear.values
    mean = 13 + 13 * np.sin(2 * np.pi * (doy - 105) / 365.25)
    x = np.zeros(len(dates))
    for t in range(1, len(dates)):
        x[t] = phi * x[t - 1] + sigma * rng.standard_normal()
    tavg = mean + x
    return pd.DataFrame({"stn": 108, "date": dates, "tavg": tavg, "tmax": tavg + 5})


def test_fit_recovers_ou_parameters():
    p = weather_sim.fit_station(_synthetic_station())
    assert abs(p["phi"] - 0.7) < 0.02
    sig = p["gvol"][0] / np.sqrt(2 / np.pi)  # 계절 변동 없는 경우 상수항이 곧 σ
    assert abs(sig - 2.0) < 0.1
    assert abs(p["beta"][1]) < 0.05  # 추세 없음


def test_simulation_deterministic_case():
    # φ=0, 변동성≈0, 평균 10℃ 고정 -> 2027년 1분기(90일) HDD = 8×90 = 720
    p = dict(beta=np.array([10.0, 0, 0, 0, 0, 0]), phi=0.0, gvol=np.array([1e-9, 0, 0, 0, 0]),
             K_mean=2, K_vol=2, ref_year=2000, spread={m: np.array([5.0]) for m in range(1, 13)})
    sim = weather_sim.simulate_year({108: p}, np.eye(1), [108], 2027, n_paths=3)
    assert np.allclose(sim["HDD"][:, 0, 0], 720, atol=0.1)  # 변동성 하한(1e-3) 때문에 아주 작은 오차 허용
    assert np.allclose(sim["TAVG"][:, 0, :], 10, atol=0.01)
    assert sim["HOT"].sum() == 0  # 최고 15℃


def test_parse_x():
    assert stress.parse_x("HDD_yoy_100") == ("HDD", None)
    assert stress.parse_x("TAVG_yoy_Q4") == ("TAVG", 4)
    assert stress.parse_x("HOT_yoy") == ("HOT", None)


def test_firm_impacts_hand_calc():
    # 1분기만: 기준매출 100, 난방도일 +100, 매출계수 0.04, 영업이익계수 0.002
    betas = {"rev_g": {1: ("HDD", 0.04, 0.0)}, "op_chg": {1: ("HDD", 0.002, 0.0)}}
    W = np.array([[600.0, 0, 0, 0]])
    imp = stress.firm_impacts(W, np.array([500.0, 0, 0, 0]), np.array([100.0, 0, 0, 0]), betas, "HDD", None)
    assert abs(imp["revenue"][0] - 100 * (np.exp(0.04) - 1)) < 1e-12
    assert abs(imp["op_income"][0] - 0.2) < 1e-12


def test_risk_metrics_hand_calc():
    m = stress.risk_metrics(-np.arange(1, 101, dtype=float))  # 손실 1..100
    assert abs(m["VaR95"] - 95.05) < 1e-9          # numpy 선형보간 분위수
    assert abs(m["ES95"] - 98.0) < 1e-9            # 96~100 평균


def test_kupiec():
    lr, p = stress.kupiec(100, 10, 0.10)
    assert abs(lr) < 1e-9 and abs(p - 1) < 1e-9
    lr, _ = stress.kupiec(100, 20, 0.10)
    expect = -2 * (80 * np.log(0.9) + 20 * np.log(0.1) - 80 * np.log(0.8) - 20 * np.log(0.2))
    assert abs(lr - expect) < 1e-9


def test_coverage():
    sims = np.arange(1, 101, dtype=float)[:, None].repeat(3, axis=1)
    out, n = stress.coverage(np.array([0.0, 50.0, 200.0]), sims)
    assert (out, n) == (2, 3)


def test_fit_recovers_quarter_shock():
    """분기 충격 τ=1.5℃를 심은 합성 자료에서 τ를 복원하는지 (실제 자료 아님)."""
    rng = np.random.default_rng(3)
    dates = pd.date_range("1960-01-01", "2019-12-31", freq="D")
    doy = dates.dayofyear.values
    mean = 13 + 13 * np.sin(2 * np.pi * (doy - 105) / 365.25)
    shocks = {k: 1.5 * rng.standard_normal() for k in set(dates.year * 10 + dates.quarter)}
    A = np.array([shocks[k] for k in dates.year * 10 + dates.quarter])
    x = np.zeros(len(dates))
    for t in range(1, len(dates)):
        x[t] = 0.7 * x[t - 1] + 2.0 * rng.standard_normal()
    df = pd.DataFrame({"stn": 108, "date": dates, "tavg": mean + A + x, "tmax": mean + A + x + 5})
    p = weather_sim.fit_station(df)
    assert abs(p["phi"] - 0.7) < 0.03
    assert np.all(np.abs(p["tau"] - 1.5) < 0.45)   # 분기당 표본 60개라 오차 허용
    assert abs(p["rho"]) < 0.25                      # 분기 충격 간 상관 없음
