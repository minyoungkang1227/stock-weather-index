"""스모크테스트: 손으로 계산한 기대값과 코드 결과가 같은지 확인합니다.
실행: python -m pytest -q tests   (또는 python tests/test_smoke.py)"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collect.dart import cumulative_to_quarterly  # noqa: E402
from core import indices, panel, quality, regress  # noqa: E402


def test_daily_indices():
    # 평균기온 10, 20, 18 -> HDD 8, 0, 0 / CDD 0, 2, 0. 최고기온 33, 32.9, 35 -> 폭염 1, 0, 1
    df = pd.DataFrame({"stn": 108, "date": pd.date_range("2020-01-01", periods=3),
                       "tavg": [10, 20, 18], "tmin": [5, 15, 10], "tmax": [33, 32.9, 35]})
    d = indices.daily_indices(df)
    assert d["HDD"].tolist() == [8, 0, 0]
    assert d["CDD"].tolist() == [0, 2, 0]
    assert d["HOT"].tolist() == [1, 0, 1]


def test_quarterly_and_missing_rule():
    # 2021Q1은 90일. 모든 날 평균 8℃ -> HDD = 10 * 90 = 900
    dates = pd.date_range("2021-01-01", "2021-03-31")
    df = pd.DataFrame({"stn": 108, "date": dates, "tavg": 8.0, "tmin": 3.0, "tmax": 12.0})
    q = indices.quarterly(indices.daily_indices(df))
    assert len(dates) == 90 and abs(q.loc[0, "HDD"] - 900) < 1e-9
    # 5일 결측(5.6% < 10%) -> 남은 85일 합 850을 90/85로 보정하면 900
    q2 = indices.quarterly(indices.daily_indices(df.iloc[5:]))
    assert abs(q2.loc[0, "HDD"] - 900) < 1e-9
    # 10일 결측(11% > 10%) -> NaN
    q3 = indices.quarterly(indices.daily_indices(df.iloc[10:]))
    assert np.isnan(q3.loc[0, "HDD"])


def test_normal_has_no_lookahead():
    # 값 = 연도. y년 평년 = (y-30..y-1) 평균 = y - 15.5
    years = range(1980, 2031)
    q = pd.DataFrame({"stn": "108", "year": list(years), "q": 1,
                      "HDD": [float(y) for y in years], "CDD": 0.0, "HOT": 0.0, "TAVG": 0.0})
    n = indices.add_normals(q, 30, 20).set_index("year")
    assert abs(n.loc[2020, "HDD_normal"] - 2004.5) < 1e-9
    # 2020년 이후 값을 바꿔도 2020년 평년은 그대로여야 함
    q2 = q.copy()
    q2.loc[q2["year"] >= 2020, "HDD"] = 1e6
    n2 = indices.add_normals(q2, 30, 20).set_index("year")
    assert abs(n2.loc[2020, "HDD_normal"] - 2004.5) < 1e-9
    # 앞쪽 20년 미만이면 NaN
    assert np.isnan(n.loc[1999, "HDD_normal"]) and not np.isnan(n.loc[2000, "HDD_normal"])
    assert n.loc[2021, "HDD_yoy"] == 1.0


def test_dart_cumulative_to_quarterly():
    # 누적 100, 250, 330, 500 -> 분기 100, 150, 80, 170
    cum = pd.DataFrame({"code": "X", "year": 2022, "q": [1, 2, 3, 4], "fs_div": "CFS",
                        "rev_cum": [100, 250, 330, 500], "op_cum": [10, 5, 20, 15]})
    qf = cumulative_to_quarterly(cum).set_index("q")
    assert qf["revenue"].tolist() == [100, 150, 80, 170]
    assert qf["op_income"].tolist() == [10, -5, 15, -5]
    # 반기 누락 -> 3분기는 추정하지 않고 NaN
    qf2 = cumulative_to_quarterly(cum[cum["q"] != 2]).set_index("q")
    assert np.isnan(qf2.loc[3, "revenue"]) and qf2.loc[4, "revenue"] == 170
    # 같은 해에 연결/별도가 섞이면 차분하지 않음
    mixed = cum.assign(fs_div=["CFS", "OFS", "CFS", "CFS"])
    assert np.isnan(cumulative_to_quarterly(mixed).set_index("q").loc[3, "revenue"])


def test_growth_and_returns():
    qf = pd.DataFrame({"code": "X", "year": [2021, 2022], "q": 1,
                       "revenue": [100.0, 110.0], "op_income": [10.0, 12.0]})
    g = panel.fin_growth(qf).set_index("year")
    assert abs(g.loc[2022, "rev_g"] - np.log(1.1)) < 1e-12
    assert abs(g.loc[2022, "op_chg"] - 0.02) < 1e-12   # (12-10)/100
    # 분기말 종가 100 -> 110 (기업), 100 -> 105 (시장): 초과수익률 0.05
    px = pd.DataFrame({"date": pd.to_datetime(["2022-03-31", "2022-06-30"]), "code": "X", "close": [100, 110]})
    mk = pd.DataFrame({"date": pd.to_datetime(["2022-03-31", "2022-06-30"]), "close": [100, 105]})
    r = panel.quarterly_returns(px, mk).set_index("q")
    assert abs(r.loc[2, "xret"] - 0.05) < 1e-12
    assert abs(r.loc[1, "xret_next"] - 0.05) < 1e-12


def test_quality_flags():
    df = pd.DataFrame({"stn": 108, "date": pd.to_datetime(["2020-01-01", "2020-01-03"]),
                       "tavg": [5, 5], "tmin": [8, 0], "tmax": [3, 10]})
    assert len(quality.weather_outliers(df)) == 1          # 최저 > 최고
    assert quality.weather_missing(df).loc[0, "missing_days"] == 1  # 1/2 빠짐


def test_regression_recovers_known_beta():
    """합성 데이터(실제 시장 자료 아님): 참값 b=-0.03을 복원하는지."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):
        for y in range(2015, 2026):
            for q in (1, 2, 3, 4):
                x = rng.normal(0, 1)
                rows.append(dict(code=f"F{i}", year=y, q=q, x=x,
                                 y=0.01 * i + 0.005 * q - 0.03 * x + rng.normal(0, 0.01)))
    df = pd.DataFrame(rows)
    r = regress.panel_reg(df, "y", ["x"]).iloc[0]
    assert abs(r["coef"] + 0.03) < 3 * r["se"] and r["p"] < 0.001
    o = regress.oos_gain(df, "y", ["x"], 2021)
    assert o["oos_gain"] > 0.5
    # 날씨와 무관한 변수는 표본외 개선이 없어야 함
    df["noise"] = rng.normal(0, 1, len(df))
    assert regress.oos_gain(df, "y", ["noise"], 2021)["oos_gain"] < 0.05


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("통과", name)
