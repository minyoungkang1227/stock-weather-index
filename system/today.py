"""오늘의 Stock Weather Index 산출 (일별).

월별 백테스트에서 고르고 저장한 상승확률 모델(models/p_model.joblib)과 업종별 수혜 규모(models/b_scale.json)를 쓰고,
날씨는 '관측 + 예보 + 평년' 나우캐스트로 계산합니다.
데이터가 오래됐거나 이상하면 점수를 내지 않고 GateError로 멈춥니다 (틀린 점수를 내는 것보다 안 내는 것이 낫기 때문).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import config as C
from core import quality
from swi import decision, engine, index
from system import nowcast


class GateError(RuntimeError):
    pass


def gate(w: pd.DataFrame, px: pd.DataFrame, asof: pd.Timestamp) -> None:
    w = w.copy()
    w["date"] = pd.to_datetime(w["date"])
    recent = w[w["date"] >= asof - pd.Timedelta(days=400)]
    bad = quality.weather_outliers(recent)
    if len(bad):
        raise GateError(f"최근 기상자료 이상치 {len(bad)}건 (예: {bad.iloc[0][['stn', 'date', 'tavg', 'tmin', 'tmax']].to_dict()})")
    last = recent.dropna(subset=["tavg"]).groupby("stn")["date"].max()
    stale = last[last < asof - pd.Timedelta(days=C.GATE_WEATHER_STALE_DAYS)]
    if len(stale) or len(last) == 0:
        raise GateError(f"기상 관측이 {C.GATE_WEATHER_STALE_DAYS}일 넘게 갱신되지 않은 지점: {stale.to_dict() or '전체'}"
                        " (공공데이터포털 키로 매일 수집되는지 확인)")
    p_last = pd.to_datetime(px["date"]).max()
    if p_last < asof - pd.Timedelta(days=C.GATE_PRICE_STALE_DAYS):
        raise GateError(f"주가가 {p_last.date()} 이후 갱신되지 않음")


def price_features(px: pd.DataFrame, mkt: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """월별 모델과 같은 의미의 특징: 최근 21·63거래일 시장 대비 수익률, 거래량 변화."""
    k = mkt.copy()
    k["date"] = pd.to_datetime(k["date"])
    k = k[k["date"] <= asof].sort_values("date").set_index("date")["close"]
    rows = []
    for code, g in px.groupby("code"):
        g = g.assign(date=pd.to_datetime(g["date"]))
        g = g[g["date"] <= asof].sort_values("date").set_index("date")
        if len(g) < 70:
            continue
        def xr(n):
            r_i = g["close"].iloc[-1] / g["close"].iloc[-1 - n] - 1
            mk = k.reindex(g.index).ffill()
            return r_i - (mk.iloc[-1] / mk.iloc[-1 - n] - 1)
        v = g["volume"] if "volume" in g else pd.Series(np.nan, index=g.index)
        vol = np.log(v.iloc[-21:].mean() / v.iloc[-273:-21].mean()) if v.notna().sum() > 100 else np.nan
        rows.append(dict(code=code, mom1=xr(21), mom3=xr(63), vol_chg=vol, price_date=g.index[-1].strftime("%Y-%m-%d")))
    return pd.DataFrame(rows)


def score(asof: pd.Timestamp, prep: dict, fc: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    import joblib
    gate(prep["w"], prep["px"], asof)
    nc = nowcast.nowcast(prep["w"], fc, asof, C)
    nci = nc.set_index("stn")
    kq = engine.known_quarter(asof, C.DISCLOSURE_LAG_DAYS)
    pf = price_features(prep["px"], prep["mkt"], asof).set_index("code")
    mpath, spath = f"{C.MODEL_DIR}/p_model.joblib", f"{C.MODEL_DIR}/b_scale.json"
    bundle = joblib.load(mpath) if os.path.exists(mpath) else None
    scales = json.load(open(spath)) if os.path.exists(spath) else {}
    rows = []
    for code, (name, group, stn, _) in C.FIRMS.items():
        I = C.STRESS_SPEC[group]
        sc = engine.SCALE[I]
        if str(stn) not in nci.index:
            continue
        w = nci.loc[str(stn)]
        q = int(w["quarter"])
        sens = prep["get_sens"](group, kq).get(code)
        b, se = (sens[0][q - 1], sens[1][q - 1]) if sens else (np.nan, np.nan)
        dW, sd = (w[f"{I}_exp"] - w[f"{I}_ly"]) / sc, w[f"{I}_sd"] / sc
        ben, ben_sd, conf = engine.benefit_and_confidence(b, se, dW, sd)
        z = lambda v: (w[f"{v}_exp"] - w[f"{v}_norm"]) / w[f"{v}_qsd"] if w[f"{v}_qsd"] and w[f"{v}_qsd"] > 0 else 0.0
        r = dict(asof=asof.strftime("%Y-%m-%d"), code=code, name=name, group=group, quarter=q, index_var=I,
                 obs_share=w["obs_share"], beta=b, beta_se=se, beta_t=b / se if se and se > 0 else np.nan,
                 dW=dW, dW_sd=sd, benefit=float(ben), benefit_sd=float(ben_sd), C=float(conf),
                 anom_z=z(I), rain_z=z("RAIN"), cold_z=z("COLD"),
                 w_obs=w[f"{I}_obs"], w_fc=w[f"{I}_fc"], w_clim=w[f"{I}_clim"], w_ly=w[f"{I}_ly"], w_norm=w[f"{I}_norm"],
                 n_obs=w["n_obs"], n_fc=w["n_fc"], n_clim=w["n_clim"])
        if code in pf.index:
            r.update(pf.loc[code].to_dict())
        sc_b = scales.get(group if C.SWI_WITHIN_GROUP else "_all") or 0.02
        r["B"] = 100 * stats.norm.cdf(ben / sc_b) if np.isfinite(ben) else np.nan
        rows.append(r)
    d = pd.DataFrame(rows)
    for c in ("benefit", "C", "beta_t", "anom_z", "rain_z", "cold_z", "obs_share"):
        d[c] = d[c].fillna(0.0)
    d["P"] = np.nan
    if bundle is not None:
        for gname in bundle["groups"][1:]:
            d[f"g_{gname}"] = (d["group"] == gname).astype(float)
        feats = bundle["features"]
        ok = d[feats].notna().all(axis=1) if all(f in d for f in feats) else pd.Series(False, index=d.index)
        if ok.any():
            d.loc[ok, "P"] = 100 * bundle["model"].predict_proba(d.loc[ok, feats].values)[:, 1]
        d["model"] = f"{bundle['kind']} (학습 ~{bundle['trained_through']})"
    d["SWI"] = index.combine(d["B"], d["P"], d["C"], C.SWI_WEIGHTS["benefit"], C.SWI_WEIGHTS["prob"])
    d = decision.flags(d, C.SWI_WATCH, C.SWI_WARN)
    return d, nc
