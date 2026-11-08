"""기온 시뮬레이션 엔진.

일평균기온을 '결정적 부분 + 확률적 부분'으로 나눕니다.
  T(t) = s(t) + X(t)
  s(t) = a + b*t + Σ_k [c_k sin(2πk·d/365.25) + d_k cos(2πk·d/365.25)]   (추세 + 계절 평균)
  X(t) = A(q) + F(t)
    F(t) = φ F(t-1) + σ(d) ε(t)     일 단위 빠른 변동 (OU 과정의 이산형, φ = e^{-κ})
    A(q) = τ_q · a(q),  a(q) = ρ a(q-1) + √(1-ρ²) η(q)   분기 단위 느린 변동 ("올겨울은 전체적으로 따뜻함")
  σ(d): 계절마다 다른 변동성 (잔차 크기를 푸리에 항으로 회귀)
  ε: 지점 간 상관을 가진 표준정규 (같은 날 서울·대구가 함께 추운 현상 반영)
일최고기온 = 일평균기온 + (같은 달 과거 '최고-평균' 차이에서 무작위 추출).

왜 이렇게 하는가:
- 기온은 계절 평균으로 되돌아가는 성질(평균회귀)이 있어 OU 과정이 표준적 모형입니다(날씨파생상품 문헌의 기본형).
- 겨울 기온 변동이 여름보다 커서 변동성을 계절별로 둡니다.
- 지점 간 상관을 무시하면 여러 지역 기업을 묶은 포트폴리오 위험이 과소평가됩니다.
- 일 단위 OU만 쓰면 하루하루의 편차가 며칠 안에 사라져, '분기 전체가 따뜻한' 상황의 확률을 크게 과소평가합니다.
  (기온의 장기기억 문제. 분기 지수의 분산이 실제보다 작아져 VaR이 낮게 나옴)
  그래서 분기 평균 편차의 분산 중 일 단위 OU로 설명되지 않는 부분을 τ_q로 따로 둡니다.
  표본외 검증에서 두 모형(OU만 / OU+분기충격)을 비교할 수 있습니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TWO_PI = 2 * np.pi


def _fourier(doy: np.ndarray, K: int) -> np.ndarray:
    cols = []
    for k in range(1, K + 1):
        cols += [np.sin(TWO_PI * k * doy / 365.25), np.cos(TWO_PI * k * doy / 365.25)]
    return np.column_stack(cols)


def _mean_design(tyears: np.ndarray, doy: np.ndarray, K: int) -> np.ndarray:
    return np.column_stack([np.ones_like(tyears), tyears, _fourier(doy, K)])


def fit_station(df: pd.DataFrame, K_mean: int = 2, K_vol: int = 2, ref_year: int = 2000) -> dict:
    """한 지점의 일자료(date, tavg, tmax)로 모수를 추정합니다."""
    d = df.dropna(subset=["tavg"]).sort_values("date")
    d = d.set_index("date").asfreq("D")  # 빠진 날은 NaN으로 두어 AR(1) 쌍에서 제외
    doy = d.index.dayofyear.values.astype(float)
    ty = (d.index.year.values + (doy - 1) / 365.25 - ref_year)
    y = d["tavg"].values
    ok = ~np.isnan(y)
    Xm = _mean_design(ty, doy, K_mean)
    beta, *_ = np.linalg.lstsq(Xm[ok], y[ok], rcond=None)
    resid = y - Xm @ beta

    # 분기 평균 편차(느린 성분)와 나머지(빠른 성분)로 분리
    rs = pd.Series(resid, index=d.index)
    key = np.asarray(d.index.year * 10 + d.index.quarter)
    qmean = rs.groupby(key).transform("mean")
    qcount = rs.groupby(key).transform("count")
    fast = (rs - qmean).values

    # AR(1): F_t = φ F_{t-1} + u_t   (연속된 두 날이 모두 있을 때만)
    x0, x1 = fast[:-1], fast[1:]
    same_q = key[:-1] == key[1:]  # 분기 경계를 넘는 쌍은 제외 (분기평균을 뺀 값끼리만 비교)
    pair = ~np.isnan(x0) & ~np.isnan(x1) & same_q
    phi_hat = float(np.dot(x0[pair], x1[pair]) / np.dot(x0[pair], x0[pair]))
    # 분기평균을 빼면 φ가 아래로 치우침 -> Marriott-Pope 보정: E[φ̂] ≈ φ − (1+3φ)/n
    nq = 91.3
    phi = float(np.clip((phi_hat * nq + 1) / (nq - 3), 0.0, 0.99))
    u = np.full_like(fast, np.nan)
    u[1:] = x1 - phi * x0

    # 계절 변동성: |u|에 푸리에 회귀 후 정규분포 기대값 보정(E|Z| = sqrt(2/π))
    uok = ~np.isnan(u)
    Xv = np.column_stack([np.ones(len(u)), _fourier(doy, K_vol)])
    gv, *_ = np.linalg.lstsq(Xv[uok], np.abs(u[uok]), rcond=None)
    sig = (Xv @ gv) / np.sqrt(2 / np.pi)
    z = u / sig

    # 분기 충격 τ_q: 관측된 분기평균 편차의 분산 - 일 단위 AR(1)이 만드는 분기평균 분산
    qm = rs.groupby(key).mean()[rs.groupby(key).count() >= 75]  # 결측 많은 분기 제외
    qq = (qm.index % 10).values
    sig2_q = pd.Series(sig ** 2).groupby(d.index.quarter.values).mean()
    tau = np.zeros(4)
    for k in range(1, 5):
        v_obs = np.var(qm.values[qq == k], ddof=1) if (qq == k).sum() > 2 else 0.0
        n = 91.0
        v_ar = sig2_q.get(k, np.nan) / (n * (1 - phi) ** 2)  # AR(1) 표본평균 분산(근사)
        tau[k - 1] = np.sqrt(max(v_obs - v_ar, 0.0))
    a = pd.Series(qm.values / np.where(tau[qq - 1] > 0, tau[qq - 1], np.nan), index=qm.index)
    yrs, qs = a.index // 10, a.index % 10
    order = yrs * 4 + qs
    a_s = pd.Series(a.values, index=order).sort_index()
    a_prev = a_s.reindex(a_s.index - 1).values
    okp = ~np.isnan(a_s.values) & ~np.isnan(a_prev)
    rho = float(np.dot(a_prev[okp], a_s.values[okp]) / np.dot(a_prev[okp], a_prev[okp])) if okp.sum() > 5 else 0.0
    rho = float(np.clip(rho, -0.9, 0.9))
    eta = pd.Series(np.nan, index=a_s.index)
    eta[a_s.index[okp]] = (a_s.values[okp] - rho * a_prev[okp]) / np.sqrt(1 - rho ** 2)

    # 최고-평균 차이의 월별 경험분포
    spread = {}
    if "tmax" in d:
        sp = (d["tmax"] - d["tavg"]).dropna()
        for m, g in sp.groupby(sp.index.month):
            spread[m] = g.values
    return dict(beta=beta, phi=phi, gvol=gv, K_mean=K_mean, K_vol=K_vol, ref_year=ref_year,
                spread=spread, z=pd.Series(z, index=d.index), tau=tau, rho=rho, eta=eta)


def fit_all(w: pd.DataFrame, **kw) -> tuple[dict, np.ndarray, list, np.ndarray]:
    """모든 지점을 추정하고, 지점 간 상관행렬 두 개(일 단위 충격, 분기 단위 충격)를 구합니다."""
    params, zs, es = {}, {}, {}
    for stn, g in w.groupby("stn"):
        p = fit_station(g, **kw)
        params[stn] = p
        zs[stn] = p["z"]
        es[stn] = p["eta"]
    stns = list(params)
    corr = pd.DataFrame(zs).dropna()[stns].corr().values
    E = pd.DataFrame(es).dropna()
    corrA = E[stns].corr().values if len(E) > 10 else np.eye(len(stns))
    return params, corr, stns, corrA


def _mean_and_sig(p: dict, dates: pd.DatetimeIndex, shift: float) -> tuple[np.ndarray, np.ndarray]:
    doy = dates.dayofyear.values.astype(float)
    ty = dates.year.values + (doy - 1) / 365.25 - p["ref_year"]
    mean = _mean_design(ty, doy, p["K_mean"]) @ p["beta"] + shift
    sig = (np.column_stack([np.ones(len(doy)), _fourier(doy, p["K_vol"])]) @ p["gvol"]) / np.sqrt(2 / np.pi)
    return mean, np.maximum(sig, 1e-3)


def simulate_year(params: dict, corr: np.ndarray, stns: list, year: int, n_paths: int = 5000,
                  seed: int = 0, shift: float = 0.0, base: float = 18.0, hot: float = 33.0,
                  burn_in: int = 90, corrA: np.ndarray | None = None,
                  quarter_component: bool = True) -> dict[str, np.ndarray]:
    """year 한 해를 n_paths번 시뮬레이션해 분기 지수를 돌려줍니다.
    반환: {"HDD","CDD","HOT","TAVG"} 각각 배열 [n_paths, n_stations, 4분기].
    shift: 계절 평균에 더할 기온 이동(℃). 기후 시나리오용.
    일자료 전체를 저장하지 않고 분기 합만 누적해 메모리를 아낍니다."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(corr + 1e-10 * np.eye(len(stns)))
    dates = pd.date_range(f"{year - 1}-12-31", periods=burn_in + 1, freq="-1D")[::-1]
    dates = dates.append(pd.date_range(f"{year}-01-01", f"{year}-12-31"))
    S = len(stns)
    mean = np.zeros((len(dates), S)); sig = np.zeros((len(dates), S)); phi = np.zeros(S)
    for j, s in enumerate(stns):
        mean[:, j], sig[:, j] = _mean_and_sig(params[s], dates, shift)
        phi[j] = params[s]["phi"]
    out = {k: np.zeros((n_paths, S, 4)) for k in ("HDD", "CDD", "HOT", "TAVG")}
    ndays = np.zeros(4)
    X = np.zeros((n_paths, S))
    # 분기 충격: 직전 해 4분기에서 정상분포로 시작해 1~4분기로 이어감
    use_q = quarter_component and all("tau" in params[s] for s in stns)
    if use_q:
        LA = np.linalg.cholesky((corrA if corrA is not None else np.eye(S)) + 1e-10 * np.eye(S))
        tau = np.array([params[s]["tau"] for s in stns])            # [S, 4]
        rho = np.array([params[s]["rho"] for s in stns])            # [S]
        a = rng.standard_normal((n_paths, S)) @ LA.T                # 직전 해 4분기
        A_q = []
        for k in range(4):
            a = rho * a + np.sqrt(1 - rho ** 2) * (rng.standard_normal((n_paths, S)) @ LA.T)
            A_q.append(a * tau[:, k])
    for i, dt in enumerate(dates):
        eps = rng.standard_normal((n_paths, S)) @ L.T
        X = phi * X + sig[i] * eps
        if dt.year != year:
            continue
        T = mean[i] + X
        if use_q:
            T = T + A_q[dt.quarter - 1]
        qi = dt.quarter - 1
        ndays[qi] += 1
        out["HDD"][:, :, qi] += np.maximum(base - T, 0)
        out["CDD"][:, :, qi] += np.maximum(T - base, 0)
        out["TAVG"][:, :, qi] += T
        tmax = np.empty_like(T)
        for j, s in enumerate(stns):
            pool = params[s]["spread"].get(dt.month)
            tmax[:, j] = T[:, j] + (rng.choice(pool, n_paths) if pool is not None and len(pool) else 8.0)
        out["HOT"][:, :, qi] += (tmax >= hot)
    out["TAVG"] /= ndays[None, None, :]
    return out


def national(sim: dict, stns: list, members: list) -> dict:
    """전국(6개 도시) 평균 지수를 시뮬레이션 결과에 'NATIONAL'로 추가할 때 쓰는 값."""
    idx = [stns.index(s) for s in members if s in stns]
    return {k: v[:, idx, :].mean(axis=1) for k, v in sim.items()}
