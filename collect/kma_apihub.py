"""기상청 API허브(apihub.kma.go.kr) 수집. 인증키: .env 의 KMA_APIHUB_KEY

API허브는 JSON이 아니라 텍스트로 응답합니다.
  #START7777 / 설명줄(# 1. NAME : 의미) / 데이터줄 / #7777END
데이터는 쉼표 구분(disp에 따라) 또는 공백 정렬이고, 결측은 -9, -99 같은 값입니다.
열 이름은 응답의 설명줄에서 읽고, 데이터 열 수와 맞는 목록을 고릅니다(추측해서 붙이지 않음).

- ASOS 일자료: kma_sfcdd.php (하루씩, stn=0이면 전 지점)
- 중기 기온예보 이력: config.KMA_FCST_TEMP_ENDPOINT (2001년 2월 이후 발표분, tmfc1~tmfc2 기간 조회)
반환 형식은 기존 수집 함수와 같습니다(분석 코드는 그대로).
"""
from __future__ import annotations

import os
import re
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests

BASE = "https://apihub.kma.go.kr/api/typ01/url"
MISSING = {-9.0, -99.0, -999.0, -9.9, -99.9}


def _decode(content: bytes) -> str:
    for enc in ("cp949", "euc-kr", "utf-8"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def parse_typ01(text: str) -> pd.DataFrame:
    """API허브 텍스트 응답 -> DataFrame (열 이름은 응답 설명줄에서)."""
    lines = text.splitlines()
    rows = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = [p.strip() for p in s.split(",")] if "," in s else s.split()
        if parts and parts[-1] == "=":
            parts = parts[:-1]
        rows.append(parts)
    if not rows:
        return pd.DataFrame()
    n = max(len(r) for r in rows)
    numbered = [m.group(1) for ln in lines for m in [re.match(r"#\s*\d+\.\s*([A-Z0-9_]+)\s*:", ln)] if m]
    candidates = []
    for ln in lines:  # "# REG_ID TM_FC ..." 처럼 열 이름만 나열한 줄
        if ln.startswith("#") and not re.match(r"#\s*\d+\.", ln):
            toks = ln.lstrip("#").split()
            if toks and all(re.fullmatch(r"[A-Z][A-Z0-9_]*", t) for t in toks) and len(set(toks)) == len(toks):
                candidates.append(toks)
    cols = next((c for c in candidates if len(c) == n), None)
    if cols is None and len(numbered) == n:
        cols = numbered
    if cols is None:
        raise ValueError(f"열 이름을 확정할 수 없습니다 (데이터 {n}열, 설명 {len(numbered)}개). 응답 형식을 확인하세요.")
    return pd.DataFrame([r + [None] * (n - len(r)) for r in rows], columns=cols)


def _num(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.where(~x.isin(MISSING))


def _get(endpoint: str, params: dict, key: str) -> str:
    r = requests.get(f"{BASE}/{endpoint}", params={**params, "authKey": key}, timeout=60)
    r.raise_for_status()
    t = _decode(r.content)
    if "#START7777" not in t:
        raise RuntimeError(f"예상과 다른 응답: {t[:200]!r}")
    return t


# ---------- ASOS 일자료 ----------
def asos_standard(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "stn": pd.to_numeric(df["STN"], errors="coerce").astype("Int64"),
        "date": pd.to_datetime(df["TM"].astype(str).str[:8], format="%Y%m%d", errors="coerce"),
        "tavg": _num(df["TA_AVG"]), "tmin": _num(df["TA_MIN"]), "tmax": _num(df["TA_MAX"]),
        "rain": _num(df["RN_DAY"]),
    })
    return out.dropna(subset=["stn", "date"])


def fetch_asos_daily(stations: list[int], start: str, end: str, key: str, cache_dir: str | None = None,
                     sleep: float = 0.15) -> pd.DataFrame:
    """[start, end] 기간을 하루씩 받아 지점 목록만 남깁니다(stn=0 한 번 호출로 전 지점).
    cache_dir 을 주면 연도별 CSV로 저장해 두고, 다시 실행하면 이미 받은 날짜는 건너뜁니다(중간에 끊겨도 이어받기)."""
    days = pd.date_range(start, end, freq="D")
    want = set(int(s) for s in stations)
    frames, failed = [], []
    by_year = {}
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        for y in sorted(set(days.year)):
            f = os.path.join(cache_dir, f"asos_{y}.csv")
            by_year[y] = pd.read_csv(f, parse_dates=["date"]) if os.path.exists(f) else pd.DataFrame()
    done = set()
    for y, d in by_year.items():
        if len(d):
            done |= set(d["date"].dt.strftime("%Y%m%d"))
    new = {}
    for day in days:
        ds = day.strftime("%Y%m%d")
        if ds in done:
            continue
        try:
            df = asos_standard(parse_typ01(_get("kma_sfcdd.php", {"tm": ds, "stn": 0, "help": 0, "disp": 0}, key)))
            new.setdefault(day.year, []).append(df[df["stn"].isin(want)])
            time.sleep(sleep)
        except Exception as e:
            failed.append((ds, str(e)[:100]))
        if cache_dir and day.is_year_end and new.get(day.year):
            _save_year(cache_dir, day.year, by_year, new)
    if cache_dir:
        for y in list(new):
            _save_year(cache_dir, y, by_year, new)
        frames = [d for d in by_year.values() if len(d)]
    else:
        frames = [x for v in new.values() for x in v]
    for f in failed[:20]:
        print(f"[asos] 실패 {f[0]}: {f[1]}")
    if len(failed) > 20:
        print(f"[asos] 실패 총 {len(failed)}일 (다시 실행하면 실패한 날만 다시 받습니다)")
    if not frames:
        return pd.DataFrame(columns=["stn", "date", "tavg", "tmin", "tmax", "rain"])
    out = pd.concat(frames, ignore_index=True)
    out = out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))]
    return out.drop_duplicates(["stn", "date"]).sort_values(["stn", "date"]).reset_index(drop=True)


def _save_year(cache_dir, y, by_year, new):
    if not new.get(y):
        return
    d = pd.concat([by_year.get(y, pd.DataFrame())] + new[y], ignore_index=True)
    d = d.drop_duplicates(["stn", "date"])
    d.to_csv(os.path.join(cache_dir, f"asos_{y}.csv"), index=False)
    by_year[y], new[y] = d, []


# ---------- 중기 기온예보 ----------
def forecast_standard(df: pd.DataFrame, reg_to_stn: dict) -> pd.DataFrame:
    """REG_ID -> 관측지점 번호로 바꾸고, 24시간 구간(A01)만 남겨 (issued_at, stn, lead_day, target_date, tmin, tmax)."""
    d = df[df["REG_ID"].isin(reg_to_stn) & (df.get("MOD", "A01") == "A01")].copy()
    if d.empty:
        return pd.DataFrame(columns=["issued_at", "stn", "lead_day", "target_date", "tmin", "tmax"])
    fc = pd.to_datetime(d["TM_FC"].astype(str).str[:12], format="%Y%m%d%H%M")
    ef = pd.to_datetime(d["TM_EF"].astype(str).str[:8], format="%Y%m%d")
    return pd.DataFrame({
        "issued_at": d["TM_FC"].astype(str).str[:12].values,
        "stn": d["REG_ID"].map(reg_to_stn).astype(int).values,
        "lead_day": (ef.dt.normalize() - fc.dt.normalize()).dt.days.values,
        "target_date": ef.dt.strftime("%Y-%m-%d").values,
        "tmin": _num(d["MIN"]).values, "tmax": _num(d["MAX"]).values,
    })


def fetch_mid_temperature_history(reg_map: dict, start: str, end: str, key: str, endpoint: str,
                                  cache_dir: str | None = None, sleep: float = 0.3) -> pd.DataFrame:
    """[start, end] 사이 발표된 중기 기온예보를 한 달씩 받습니다 (전 구역 받은 뒤 필요한 구역만).
    reg_map: {관측지점번호: 예보구역코드}. cache_dir 를 주면 월별 CSV로 저장해 이어받기."""
    reg_to_stn = {v: int(k) for k, v in reg_map.items()}
    months = pd.period_range(pd.Timestamp(start).to_period("M"), pd.Timestamp(end).to_period("M"), freq="M")
    frames, failed = [], []
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    for m in months:
        f = os.path.join(cache_dir, f"fcst_{m}.csv") if cache_dir else None
        if f and os.path.exists(f) and m < pd.Timestamp.today().to_period("M"):
            frames.append(pd.read_csv(f, dtype={"issued_at": str}))
            continue
        t1 = m.start_time.strftime("%Y%m%d") + "00"
        t2 = m.end_time.strftime("%Y%m%d") + "23"
        try:
            raw = parse_typ01(_get(endpoint, {"tmfc1": t1, "tmfc2": t2, "disp": 1, "help": 1}, key))
            std = forecast_standard(raw, reg_to_stn) if len(raw) else pd.DataFrame()
            if f is not None and len(std):
                std.to_csv(f, index=False)
            frames.append(std)
            time.sleep(sleep)
        except Exception as e:
            failed.append((str(m), str(e)[:100]))
    for x in failed:
        print(f"[forecast-hist] 실패 {x[0]}: {x[1]}")
    frames = [x for x in frames if len(x)]
    if not frames:
        return pd.DataFrame(columns=["issued_at", "stn", "lead_day", "target_date", "tmin", "tmax"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["issued_at", "stn", "target_date"])


def latest_tmfc10(now: datetime) -> str:
    """가장 최근 발표(06·18시) 시각, 10자리 YYYYMMDDHH."""
    if now.hour >= 18:
        t = now.replace(hour=18)
    elif now.hour >= 6:
        t = now.replace(hour=6)
    else:
        t = (now - timedelta(days=1)).replace(hour=18)
    return t.strftime("%Y%m%d%H")


def fetch_mid_temperature_latest(reg_map: dict, key: str, endpoint: str, now: datetime | None = None) -> pd.DataFrame:
    t = latest_tmfc10(now or datetime.now())
    raw = parse_typ01(_get(endpoint, {"tmfc1": t, "tmfc2": t, "disp": 1, "help": 1}, key))
    return forecast_standard(raw, {v: int(k) for k, v in reg_map.items()}) if len(raw) else pd.DataFrame()


def forecast_skill(fc: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """예보 일평균기온((최저+최고)/2) vs 관측 일평균기온의 오차를 리드(며칠 뒤 예보)별로.
    nowcast 의 FORECAST_ERR_SD(현재 2℃ 가정)를 이 실제 값으로 바꾸는 데 씁니다."""
    f = fc.copy()
    f["date"] = pd.to_datetime(f["target_date"])
    f["fc_tavg"] = (f["tmin"] + f["tmax"]) / 2
    o = obs[["stn", "date", "tavg"]].copy()
    o["stn"] = o["stn"].astype(int)
    m = f.merge(o, on=["stn", "date"], how="inner")
    m["err"] = m["fc_tavg"] - m["tavg"]
    return m.groupby("lead_day")["err"].agg(n="count", bias="mean", sd="std",
                                            mae=lambda e: np.mean(np.abs(e))).reset_index()
