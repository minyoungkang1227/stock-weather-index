"""기상 일자료 수집.
(강수량 rain: 기상청 자료에서 강수가 없는 날은 빈칸이므로 분석 단계에서 0으로 봅니다.)

수집 함수는 '지점 + 기간'만 받아 표준 형식(stn, date, tavg, tmin, tmax)을 반환합니다.
한 지점이 실패해도 나머지 지점은 계속 받습니다.

경로 두 가지:
  A) 기상자료개방포털(data.kma.go.kr)에서 내려받은 CSV를 읽기 (키 불필요)
  B) 공공데이터포털 '기상청_지상(종관, ASOS) 일자료 조회서비스' API (DATA_GO_KR_KEY 필요)
"""
from __future__ import annotations

import glob
import time
from datetime import date

import pandas as pd
import requests

STD_COLS = ["stn", "date", "tavg", "tmin", "tmax", "rain"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=STD_COLS)


# ---------- A) 기상자료개방포털 CSV ----------
_CSV_MAP = {
    "지점": "stn", "일시": "date",
    "평균기온(°C)": "tavg", "최저기온(°C)": "tmin", "최고기온(°C)": "tmax",
}
_CSV_OPTIONAL = {"일강수량(mm)": "rain"}  # 없으면 결측으로 둠


def _read_any_encoding(path: str) -> pd.DataFrame:
    for enc in ("cp949", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"인코딩을 읽지 못했습니다: {path}")


def read_kma_csv(pattern: str) -> pd.DataFrame:
    """data/raw/kma/*.csv 처럼 패턴에 맞는 CSV를 모두 읽어 표준 형식으로 합칩니다."""
    frames = []
    for path in sorted(glob.glob(pattern)):
        df = _read_any_encoding(path)
        df.columns = [c.strip() for c in df.columns]
        missing = [k for k in _CSV_MAP if k not in df.columns]
        if missing:
            print(f"[weather] 건너뜀 {path}: 필요한 열 없음 {missing}")
            continue
        keep = list(_CSV_MAP) + [c for c in _CSV_OPTIONAL if c in df.columns]
        df = df[keep].rename(columns={**_CSV_MAP, **_CSV_OPTIONAL})
        if "rain" not in df:
            df["rain"] = float("nan")
        frames.append(df)
    if not frames:
        return _empty()
    return _standardize(pd.concat(frames, ignore_index=True))


# ---------- B) 공공데이터포털 API ----------
_URL = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"


def _fetch_year(stn: int, year: int, key: str, last_day: date) -> pd.DataFrame:
    start = date(year, 1, 1)
    end = min(date(year, 12, 31), last_day)
    if end < start:
        return _empty()
    params = {
        "serviceKey": key, "pageNo": 1, "numOfRows": 999, "dataType": "JSON",
        "dataCd": "ASOS", "dateCd": "DAY",
        "startDt": start.strftime("%Y%m%d"), "endDt": end.strftime("%Y%m%d"),
        "stnIds": stn,
    }
    r = requests.get(_URL, params=params, timeout=30)
    r.raise_for_status()
    body = r.json()["response"]
    code = body["header"]["resultCode"]
    if code != "00":
        raise RuntimeError(f"API 오류 {code}: {body['header'].get('resultMsg')}")
    items = body["body"]["items"]["item"]
    df = pd.DataFrame(items)
    if "sumRn" not in df:
        df["sumRn"] = ""
    df = df[["stnId", "tm", "avgTa", "minTa", "maxTa", "sumRn"]]
    df.columns = STD_COLS
    return df


def fetch_asos(stations: list[int], start_year: int, end_year: int, key: str) -> pd.DataFrame:
    """지점 목록과 연도 범위만 받아 일자료를 반환합니다. API는 '어제 이전'까지만 제공합니다."""
    last_day = date.today() - pd.Timedelta(days=2)
    frames, failed = [], []
    for stn in stations:
        for y in range(start_year, end_year + 1):
            try:
                frames.append(_fetch_year(stn, y, key, last_day))
                time.sleep(0.2)
            except Exception as e:  # 한 지점·연도 실패해도 계속
                failed.append((stn, y, str(e)[:120]))
    for f in failed:
        print(f"[weather] 실패 지점 {f[0]} {f[1]}년: {f[2]}")
    if not frames:
        return _empty()
    return _standardize(pd.concat(frames, ignore_index=True))


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["stn"] = pd.to_numeric(df["stn"], errors="coerce").astype("Int64")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "rain" not in df:
        df["rain"] = float("nan")
    for c in ("tavg", "tmin", "tmax", "rain"):
        df[c] = pd.to_numeric(df[c].replace("", None), errors="coerce")
    df = df.dropna(subset=["stn", "date"]).drop_duplicates(["stn", "date"], keep="last")
    return df.sort_values(["stn", "date"]).reset_index(drop=True)[STD_COLS]
