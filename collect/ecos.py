"""한국은행 ECOS 품목별 소비자물가 수집 (도시가스 요금 변화 통제용).

통계표 코드와 품목 코드는 직접 적지 않고, 품목 목록 API에서 이름으로 찾습니다.
찾지 못하면 멈추고 후보 목록을 보여줍니다(추측해서 다른 품목을 쓰지 않음).
"""
from __future__ import annotations

import pandas as pd
import requests

BASE = "https://ecos.bok.or.kr/api"
CPI_ITEM_TABLE = "901Y009"  # 소비자물가지수(품목별). 다르면 config에서 바꾸세요.


def item_list(key: str, table: str = CPI_ITEM_TABLE) -> pd.DataFrame:
    url = f"{BASE}/StatisticItemList/{key}/json/kr/1/2000/{table}"
    js = requests.get(url, timeout=30).json()
    if "StatisticItemList" not in js:
        raise RuntimeError(f"ECOS 품목 목록 오류: {js}")
    return pd.DataFrame(js["StatisticItemList"]["row"])


def find_item(items: pd.DataFrame, name: str) -> str:
    hit = items[items["ITEM_NAME"].str.strip() == name]
    if hit.empty:
        cand = items[items["ITEM_NAME"].str.contains(name[:2], na=False)]["ITEM_NAME"].unique()[:20]
        raise LookupError(f"'{name}' 품목을 찾지 못했습니다. 비슷한 후보: {list(cand)}")
    return hit.iloc[0]["ITEM_CODE"]


def fetch_monthly(key: str, item_code: str, start: str, end: str, table: str = CPI_ITEM_TABLE) -> pd.DataFrame:
    """start/end 는 'YYYYMM'. (month, value) 반환."""
    url = f"{BASE}/StatisticSearch/{key}/json/kr/1/10000/{table}/M/{start}/{end}/{item_code}"
    js = requests.get(url, timeout=30).json()
    if "StatisticSearch" not in js:
        raise RuntimeError(f"ECOS 조회 오류: {js}")
    df = pd.DataFrame(js["StatisticSearch"]["row"])
    return pd.DataFrame({"month": pd.to_datetime(df["TIME"], format="%Y%m"),
                         "value": pd.to_numeric(df["DATA_VALUE"], errors="coerce")})
