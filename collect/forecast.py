"""기상청 중기기온예보 수집 (공공데이터포털 '기상청_중기예보 조회서비스', DATA_GO_KR_KEY 사용).

하루 두 번(06시·18시) 발표되는 3~10일 뒤 최저·최고기온을 지점별로 받아 '발표 시각'과 함께 저장합니다.
과거 예보 이력은 공개 API로 구하기 어려워서, 오늘부터 매일 쌓아야 나중에 '예보를 썼다면 어땠나'를 검증할 수 있습니다.
반환 표준 형식: issued_at, stn, lead_day, target_date, tmin, tmax
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

import pandas as pd
import requests

URL = "http://apis.data.go.kr/1360000/MidFcstInfoService/getMidTa"


def latest_tmfc(now: datetime) -> str:
    """가장 최근 발표 시각 문자열(YYYYMMDDHHMM). 06시 전이면 전날 18시."""
    if now.hour >= 18:
        t = now.replace(hour=18, minute=0)
    elif now.hour >= 6:
        t = now.replace(hour=6, minute=0)
    else:
        t = (now - timedelta(days=1)).replace(hour=18, minute=0)
    return t.strftime("%Y%m%d%H%M")


def parse_item(item: dict, stn: int, tmfc: str) -> pd.DataFrame:
    """응답의 taMinN / taMaxN 항목을 (N일 뒤) 행으로 펼칩니다. 버전마다 제공 일수가 달라 이름으로 찾습니다."""
    base = datetime.strptime(tmfc[:8], "%Y%m%d")
    leads = sorted({int(m.group(1)) for k in item for m in [re.fullmatch(r"taM(?:in|ax)(\d+)", k)] if m})
    rows = []
    for n in leads:
        tmin, tmax = item.get(f"taMin{n}"), item.get(f"taMax{n}")
        if tmin in (None, "") or tmax in (None, ""):
            continue
        rows.append(dict(issued_at=tmfc, stn=stn, lead_day=n, target_date=(base + timedelta(days=n)).strftime("%Y-%m-%d"),
                         tmin=float(tmin), tmax=float(tmax)))
    return pd.DataFrame(rows)


def fetch_mid_temperature(reg_map: dict, key: str, now: datetime | None = None) -> pd.DataFrame:
    """지점번호 -> 예보구역코드 표만 받아 최신 발표분을 수집. 한 지점이 실패해도 나머지는 계속."""
    tmfc = latest_tmfc(now or datetime.now())
    frames, failed = [], []
    for stn, reg in reg_map.items():
        try:
            r = requests.get(URL, timeout=30, params={"serviceKey": key, "pageNo": 1, "numOfRows": 10,
                                                      "dataType": "JSON", "regId": reg, "tmFc": tmfc})
            r.raise_for_status()
            body = r.json()["response"]
            if body["header"]["resultCode"] != "00":
                raise RuntimeError(body["header"].get("resultMsg"))
            item = body["body"]["items"]["item"][0]
            frames.append(parse_item(item, stn, tmfc))
        except Exception as e:
            failed.append((stn, reg, str(e)[:100]))
    for f in failed:
        print(f"[forecast] 실패 지점 {f[0]}({f[1]}): {f[2]}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["issued_at", "stn", "lead_day", "target_date", "tmin", "tmax"])
