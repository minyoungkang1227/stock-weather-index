"""키 4개가 제대로 들어갔는지 한 번씩만 호출해 확인합니다. 키 값은 화면에 절대 출력하지 않습니다.
실행: python check_keys.py
"""
import os
from datetime import datetime

import requests
from dotenv import load_dotenv

import config as C
from collect import kma_apihub

load_dotenv()
KEYS = {k: (os.getenv(k) or "").strip() for k in ["DART_API_KEY", "ECOS_API_KEY", "KMA_APIHUB_KEY", "DATA_GO_KR_KEY"]}


def mask(msg):
    msg = str(msg)
    for v in KEYS.values():
        if v:
            msg = msg.replace(v, "***")
    return msg[:300]


def show(name, ok, detail):
    print(f"[{'OK ' if ok else '실패'}] {name:<28} {mask(detail)}")


def dart():
    k = KEYS["DART_API_KEY"]
    if not k:
        return show("OpenDART", False, "키 없음")
    j = requests.get("https://opendart.fss.or.kr/api/company.json", params={"crtfc_key": k, "corp_code": "00126380"}, timeout=30).json()
    show("OpenDART", j.get("status") == "000", f"{j.get('status')} {j.get('message')} / {j.get('corp_name', '')} (길이 {len(k)}자, 정상 40자)")


def ecos():
    k = KEYS["ECOS_API_KEY"]
    if not k:
        return show("ECOS", False, "키 없음")
    j = requests.get(f"https://ecos.bok.or.kr/api/StatisticTableList/{k}/json/kr/1/1/", timeout=30).json()
    ok = "StatisticTableList" in j
    show("ECOS", ok, "통계표 목록 응답 받음" if ok else j.get("RESULT", j))


def kma_asos():
    k = KEYS["KMA_APIHUB_KEY"]
    if not k:
        return show("API허브 관측(ASOS)", False, "키 없음")
    raw = kma_apihub._get("kma_sfcdd.php", {"tm": "20240101", "stn": 0, "help": 1}, k)
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/check_asos_raw.txt", "w", encoding="utf-8") as f:   # 응답 원문(키 없음) — 형식 확인용
        f.write(raw)
    d = kma_apihub.asos_standard(kma_apihub.parse_typ01(raw))
    d = d[d["stn"] == 108]
    r = d.iloc[0] if len(d) else None
    show("API허브 관측(ASOS)", r is not None and r["tavg"] == 3.3,
         "응답 0줄" if r is None else f"서울 2024-01-01 평균기온 {r['tavg']} (기대값 3.3)")


def kma_fcst():
    k = KEYS["KMA_APIHUB_KEY"]
    if not k:
        return show("API허브 기온예보", False, "키 없음")
    f = kma_apihub.fetch_mid_temperature_latest(C.FORECAST_REG, k, C.KMA_FCST_TEMP_ENDPOINT)
    show(f"API허브 기온예보 ({C.KMA_FCST_TEMP_ENDPOINT})", len(f) > 0,
         f"{len(f)}행, 지점 {sorted(f['stn'].unique()) if len(f) else []}, 발표 {kma_apihub.latest_tmfc10(datetime.now())}")


for fn in (dart, ecos, kma_asos, kma_fcst):
    try:
        fn()
    except Exception as e:
        show(fn.__name__, False, f"{type(e).__name__}: {e}")
print("\n'실패'가 있으면 이 화면을 그대로 캡처해서 보내 주세요 (키는 *** 로 가려져 있습니다).")
