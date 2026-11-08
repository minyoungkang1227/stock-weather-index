"""데이터 수집. 사용법:  python run_collect.py [weather] [dart] [prices] [ecos]
인자 없이 실행하면 가능한 것을 모두 수집합니다. 키는 .env 에서 읽습니다."""
from __future__ import annotations

import os
import sys

import pandas as pd
from dotenv import load_dotenv

import config as C
from collect import dart, ecos, prices, weather
from core import quality

load_dotenv()
os.makedirs(C.RAW, exist_ok=True)
steps = set(sys.argv[1:]) or {"weather", "dart", "prices", "ecos", "disclosure"}


def save(df: pd.DataFrame, name: str) -> None:
    path = f"{C.RAW}/{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  -> {path} ({len(df):,}행)")


if "weather" in steps:
    print("[1] 기상 일자료")
    if any(os.scandir(f"{C.RAW}/kma")) if os.path.isdir(f"{C.RAW}/kma") else False:
        w = weather.read_kma_csv(f"{C.RAW}/kma/*.csv")
        print("  기상자료개방포털 CSV 사용")
    elif os.getenv("DATA_GO_KR_KEY"):
        w = weather.fetch_asos(list(C.STATIONS), int(C.WEATHER_START[:4]), C.ANALYSIS_END_YEAR,
                               os.environ["DATA_GO_KR_KEY"])
    else:
        w = None
        print("  건너뜀: data/raw/kma/ 에 CSV를 넣거나 .env 에 DATA_GO_KR_KEY 를 넣으세요.")
    if w is not None and not w.empty:
        save(w, "weather_daily")
        print(quality.weather_missing(w).to_string(index=False))
        print(f"  이상치 {len(quality.weather_outliers(w))}건")

if "dart" in steps:
    print("[2] OpenDART 분기 실적")
    if os.getenv("DART_API_KEY"):
        qf, cum = dart.fetch_quarterly(list(C.FIRMS), C.ANALYSIS_START_YEAR - 1, C.ANALYSIS_END_YEAR,
                                       os.environ["DART_API_KEY"])
        if not qf.empty:
            save(qf, "fin_quarterly")
            save(cum, "fin_cumulative_raw")
            for k, v in quality.fin_checks(qf, cum).items():
                print(f"  점검 {k}: {len(v)}건")
    else:
        print("  건너뜀: .env 에 DART_API_KEY 가 없습니다.")

if "prices" in steps:
    print("[3] 주가 (FinanceDataReader)")
    names = prices.listing_names()
    keep = []
    for code, (exp, *_rest) in C.FIRMS.items():
        actual = names.get(code)
        if actual is None or actual.replace(" ", "") != exp.replace(" ", ""):
            print(f"  [중단] {code}: 기대 '{exp}' / 실제 '{actual}' -> 이 종목은 수집하지 않습니다. config.py 확인")
        else:
            keep.append(code)
    start = f"{C.ANALYSIS_START_YEAR - 1}-10-01"
    px = prices.fetch_prices(keep, start)
    mkt = prices.fetch_prices([C.MARKET_INDEX], start)
    save(px, "prices")
    save(mkt.drop(columns="code"), "market")

if "ecos" in steps:
    print("[4] ECOS 품목별 소비자물가 (도시가스 요금 통제용)")
    key = os.getenv("ECOS_API_KEY")
    if key:
        try:
            items = ecos.item_list(key)
            code = ecos.find_item(items, "도시가스")
            m = ecos.fetch_monthly(key, code, f"{C.ANALYSIS_START_YEAR - 1}01", f"{C.ANALYSIS_END_YEAR}12")
            save(m, "cpi_citygas")
        except Exception as e:
            print(f"  실패: {e}")
    else:
        print("  건너뜀: .env 에 ECOS_API_KEY 가 없습니다.")

if "disclosure" in steps:
    print("[5] 실적 공시일 (이벤트 스터디용)")
    if os.getenv("DART_API_KEY"):
        dd = dart.fetch_disclosure_dates(list(C.FIRMS), C.ANALYSIS_START_YEAR, C.ANALYSIS_END_YEAR,
                                         os.environ["DART_API_KEY"])
        if not dd.empty:
            save(dd, "disclosures")
    else:
        print("  건너뜀: .env 에 DART_API_KEY 가 없습니다.")
print("참고: 도시가스 판매량은 API 대신 직접 내려받아 data/raw/citygas_volume.csv 로 둡니다 (README 7-1).")
