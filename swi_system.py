"""Stock Weather Index 운영 시스템.

  python swi_system.py init        DB·폴더 만들기
  python swi_system.py backfill    전체 과거자료 수집(run_collect.py) → DB 적재 → 월간 작업 1회
  python swi_system.py daily       [매일] 관측·주가·예보 갱신 → 품질 점검 → 오늘의 지수 → DB 저장
  python swi_system.py monthly     [매월 초] 실적·공시일 갱신 → 백테스트·모델 재학습 → 검증 결과 DB 저장
  python swi_system.py dashboard   대시보드 열기 (streamlit)
  옵션: --asof YYYY-MM-DD (daily를 특정 날짜 기준으로)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
import uuid
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

import config as C
from system import db

load_dotenv()


def log_run(job: str, status: str, message: str, run_id: str, started: str, asof: str = "") -> None:
    db.upsert(C.DB_PATH, "runs", pd.DataFrame([dict(run_id=run_id, job=job, status=status, message=message[:2000],
                                                    started=started, finished=datetime.now().isoformat(timespec="seconds"),
                                                    asof=asof)]))
    print(f"[{job}] {status}: {message[:300]}")


def cmd_init() -> None:
    for d in (C.RAW, C.PROCESSED, C.OUTPUT, C.MODEL_DIR, os.path.dirname(C.DB_PATH) or "."):
        os.makedirs(d, exist_ok=True)
    rid = uuid.uuid4().hex[:8]
    log_run("init", "ok", "DB 생성", rid, datetime.now().isoformat(timespec="seconds"))


def cmd_backfill() -> None:
    rid, t0 = uuid.uuid4().hex[:8], datetime.now().isoformat(timespec="seconds")
    try:
        subprocess.run([sys.executable, "run_collect.py"], check=True)
        n = db.ingest_raw(C.DB_PATH, C.RAW)
        log_run("backfill", "ok", f"적재 {n}", rid, t0)
    except Exception as e:
        log_run("backfill", "error", f"{e}\n{traceback.format_exc()}", rid, t0)
        return
    cmd_monthly(collect=False)


def cmd_daily(asof: pd.Timestamp) -> None:
    from collect import forecast, prices, weather
    from system import today
    rid, t0 = uuid.uuid4().hex[:8], datetime.now().isoformat(timespec="seconds")
    msgs = []
    try:
        hub, key = os.getenv("KMA_APIHUB_KEY"), os.getenv("DATA_GO_KR_KEY")
        if hub:
            from collect import kma_apihub
            start = (asof - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
            w = kma_apihub.fetch_asos_daily(list(C.STATIONS), start, asof.strftime("%Y-%m-%d"), hub)
            msgs.append(f"기상 관측(API허브) {db.upsert(C.DB_PATH, 'weather_daily', w)}행")
            f = kma_apihub.fetch_mid_temperature_latest(C.FORECAST_REG, hub, C.KMA_FCST_TEMP_ENDPOINT)
            msgs.append(f"예보(API허브) {db.upsert(C.DB_PATH, 'forecast_mid', f)}행")
        elif key:
            w = weather.fetch_asos(list(C.STATIONS), asof.year - (1 if asof.month == 1 else 0), asof.year, key)
            msgs.append(f"기상 관측 {db.upsert(C.DB_PATH, 'weather_daily', w)}행")
            f = forecast.fetch_mid_temperature(C.FORECAST_REG, key)
            msgs.append(f"예보 {db.upsert(C.DB_PATH, 'forecast_mid', f)}행")
        else:
            msgs.append("KMA_APIHUB_KEY·DATA_GO_KR_KEY 없음: 관측·예보 갱신 생략")
        try:
            start = (asof - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
            px = prices.fetch_prices(list(C.FIRMS), start)
            mk = prices.fetch_prices([C.MARKET_INDEX], start).drop(columns="code")
            msgs.append(f"주가 {db.upsert(C.DB_PATH, 'prices', px)}행, 지수 {db.upsert(C.DB_PATH, 'market', mk)}행")
        except Exception as e:  # 주가 수집 실패는 품질 점검(신선도)에서 걸러짐
            msgs.append(f"주가 갱신 실패: {str(e)[:100]}")

        import run_swi
        prep = run_swi.prepare()
        fc = db.read(C.DB_PATH, "forecast_mid", "WHERE substr(issued_at,1,8) <= ?", (asof.strftime("%Y%m%d"),))
        scores, nc = today.score(asof, prep, fc)
        db.upsert(C.DB_PATH, "swi_daily", scores.assign(run_id=rid))
        db.upsert(C.DB_PATH, "nowcast_daily", nc.assign(run_id=rid))
        al = pd.concat([scores[scores["관심종목"]].assign(kind="관심종목"), scores[scores["위험경고"]].assign(kind="위험경고")])
        if len(al):
            db.upsert(C.DB_PATH, "alerts", al[["asof", "code", "name", "group", "kind", "SWI", "B", "P", "C"]])
        msgs.append(f"지수 {len(scores)}종목, 관심 {int(scores['관심종목'].sum())}, 경고 {int(scores['위험경고'].sum())}")
        log_run("daily", "ok", " / ".join(msgs), rid, t0, asof.strftime("%Y-%m-%d"))
    except SystemExit as e:
        log_run("daily", "halted", " / ".join(msgs + [f"자료 부족으로 중단: {e}"]), rid, t0, asof.strftime("%Y-%m-%d"))
    except today.GateError as e:
        log_run("daily", "halted", " / ".join(msgs + [f"품질 점검에서 중단: {e}"]), rid, t0, asof.strftime("%Y-%m-%d"))
    except Exception as e:
        log_run("daily", "error", " / ".join(msgs + [str(e), traceback.format_exc()]), rid, t0, asof.strftime("%Y-%m-%d"))


def cmd_monthly(collect: bool = True) -> None:
    rid, t0 = uuid.uuid4().hex[:8], datetime.now().isoformat(timespec="seconds")
    try:
        if collect:
            subprocess.run([sys.executable, "run_collect.py", "dart", "disclosure", "ecos"], check=False)
            db.ingest_raw(C.DB_PATH, C.RAW)
        import run_analysis
        import run_swi
        try:
            run_analysis.main()
        except SystemExit:
            pass  # 분석 리포트는 참고용, 실패해도 지수 백테스트는 진행
        run_swi.main()
        out = C.OUTPUT
        m = pd.read_csv(f"{out}/swi_monthly.csv", dtype={"code": str})
        db.upsert(C.DB_PATH, "swi_monthly", m.assign(run_id=rid))
        for t in ("swi_model_eval", "swi_backtest", "swi_backtest_by_group", "swi_event_study"):
            f = f"{out}/{t}.csv"
            if os.path.exists(f):
                db.upsert(C.DB_PATH, t, pd.read_csv(f).assign(run_id=rid))
        log_run("monthly", "ok", f"월간 {len(m)}행, 모델 저장", rid, t0)
    except SystemExit as e:
        log_run("monthly", "halted", f"중단: {e}", rid, t0)
    except Exception as e:
        log_run("monthly", "error", f"{e}\n{traceback.format_exc()}", rid, t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["init", "backfill", "daily", "monthly", "dashboard"])
    ap.add_argument("--asof", default=None)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(C.DB_PATH) or ".", exist_ok=True)
    if a.job == "init":
        cmd_init()
    elif a.job == "backfill":
        cmd_backfill()
    elif a.job == "daily":
        asof = pd.Timestamp(a.asof) if a.asof else pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=1)
        cmd_daily(asof)
    elif a.job == "monthly":
        cmd_monthly()
    else:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])


if __name__ == "__main__":
    main()
