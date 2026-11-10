"""공개 웹용 결과 스냅샷 (python swi_system.py publish).

왜 따로 만드나: 공개 저장소·웹에는 원자료(주가·판매량·DART)와 DB를 올리지 않는다.
대신 '계산 결과'만 작은 CSV 로 뽑아 public_snapshot/ 에 둔다. DB 가 없는 곳(Streamlit Cloud)에서는
대시보드가 이 폴더를 읽어 날씨 짐작·검증·데이터 기준일만 보여 준다.

넣는 것: 최신 기준일의 지역별 분기 날씨 짐작(HDD·CDD, 기상청 공개자료에서 계산), 사전 등록 판정 요약표,
        데이터 기준일(지점별 마지막 관측일, 최신 예보 발표 시각, 마지막 daily 실행 상태)
넣지 않는 것: 원자료 전부, 주가와 주가로 만든 종목 점수, 실행 로그 메시지, 인증키
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

import pandas as pd

from system import db

SNAP_DIR = "public_snapshot"

# 대시보드 '검증' 탭이 읽는 요약표 (output/ 에서 그대로 복사, 모두 통계 요약값만 담김)
RESULT_FILES = ["prereg_stage1", "prereg_gas_volume", "prereg_gas_opinc", "prereg_sector_volume",
                "prereg_nowcast_oos_verdict", "prereg_forecast_value", "prereg_stage2",
                "forecast_value_by_days_left"]

NOWCAST_COLS = ["asof", "year", "quarter", "obs_share", "n_obs", "n_fc", "n_clim", "stn"] + [
    f"{v}_{k}" for v in ("HDD", "CDD") for k in ("exp", "sd", "ly", "norm")]


class SnapshotError(RuntimeError):
    pass


def export(db_path: str, output_dir: str, snap_dir: str = SNAP_DIR) -> dict:
    """DB·output 에서 공개용 파일을 만든다. 필요한 자료가 없으면 만들지 않고 멈춘다(빈 스냅샷을 공개하지 않음)."""
    if not os.path.exists(db_path):
        raise SnapshotError(f"DB 없음: {db_path}")
    nc = db.read(db_path, "nowcast_daily")
    if nc is None or nc.empty:
        raise SnapshotError("nowcast_daily 가 비어 있습니다. 먼저 daily 를 실행하세요.")
    asof = str(nc["asof"].max())
    cur = nc[nc["asof"].astype(str) == asof]
    cur = cur.drop_duplicates(subset=["stn"], keep="last")[[c for c in NOWCAST_COLS if c in cur.columns]]
    missing = [f for f in RESULT_FILES if not os.path.exists(os.path.join(output_dir, f"{f}.csv"))]
    if missing:
        raise SnapshotError(f"output 에 판정표가 없습니다: {missing}. 먼저 load 또는 monthly 를 실행하세요.")

    meta = dict(created=datetime.now().isoformat(timespec="seconds"), asof=asof, last_obs={}, forecast_latest=None,
                last_daily=None)
    w = db.read(db_path, "weather_daily")
    if w is not None and len(w):
        last = w.dropna(subset=["tavg"]).groupby(w["stn"].astype(str))["date"].max()
        meta["last_obs"] = {k: str(v) for k, v in last.items()}
    f = db.read(db_path, "forecast_mid")
    if f is not None and len(f):
        meta["forecast_latest"] = str(f["issued_at"].astype(str).max())
    r = db.read(db_path, "runs")
    if r is not None and len(r):
        d = r[r["job"] == "daily"].sort_values("finished")
        if len(d):
            meta["last_daily"] = dict(finished=str(d.iloc[-1]["finished"]), status=str(d.iloc[-1]["status"]),
                                      asof=str(d.iloc[-1]["asof"]))
            if meta["last_daily"]["status"] != "ok":
                # 마지막 daily 가 품질 점검 등으로 멈췄으면 공개하지 않는다(멈춘 이유를 먼저 확인)
                raise SnapshotError(f"마지막 daily 실행이 '{meta['last_daily']['status']}' 입니다. 원인을 확인한 뒤 다시 실행하세요.")
    # 점검을 모두 통과한 뒤에만 쓴다
    os.makedirs(snap_dir, exist_ok=True)
    cur.to_csv(os.path.join(snap_dir, "nowcast_latest.csv"), index=False, encoding="utf-8-sig")
    for name in RESULT_FILES:
        shutil.copyfile(os.path.join(output_dir, f"{name}.csv"), os.path.join(snap_dir, f"{name}.csv"))
    with open(os.path.join(snap_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    return meta


def load_meta(snap_dir: str = SNAP_DIR) -> dict | None:
    p = os.path.join(snap_dir, "meta.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)
