"""SQLite 저장소.

왜 SQLite인가: 파일 하나라 설치가 필요 없고, 매일 쌓이는 예보·지수 이력을 키 기준으로 중복 없이 갱신(upsert)할 수 있습니다.
나중에 서버 DB로 옮길 때는 이 파일의 함수만 바꾸면 됩니다.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing

import pandas as pd

# 테이블별 기본키: 같은 키의 행이 다시 들어오면 새 값으로 교체
KEYS = {
    "weather_daily": ["stn", "date"],
    "fin_quarterly": ["code", "year", "q"],
    "fin_cumulative_raw": ["code", "year", "q", "fs_div"],
    "prices": ["code", "date"],
    "market": ["date"],
    "disclosures": ["code", "year", "q"],
    "citygas_volume": ["region", "ym"],
    "cpi_citygas": ["month"],
    "forecast_mid": ["issued_at", "stn", "target_date"],
    "swi_daily": ["asof", "code"],
    "swi_monthly": ["ym", "code"],
    "nowcast_daily": ["asof", "stn"],
    "alerts": ["asof", "code", "kind"],
    "runs": ["run_id"],
    "swi_model_eval": ["run_id", "model", "features", "window"],
    "swi_backtest": ["run_id", "score", "window", "ranking"],
    "swi_backtest_by_group": ["run_id", "score", "group"],
    "swi_event_study": ["run_id", "window"],
}


def connect(path: str) -> sqlite3.Connection:
    return sqlite3.connect(path)


def has_table(con: sqlite3.Connection, name: str) -> bool:
    q = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"
    return con.execute(q, (name,)).fetchone() is not None


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime("%Y-%m-%d")
        elif isinstance(d[c].dtype, pd.PeriodDtype):
            d[c] = d[c].astype(str)
        elif d[c].dtype == object or pd.api.types.is_string_dtype(d[c]):
            d[c] = d[c].astype(object).where(d[c].notna(), None)
            d[c] = d[c].map(lambda v: v if v is None or isinstance(v, (str, int, float)) else str(v))
    return d


def upsert(path: str, table: str, df: pd.DataFrame) -> int:
    """키가 같은 기존 행은 지우고 새 행을 넣습니다. 새 열이 생기면 테이블에 추가합니다. 넣은 행 수 반환."""
    if df is None or df.empty:
        return 0
    keys = KEYS[table]
    d = _prep(df).drop_duplicates(keys, keep="last")
    with closing(connect(path)) as con:
        if not has_table(con, table):
            d.to_sql(table, con, index=False)
            keylist = ",".join(f'"{k}"' for k in keys)  # group 같은 예약어 열 이름 대비
            con.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS ux_{table} ON "{table}" ({keylist})')
            con.commit()
            return len(d)
        cols = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]
        for c in d.columns:
            if c not in cols:
                con.execute(f'ALTER TABLE "{table}" ADD COLUMN "{c}"')
        d.to_sql("_tmp", con, index=False, if_exists="replace")
        cond = " AND ".join(f'"{table}"."{k}" = _tmp."{k}"' for k in keys)
        con.execute(f'DELETE FROM "{table}" WHERE EXISTS (SELECT 1 FROM _tmp WHERE {cond})')
        collist = ",".join(f'"{c}"' for c in d.columns)
        con.execute(f'INSERT INTO "{table}" ({collist}) SELECT {collist} FROM _tmp')
        con.execute("DROP TABLE _tmp")
        con.commit()
    return len(d)


def read(path: str, table: str, where: str = "", params: tuple = ()) -> pd.DataFrame | None:
    with closing(connect(path)) as con:
        if not has_table(con, table):
            return None
        return pd.read_sql(f'SELECT * FROM "{table}" {where}', con, params=params)


def ingest_raw(path: str, raw_dir: str) -> dict[str, int]:
    """data/raw/*.csv 를 같은 이름의 테이블로 적재 (수집 결과 → DB)."""
    import os
    out = {}
    for t in ("weather_daily", "fin_quarterly", "fin_cumulative_raw", "prices", "market", "disclosures",
              "citygas_volume", "cpi_citygas", "forecast_mid"):
        f = os.path.join(raw_dir, f"{t}.csv")
        if os.path.exists(f):
            df = pd.read_csv(f, dtype={"code": str})
            if "code" in df:
                df["code"] = df["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
            out[t] = upsert(path, t, df)
    return out
