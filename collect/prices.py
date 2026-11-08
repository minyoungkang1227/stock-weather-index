"""주가 수집 (FinanceDataReader). 종목코드 + 기간만 받아 (date, code, close, volume)을 반환합니다."""
from __future__ import annotations

import pandas as pd


def listing_names() -> dict[str, str]:
    import FinanceDataReader as fdr
    lst = fdr.StockListing("KRX")
    code_col = "Code" if "Code" in lst.columns else "Symbol"
    return dict(zip(lst[code_col].astype(str).str.zfill(6), lst["Name"]))


def fetch_prices(codes: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    import FinanceDataReader as fdr
    frames, failed = [], []
    for code in codes:
        try:
            df = fdr.DataReader(code, start, end)
            if df.empty:
                raise ValueError("빈 데이터")
            vol = df["Volume"].values if "Volume" in df else float("nan")
            frames.append(pd.DataFrame({"date": df.index, "code": code, "close": df["Close"].values, "volume": vol}))
        except Exception as e:
            failed.append((code, str(e)[:100]))
    for f in failed:
        print(f"[prices] 실패 {f[0]}: {f[1]}")
    if not frames:
        return pd.DataFrame(columns=["date", "code", "close", "volume"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out
