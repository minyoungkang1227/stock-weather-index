"""OpenDART 분기 실적 수집.

입력은 '종목코드 + 연도 범위'만 받고, 출력은 표준 형식
(code, year, q, revenue, op_income, fs_div) 입니다. 단위는 원.

분기 손익은 '누적 금액의 차분'으로 만듭니다.
  Q1 = 1분기 누적, Q2 = 반기 누적 - Q1 누적, Q3 = 3분기 누적 - 반기 누적, Q4 = 연간 - 3분기 누적
사업보고서(연간)에는 4분기 단독 금액이 없기 때문에 이 방식이 필요합니다.
"""
from __future__ import annotations

import io
import time
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd
import requests

BASE = "https://opendart.fss.or.kr/api"
REPRT = {1: "11013", 2: "11012", 3: "11014", 4: "11011"}  # 1분기, 반기, 3분기, 사업보고서
REVENUE_NAMES = {"매출액", "수익(매출액)", "영업수익", "매출"}
OPINC_NAMES = {"영업이익", "영업이익(손실)"}


def corp_code_map(key: str) -> dict[str, str]:
    """종목코드(6자리) -> DART 고유번호(8자리)."""
    r = requests.get(f"{BASE}/corpCode.xml", params={"crtfc_key": key}, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        root = ET.fromstring(z.read(z.namelist()[0]))
    out = {}
    for el in root.iter("list"):
        stock = (el.findtext("stock_code") or "").strip()
        if stock:
            out[stock] = el.findtext("corp_code").strip()
    return out


def _to_num(s) -> float:
    if s is None:
        return float("nan")
    s = str(s).replace(",", "").strip()
    if s in ("", "-"):
        return float("nan")
    return float(s)


def _pick(items: list[dict], names: set[str], fs_div: str) -> tuple[float, float]:
    """(누적 금액, 해당기간 단독 금액). 연간·1분기는 둘이 같습니다."""
    for it in items:
        if it.get("fs_div") == fs_div and it.get("sj_div") in ("IS", "CIS") \
                and it.get("account_nm", "").strip() in names:
            cur = _to_num(it.get("thstrm_amount"))
            cum = _to_num(it.get("thstrm_add_amount")) if it.get("thstrm_add_amount") else cur
            return cum, cur
    return float("nan"), float("nan")


def fetch_cumulative(code: str, corp: str, year: int, q: int, key: str) -> dict | None:
    params = {"crtfc_key": key, "corp_code": corp, "bsns_year": str(year), "reprt_code": REPRT[q]}
    r = requests.get(f"{BASE}/fnlttSinglAcnt.json", params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    if js.get("status") != "000":
        return None  # 013 = 데이터 없음 등
    items = js["list"]
    fs = "CFS" if any(i.get("fs_div") == "CFS" for i in items) else "OFS"
    rev_cum, rev_cur = _pick(items, REVENUE_NAMES, fs)
    op_cum, op_cur = _pick(items, OPINC_NAMES, fs)
    return dict(code=code, year=year, q=q, fs_div=fs,
                rev_cum=rev_cum, rev_cur=rev_cur, op_cum=op_cum, op_cur=op_cur)


def cumulative_to_quarterly(cum: pd.DataFrame) -> pd.DataFrame:
    """누적 금액 표(code, year, q, rev_cum, op_cum, fs_div)를 분기 단독 금액으로 바꿉니다.

    앞 분기 누적이 없으면 그 분기는 NaN으로 둡니다(추정하지 않음).
    같은 연도 안에서 연결/별도 기준이 섞이면 차분이 의미 없으므로 NaN 처리합니다.
    """
    cum = cum.sort_values(["code", "year", "q"]).copy()
    out = []
    for (code, year), g in cum.groupby(["code", "year"]):
        g = g.set_index("q")
        mixed = g["fs_div"].nunique() > 1
        for q in g.index:
            row = {"code": code, "year": year, "q": q, "fs_div": g.loc[q, "fs_div"]}
            for name in ("rev", "op"):
                c = g.loc[q, f"{name}_cum"]
                if q == 1:
                    v = c
                elif (q - 1) in g.index and not mixed:
                    v = c - g.loc[q - 1, f"{name}_cum"]
                else:
                    v = float("nan")
                row["revenue" if name == "rev" else "op_income"] = v
            out.append(row)
    return pd.DataFrame(out)


def fetch_quarterly(codes: list[str], start_year: int, end_year: int, key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(분기 실적, 원자료 누적표) 반환. 한 종목이 실패해도 나머지는 계속합니다."""
    cmap = corp_code_map(key)
    rows, failed = [], []
    for code in codes:
        corp = cmap.get(code)
        if not corp:
            failed.append((code, "DART 고유번호 없음"))
            continue
        for y in range(start_year, end_year + 1):
            for q in (1, 2, 3, 4):
                try:
                    rec = fetch_cumulative(code, corp, y, q, key)
                    if rec:
                        rows.append(rec)
                    time.sleep(0.15)
                except Exception as e:
                    failed.append((code, f"{y}Q{q} {str(e)[:80]}"))
    for f in failed:
        print(f"[dart] 실패 {f[0]}: {f[1]}")
    cum = pd.DataFrame(rows)
    if cum.empty:
        return cum, cum
    return cumulative_to_quarterly(cum), cum


def fetch_disclosure_dates(codes: list[str], start_year: int, end_year: int, key: str) -> pd.DataFrame:
    """실적이 처음 시장에 알려진 날. 잠정실적 공정공시가 있으면 그 날, 없으면 정기보고서 접수일.
    반환: code, year, q, date, kind. 한 종목이 실패해도 계속합니다."""
    import re
    cmap = corp_code_map(key)
    rows, failed = [], []
    for code in codes:
        corp = cmap.get(code)
        if not corp:
            failed.append((code, "DART 고유번호 없음"))
            continue
        for y in range(start_year, end_year + 2):
            for ty in ("A", "I"):  # A=정기공시, I=거래소공시(잠정실적)
                page = 1
                while True:
                    try:
                        r = requests.get(f"{BASE}/list.json", timeout=30, params={
                            "crtfc_key": key, "corp_code": corp, "bgn_de": f"{y}0101", "end_de": f"{y}1231",
                            "pblntf_ty": ty, "page_no": page, "page_count": 100})
                        js = r.json()
                    except Exception as e:
                        failed.append((code, f"{y} {ty} {str(e)[:60]}"))
                        break
                    if js.get("status") != "000":
                        break
                    for it in js.get("list", []):
                        nm, dt = it.get("report_nm", ""), pd.Timestamp(it["rcept_dt"])
                        m = re.search(r"(분기보고서|반기보고서|사업보고서)\s*\((\d{4})\.(\d{2})\)", nm)
                        if m and "정정" not in nm:
                            yy, mm = int(m.group(2)), int(m.group(3))
                            rows.append(dict(code=code, year=yy, q=(mm - 1) // 3 + 1, date=dt, kind="정기보고서"))
                        elif "잠정" in nm and "실적" in nm:
                            qe = dt - pd.offsets.QuarterEnd(1)  # 접수일 직전 분기말
                            if (dt - qe).days <= 75:
                                rows.append(dict(code=code, year=qe.year, q=qe.quarter, date=dt, kind="잠정실적"))
                    if page >= int(js.get("total_page", 1)):
                        break
                    page += 1
                time.sleep(0.15)
    for f in failed:
        print(f"[dart-공시일] 실패 {f[0]}: {f[1]}")
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d = d[(d["year"] >= start_year) & (d["year"] <= end_year)]
    return first_disclosure(d)


def first_disclosure(d: pd.DataFrame) -> pd.DataFrame:
    """같은 분기에 여러 공시가 있으면 가장 이른 날짜를 남깁니다."""
    d = d.sort_values("date")
    return d.groupby(["code", "year", "q"], as_index=False).first()
