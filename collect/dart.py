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


def fetch_cumulative(code: str, corp: str, year: int, q: int, key: str) -> list[dict]:
    """보고서 하나에서 연결(CFS)·별도(OFS) 기준을 각각 한 행씩 돌려줍니다(있는 것만).

    기준 선택은 여기서 하지 않고 choose_basis 에서 종목 단위로 한 번에 합니다.
    (예전처럼 보고서마다 '연결 있으면 연결'로 고르면, 연중에 자회사가 생긴 해에
    1분기=별도, 2분기=연결처럼 섞여 그 해 분기 실적을 계산할 수 없었습니다.)
    """
    params = {"crtfc_key": key, "corp_code": corp, "bsns_year": str(year), "reprt_code": REPRT[q]}
    r = requests.get(f"{BASE}/fnlttSinglAcnt.json", params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    if js.get("status") != "000":
        return []  # 013 = 데이터 없음 등
    items = js["list"]
    out = []
    for fs in ("CFS", "OFS"):
        if not any(i.get("fs_div") == fs for i in items):
            continue
        rev_cum, rev_cur = _pick(items, REVENUE_NAMES, fs)
        op_cum, op_cur = _pick(items, OPINC_NAMES, fs)
        out.append(dict(code=code, year=year, q=q, fs_div=fs,
                        rev_cum=rev_cum, rev_cur=rev_cur, op_cum=op_cum, op_cur=op_cur))
    return out


def choose_basis(cum: pd.DataFrame) -> pd.DataFrame:
    """종목마다 연결/별도 중 하나만 전 기간에 씁니다: 매출이 있는 분기가 더 많은 쪽, 같으면 연결.

    전년동기 대비로 분석하므로 해마다 기준이 바뀌면 그 자체로 가짜 변화가 생깁니다.
    """
    if cum.empty or cum.groupby(["code", "year", "q"])["fs_div"].nunique().max() <= 1 and \
            cum.groupby("code")["fs_div"].nunique().max() <= 1:
        return cum
    cnt = cum.dropna(subset=["rev_cum"]).groupby(["code", "fs_div"]).size().unstack(fill_value=0)
    for fs in ("CFS", "OFS"):
        if fs not in cnt:
            cnt[fs] = 0
    pick = pd.Series(["CFS" if c >= o else "OFS" for c, o in zip(cnt["CFS"], cnt["OFS"])], index=cnt.index)
    return cum[cum["fs_div"] == cum["code"].map(pick)].copy()


def cumulative_to_quarterly(cum: pd.DataFrame, tol: float = 0.01) -> pd.DataFrame:
    """누적 금액 표를 분기 단독 금액으로 바꿉니다.

    - 1분기: 누적 = 단독.
    - 2·3분기: 그 보고서에 회사가 직접 적은 3개월 금액(rev_cur/op_cur)이 있으면 그것을 씁니다.
      없으면 누적 차분. (중간에 사업 분할·중단영업으로 앞 분기를 고쳐 쓴 경우, 차분은 틀리고
      회사가 적은 3개월 금액이 맞습니다. 예: 롯데쇼핑 2017년 3분기.)
    - 4분기: 사업보고서에는 3개월 금액이 없어 연간 - 3분기 누적. 단, 그 해에 2·3분기에서
      '차분 ≠ 회사 기재 금액'(>tol)이 보였다면 앞 분기가 고쳐 쓰인 해이므로 4분기는 NaN(추정 안 함).
    - 앞 분기 누적이 없으면 NaN. 같은 연도 안에서 연결/별도가 섞이면 차분하지 않음.
    """
    cum = cum.sort_values(["code", "year", "q"]).copy()
    out = []
    for (code, year), g in cum.groupby(["code", "year"]):
        g = g.set_index("q")
        mixed = g["fs_div"].nunique() > 1
        vals, restated = {}, False
        for q in g.index:
            for name in ("rev", "op"):
                c = g.loc[q, f"{name}_cum"]
                diff = c - g.loc[q - 1, f"{name}_cum"] if (q - 1) in g.index and not mixed else float("nan")
                cur = g.loc[q, f"{name}_cur"] if f"{name}_cur" in g else float("nan")
                if q == 1:
                    v = c
                elif q in (2, 3) and pd.notna(cur):
                    v = cur
                    if name == "rev" and pd.notna(diff) and cur != 0 and abs(diff - cur) / abs(cur) > tol:
                        restated = True
                else:
                    v = diff
                vals[(q, name)] = v
        for q in g.index:
            rev, op = vals[(q, "rev")], vals[(q, "op")]
            if q == 4 and restated:
                rev = op = float("nan")
            out.append({"code": code, "year": year, "q": q, "fs_div": g.loc[q, "fs_div"],
                        "revenue": rev, "op_income": op})
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
                    rows.extend(fetch_cumulative(code, corp, y, q, key))
                    time.sleep(0.15)
                except Exception as e:
                    failed.append((code, f"{y}Q{q} {str(e)[:80]}"))
    for f in failed:
        print(f"[dart] 실패 {f[0]}: {f[1]}")
    cum = pd.DataFrame(rows)
    if cum.empty:
        return cum, cum
    return cumulative_to_quarterly(choose_basis(cum)), cum


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
