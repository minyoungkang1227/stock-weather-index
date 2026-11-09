"""업종 넓히기 (config.PREREG_SECTOR_VOLUME): 가격이 섞이지 않은 '물량' 자료로 날씨 영향을 검정.

자료 읽기는 원본 형식별로 따로 두고(수집과 분석 분리), 검정은 모두 같은 틀:
  물량의 전년 대비 로그성장률 ~ 전국(6개 도시 평균) Δ날씨 + 분기더미, 연도 순열검정 단측.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _read(path: str, **kw) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 읽을 수 없습니다: {path}")


def read_epsis_monthly(path: str) -> pd.DataFrame:
    """EPSIS 월별 판매전력량(가로형: 연도, 1월..12월, MWh) → item='합계', ym, volume."""
    raw = _read(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    ycol = raw.columns[0]
    months = [c for c in raw.columns[1:] if str(c).replace("월", "").isdigit()]
    if len(months) != 12:
        raise ValueError(f"1~12월 열을 찾지 못했습니다: {list(raw.columns)}")
    long = raw.melt(id_vars=ycol, value_vars=months, var_name="m", value_name="volume")
    long["year"] = pd.to_numeric(long[ycol], errors="coerce")
    long = long.dropna(subset=["year"])
    long["m"] = long["m"].str.replace("월", "").astype(int)
    long["ym"] = long["year"].astype(int).astype(str) + "-" + long["m"].map("{:02d}".format)
    long["volume"] = pd.to_numeric(long["volume"], errors="coerce")
    return long.assign(item="합계")[["item", "ym", "volume"]].sort_values("ym").reset_index(drop=True)


def read_epsis_annual_by_use(path: str) -> pd.DataFrame:
    """EPSIS 용도별 판매전력량(연, 머리글 3줄: 대분류/중분류/점유율) → item, year, volume.
    '점유율' 열은 쓰지 않음 (2024~2025년 점유율 값이 실제 비율과 맞지 않음 — 확인됨)."""
    raw = _read(path, header=None)
    h0, h1 = raw.iloc[0].ffill(), raw.iloc[1]
    body = raw.iloc[3:].copy()
    out = []
    for j in range(1, raw.shape[1]):
        name = str(h1.iloc[j]) if pd.notna(h1.iloc[j]) else str(h0.iloc[j])
        if name in ("점유율", "nan") or (pd.notna(raw.iloc[2, j]) and str(raw.iloc[2, j]) == "점유율"):
            continue
        out.append(pd.DataFrame({"item": name, "year": pd.to_numeric(body[0], errors="coerce"),
                                 "volume": pd.to_numeric(body[j], errors="coerce")}))
    d = pd.concat(out).dropna(subset=["year"])
    d["year"] = d["year"].astype(int)
    return d.drop_duplicates(["item", "year"])


def quarterly_growth(v: pd.DataFrame) -> pd.DataFrame:
    """item, ym, volume (월) → item, year, q, vol_g (분기 합의 전년동기 로그성장률, 3개월 다 있을 때만)."""
    d = v.dropna(subset=["volume"]).copy()
    d["year"] = d["ym"].str[:4].astype(int)
    d["q"] = (d["ym"].str[5:7].astype(int) - 1) // 3 + 1
    g = d.groupby(["item", "year", "q"])["volume"].agg(["sum", "count"]).reset_index()
    g = g[g["count"] == 3].rename(columns={"sum": "vol"})
    ly = g[["item", "year", "q", "vol"]].assign(year=g["year"] + 1).rename(columns={"vol": "vol_ly"})
    g = g.merge(ly, on=["item", "year", "q"], how="left")
    g["vol_g"] = np.where((g["vol"] > 0) & (g["vol_ly"] > 0), np.log(g["vol"] / g["vol_ly"]), np.nan)
    return g[["item", "year", "q", "vol_g"]]


def annual_growth(v: pd.DataFrame) -> pd.DataFrame:
    d = v.sort_values(["item", "year"]).copy()
    d["vol_ly"] = d.groupby("item")["volume"].shift(1)
    ok = (d["year"].diff() == 1) & (d["vol_ly"] > 0) & (d["volume"] > 0)
    d["vol_g"] = np.where(ok, np.log(d["volume"] / d["vol_ly"]), np.nan)
    return d.assign(q=1)[["item", "year", "q", "vol_g"]]


def national_weather(wq: pd.DataFrame, annual: bool = False) -> pd.DataFrame:
    """전국(6개 도시 평균) 분기 HDD·CDD 와 전년 대비 변화(100도일 단위). annual=True 면 연 합계 기준."""
    n = wq[wq["stn"].astype(str) == "NATIONAL"][["year", "q", "HDD", "CDD"]].copy()
    if annual:
        n = n.groupby("year")[["HDD", "CDD"]].sum(min_count=4).reset_index().assign(q=1)
    n = n.sort_values(["q", "year"])
    for c in ("HDD", "CDD"):
        prev = n.groupby("q")[c].shift(1)
        same = n.groupby("q")["year"].diff() == 1
        n[f"{c}_yoy_100"] = np.where(same, (n[c] - prev) / 100, np.nan)
    return n[["year", "q", "HDD_yoy_100", "CDD_yoy_100"]]


def placebo_check(g: pd.DataFrame, q: int | None, tol: float, y0: int, y1: int) -> dict:
    """날씨 영향이 거의 없는 대조 분기(q)의 전년비가 tol 을 넘는 해 = 날씨 밖 충격 의심. q=None(연간)이면 전체 연도."""
    sub = g[g["year"].between(y0, y1)]
    if q is not None:
        sub = sub[sub["q"] == q]
    bad = sub[sub["vol_g"].abs() > tol]
    return {int(r.year): float(r.vol_g) for r in bad.itertuples()}


def read_kosis_industry_index(path: str, item_word: str = "출하지수(원지수)", region: str = "전국") -> pd.DataFrame:
    """KOSIS 시도/산업별 광공업생산지수 CSV(가로형: 시도별, 산업별, 항목, 단위, 'YYYY.MM 월'...) → item(산업), ym, volume.
    원지수(계절조정 아님) 출하지수만 사용 — 계절조정값을 쓰면 전년동기 비교와 이중 처리가 된다."""
    raw = _read(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    sub = raw[(raw["시도별"] == region) & (raw["항목"].astype(str).str.contains(item_word, regex=False))]
    if sub.empty:
        raise ValueError(f"'{region}' / '{item_word}' 행이 없습니다. 항목: {raw['항목'].unique()[:8]}")
    tcols = [c for c in raw.columns if c[:4].isdigit() and "." in c[:7]]
    long = sub.melt(id_vars="산업별", value_vars=tcols, var_name="t", value_name="volume")
    long["ym"] = long["t"].str[:4] + "-" + long["t"].str[5:7]
    long["volume"] = pd.to_numeric(long["volume"], errors="coerce")
    return long.rename(columns={"산업별": "item"})[["item", "ym", "volume"]]


def match_item(names, keys: list[str]):
    """자료 안 이름 중 후보 단어와 맞는 것. 공백 무시, '그 단어로 시작'하는 이름을 우선
    (예: '알코올음료'는 '비알코올 음료…'에도 들어 있으므로 시작 일치를 먼저 본다)."""
    norm = {n: str(n).replace(" ", "") for n in names}
    for k in keys:
        kk = k.replace(" ", "")
        hit = [n for n, v in norm.items() if v.startswith(kk)]
        if hit:
            return hit[0]
    for k in keys:
        kk = k.replace(" ", "")
        hit = [n for n, v in norm.items() if kk in v]
        if hit:
            return hit[0]
    return None


def run_family(spec: dict, wq: pd.DataFrame, readers: dict, perm, B: int, seed: int) -> tuple[pd.DataFrame, list]:
    """사전 등록 묶음 실행. 반환: (검정표, 자료 점검에서 걸린 목록). 걸린 것이 있으면 검정하지 않음(호출 쪽에서 멈춤)."""
    from statsmodels.stats.multitest import multipletests
    y1 = spec["end_year"]
    w_q, w_a = national_weather(wq), national_weather(wq, annual=True)
    rows, flags, cache = [], [], {}
    for t in spec["tests"]:
        src = t["src"]
        if src not in cache:
            cache[src] = readers[src]() if src in readers else None
        v = cache[src]
        name_match = None if v is None else match_item(v["item"].unique(), t["item"])
        if v is None or name_match is None:
            rows.append(dict(name=t["name"], item=None, n=0, coef=np.nan, p_one=np.nan, note="자료 없음"))
            continue
        vi = v[v["item"] == name_match]
        annual = src == "power_annual"
        g = annual_growth(vi) if annual else quarterly_growth(vi)
        g = g[~g["year"].isin(t.get("exclude_years", []))]     # 자료 점검 뒤 사람이 정한 제외 연도 (config 에 이유 기록)
        tol = spec["annual_tol"] if annual else spec["placebo_tol"][src]
        bad = placebo_check(g, None if annual else t["placebo_q"], tol, t["start"], y1)
        if bad:
            flags.append((t["name"], name_match, bad))
        d = g.merge(w_a if annual else w_q, on=["year", "q"], how="left").assign(code=name_match)
        d = d[(d["year"] >= t["start"]) & (d["year"] <= y1)]
        rows.append(dict(name=t["name"], item=name_match, n=int(d.dropna(subset=["vol_g"] + t["xs"]).shape[0]),
                         _d=d, _t=t, note=""))
    if flags:
        return pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]), flags
    for r in rows:
        if "_d" in r:
            t = r.pop("_t"); d = r.pop("_d")
            res = perm(d, "vol_g", t["xs"], t["x"], [], t["sign"], False, B, seed)
            r.update(coef=res["coef"], p_one=res["p_perm"])
    out = pd.DataFrame(rows)
    ok = out["p_one"].notna()
    out["q_bh"] = np.nan
    if ok.any():
        out.loc[ok, "q_bh"] = multipletests(out.loc[ok, "p_one"], method="fdr_bh")[1]
    signs = {t["name"]: t["sign"] for t in spec["tests"]}
    out["pass"] = ok & (np.sign(out["coef"]) == out["name"].map(signs)) & (out["q_bh"] <= spec["bh_q"])
    return out, []
