"""분석 대상·기간·관측소 설정. 값을 바꿔도 코드 로직은 그대로 둡니다."""

# 분석 기간 (실적·주가). 평년(30년) 계산용 기상 자료는 WEATHER_START부터 받습니다.
ANALYSIS_START_YEAR = 2015
ANALYSIS_END_YEAR = 2025
WEATHER_START = "1985-01-01"

BASE_TEMP = 18.0        # 냉·난방도일 기준온도(℃)
HEATWAVE_TMAX = 33.0    # 폭염일 기준 일최고기온(℃)
NORMAL_YEARS = 30       # 평년 산출 연수 (해당 연도 '이전' 30년만 사용)
MIN_NORMAL_YEARS = 20   # 평년 계산에 필요한 최소 연수. 부족하면 NaN

# 종관기상관측(ASOS) 지점번호
STATIONS = {
    108: "서울", 112: "인천", 119: "수원", 133: "대전",
    143: "대구", 152: "울산", 156: "광주", 159: "부산",
}
NATIONAL = [108, 112, 133, 143, 156, 159]  # 전국형 기업에 쓰는 6개 도시 평균

# 대상 종목: 코드 -> (기대 종목명, 업종그룹, 기상 매핑, 주 날씨지표)
# 기대 종목명은 수집 시 실제 상장명과 대조해 다르면 그 종목을 멈추고 알립니다.
# 기상 매핑: 정수 = 해당 지점, "NATIONAL" = 6개 도시 평균
FIRMS = {
    # 도시가스·지역난방 (난방도일)
    "017390": ("서울가스", "gas", 108, "HDD"),
    "004690": ("삼천리", "gas", 119, "HDD"),
    "034590": ("인천도시가스", "gas", 112, "HDD"),
    "117580": ("대성에너지", "gas", 143, "HDD"),
    "267290": ("경동도시가스", "gas", 152, "HDD"),
    "071320": ("지역난방공사", "gas", 108, "HDD"),
    # 음료·빙과 (냉방도일·폭염일수)
    "005180": ("빙그레", "beverage", "NATIONAL", "CDD"),
    "280360": ("롯데웰푸드", "beverage", "NATIONAL", "CDD"),
    "005300": ("롯데칠성", "beverage", "NATIONAL", "CDD"),
    "000080": ("하이트진로", "beverage", "NATIONAL", "CDD"),
    # 의류·유통 (평년 대비 기온 편차)
    "020000": ("한섬", "apparel", "NATIONAL", "TAVG"),
    "093050": ("LF", "apparel", "NATIONAL", "TAVG"),
    "031430": ("신세계인터내셔날", "apparel", "NATIONAL", "TAVG"),
    "081660": ("휠라홀딩스", "apparel", "NATIONAL", "TAVG"),
    "139480": ("이마트", "retail", "NATIONAL", "TAVG"),
    "023530": ("롯데쇼핑", "retail", "NATIONAL", "TAVG"),
    "069960": ("현대백화점", "retail", "NATIONAL", "TAVG"),
    "004170": ("신세계", "retail", "NATIONAL", "TAVG"),
}

MARKET_INDEX = "KS11"  # KOSPI

RAW = "data/raw"
PROCESSED = "data/processed"
OUTPUT = "output"

# ---------- 스트레스 테스트 ----------
STRESS_YEAR = 2027          # 시뮬레이션할 연도
STRESS_PATHS = 5000         # 몬테카를로 경로 수
STRESS_SEED = 42
# 업종별로 쓸 민감도 하나 (같은 업종에 지표 두 개를 동시에 쓰면 효과를 이중으로 셈)
STRESS_SPEC = {"gas": "HDD", "beverage": "CDD", "apparel": "TAVG", "retail": "TAVG"}
BETA_UNCERTAINTY = True     # 회귀계수의 추정오차(표준오차)도 경로마다 반영
CLIMATE_SHIFTS = [1.0]      # 계절 평균 기온을 +1℃ 옮긴 기후 시나리오
VALIDATION_SPLIT = 2010     # 생성기 검증: 이 연도까지로 추정하고 이후 연도를 맞히는지 확인

# ---------- Stock Weather Index ----------
COLD_TMIN = -12.0           # 한파일: 일최저기온 -12℃ 이하 (기상청 한파경보 기준 기온)
RAIN_DAY_MM = 1.0           # 강수일 기준
DISCLOSURE_LAG_DAYS = 90    # 분기말 후 이 기간이 지나야 그 분기 실적을 '알고 있다'고 봄 (사업보고서 기한 반영)
MIN_FIRM_OBS = 12           # 종목 단독 민감도 추정에 필요한 최소 분기 수
SWI_MIN_TRAIN_MONTHS = 36   # 상승확률 모델 최소 학습 기간
SWI_REFIT_EVERY = 3         # 모델 재학습 주기(개월)
SWI_SELECT_MONTHS = 36      # 표본외 앞부분 이 기간으로 모델 선택, 나머지로 최종 평가
SWI_WEIGHTS = {"benefit": 0.5, "prob": 0.5}
SWI_WATCH = (65, 50)        # 관심종목: SWI >= 65 이고 신뢰도 >= 50
SWI_WARN = (30, 60)         # 위험경고: 수혜도 <= 30 이고 신뢰도 >= 60
COST_BPS_ROUNDTRIP = 30     # 포트폴리오 백테스트 왕복 거래비용(bp)

# ---------- 개선 (2026-11-08) ----------
# 1) 도시가스: 매출 대신 지역 도시가스 판매량으로 민감도 추정 (data/raw/citygas_volume.csv 가 있을 때)
#    판매량 자료가 없거나 지역이 없는 종목은 기존처럼 매출을 씁니다.
GAS_REGION = {"017390": "서울", "004690": "경기", "034590": "인천", "117580": "대구", "267290": "울산"}
SWI_GAS_USE_VOLUME = True
# 2) 이벤트 스터디: 공시일 자료가 없으면 분기말 + 이 일수를 공시일로 가정
EVENT_FALLBACK_DAYS = 45
EVENT_WINDOWS = {"분기중": None, "공시 전후(-1~+1일)": (-1, 1), "공시 후(+2~+40일)": (2, 40)}
# 3) 순위·점수: 업종 안에서 비교 (False면 이전처럼 전 종목 비교)
SWI_WITHIN_GROUP = True

# ---------- 시스템 (SQLite + 예약 실행 + 대시보드) ----------
DB_PATH = "data/swi.db"          # 이 파일이 있으면 분석 코드가 CSV 대신 DB에서 읽습니다
MODEL_DIR = "models"
# 공공데이터포털 '기상청_중기예보 조회서비스' 중기기온 예보구역 코드 (기온 예보 3~10일 뒤)
# 코드가 맞지 않으면 수집 시 오류로 표시됩니다. 기상청 예보구역 코드표로 확인하세요.
FORECAST_REG = {108: "11B10101", 112: "11B20201", 119: "11B20601", 133: "11C20401",
                143: "11H10701", 152: "11H20101", 156: "11F20501", 159: "11H20201"}
FORECAST_ERR_SD = 2.0            # 중기예보 일평균기온 오차 표준편차(℃) 가정 — 예보가 쌓이면 실제 오차로 교체
CLIM_SMOOTH_DAYS = 15            # 일별 평년값을 앞뒤로 평균내는 창(일)
GATE_WEATHER_STALE_DAYS = 5      # 관측이 이보다 오래되면 지수 산출 중단
GATE_PRICE_STALE_DAYS = 7

# ---------- 기상청 API허브 (KMA_APIHUB_KEY 가 있으면 공공데이터포털 대신 사용) ----------
# 중기 기온예보 엔드포인트: API허브 '예특보 > 중기예보 > 중기예보자료(2001년 2월 이후) 조회'의 기온 항목 URL 파일명
KMA_FCST_TEMP_ENDPOINT = "fct_afs_wc.php"   # 확인 필요: 실제 URL의 파일명과 다르면 여기만 고치세요
FORECAST_HIST_START = "2015-01-01"          # 예보 이력 수집 시작 (분석 기간과 맞춤)
KMA_CACHE = "data/raw/kma_cache"            # 연·월별 이어받기용 임시 저장
