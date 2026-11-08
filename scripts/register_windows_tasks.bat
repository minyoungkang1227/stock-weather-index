@echo off
REM 윈도우 작업 스케줄러에 등록 (관리자 권한 불필요, 로그인 상태에서 실행)
REM 매일 07:30 (중기예보 06시 발표 이후), 매월 2일 08:00
schtasks /Create /F /SC DAILY /TN "SWI_daily" /TR "\"%~dp0run_daily.bat\"" /ST 07:30
schtasks /Create /F /SC MONTHLY /D 2 /TN "SWI_monthly" /TR "\"%~dp0run_monthly.bat\"" /ST 08:00
echo 등록 완료. 확인: schtasks /Query /TN SWI_daily
