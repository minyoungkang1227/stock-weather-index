@echo off
REM 매일 아침 실행: 관측·주가·예보 갱신 → 오늘의 지수
cd /d "%~dp0.."
python swi_system.py daily >> logs_daily.txt 2>&1
