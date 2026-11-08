@echo off
REM 매월 초 실행: 실적·공시일 갱신 → 백테스트·모델 재학습
cd /d "%~dp0.."
python swi_system.py monthly >> logs_monthly.txt 2>&1
