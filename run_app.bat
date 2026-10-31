
@echo off
title Business Intelligence Dashboard
color 0A

echo ==========================================
echo   Business Intelligence Recommendation
echo ==========================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating environment...
call venv\Scripts\activate.bat

echo.
echo Installing required packages...
python -m pip install -r requirements.txt

echo.
echo Starting Dashboard...
python -m streamlit run app.py

pause
