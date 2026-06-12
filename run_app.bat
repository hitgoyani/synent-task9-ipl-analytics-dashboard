@echo off
title IPL Live Win Probability Predictor
cd /d "%~dp0"
echo =======================================================
echo    Starting IPL Live Win Probability Predictor...
echo =======================================================
..\.venv\Scripts\python.exe -m streamlit run app.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to start Streamlit app. Please ensure the virtual environment is present.
    pause
)
