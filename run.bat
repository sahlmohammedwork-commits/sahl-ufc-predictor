@echo off
REM Sahl UFC Predictor - One-Click Bootstrap (Windows)
REM Usage:  run.bat           (synthetic data, fast)
REM         run.bat --real    (scrape ufcstats.com, ~1 hour)

cd /d "%~dp0"

echo === Sahl UFC Predictor ===
echo Installing dependencies...
pip install -q -r requirements.txt
if errorlevel 1 exit /b 1

cd backend

if "%1"=="--real" (
    echo.
    echo [1/3] Scraping ufcstats.com ^(last 10 years, ~1 hour^)...
    python scraper.py --output ../data/ufc_fights_raw.csv --years 10
    echo.
    echo [2/3] Building features...
    python features.py
) else (
    echo.
    echo [1/2] Generating synthetic data + features...
    python features.py --synthetic
)

echo.
echo [Training] XGBoost + calibration + SHAP...
python train.py

echo.
echo ===============================================
echo   Sahl UFC Predictor is starting at:
echo     http://localhost:8000
echo   Press Ctrl+C to stop.
echo ===============================================
echo.
python server.py
