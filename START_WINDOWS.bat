@echo off
cd /d "%~dp0"
echo ============================================
echo  ProQuant One-Click Setup & Daily Scanner
echo ============================================

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed. Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo [OK] Created default .env configuration (Capital: Rs 5,00,000).
)

echo [1/3] Installing required packages...
python -m pip install --quiet -r requirements.txt

echo [2/3] Running today's 4-engine market scan...
python run_daily.py

echo [3/3] Opening your Groww Call Cards in browser...
for /f "delims=" %%F in ('dir /b /o-d reports\daily_calls_*.html 2^>nul') do (
    start "" "reports\%%F"
    goto :opened
)
:opened

echo.
echo Setting up automatic weekday 9:30 AM scan (Windows Task Scheduler)...
schtasks /create /tn "ProQuantDailyScan" /tr "cmd /c cd /d \"%~dp0\" && python run_daily.py" /sc weekly /d MON,TUE,WED,THU,FRI /st 09:30 /f >nul 2>&1
if errorlevel 1 (
    echo [INFO] Auto-schedule skipped (run as Administrator if you want 9:30 AM auto-runs).
) else (
    echo [OK] Scheduled automatic daily scan at 9:30 AM Mon-Fri!
)

echo.
echo All done! Review the browser report before placing any manual order on Groww.
pause
