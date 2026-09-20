@echo off
chcp 65001 >nul
cd /d %~dp0

if not exist .venv (
  echo [1/4] Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 (
    echo Python not found. Install from python.org and check "Add to PATH".
    pause
    exit /b 1
  )
  call .venv\Scripts\activate
  echo [2/4] Installing packages...
  pip install -r requirements.txt

) else (
  call .venv\Scripts\activate
  pip install -q -r requirements.txt
)

if not exist .env (
  copy .env.example .env >nul
  echo.
  echo ========================================
  echo .env created. Edit it if needed, then re-run.
  echo ========================================
  notepad .env
  pause
  exit /b 0
)

echo [3/4] Starting Micron Tools AI Consultant...
echo Open: http://127.0.0.1:8000
echo Docs: http://127.0.0.1:8000/docs
echo.
uvicorn main:app --reload --host 0.0.0.0 --port 8000
pause

