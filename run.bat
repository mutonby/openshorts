@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set ROOT=%~dp0
set VENV_DIR=.\venv
set BACKEND_PORT=8000
set RENDERER_PORT=3100

echo ==========================================
echo   OpenShorts Launcher
echo ==========================================
echo.

:: ── Python venv ──────────────────────────────────────────────
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [1/6] Creating Python virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create venv. Make sure Python 3.11+ is installed and on PATH.
        pause & exit /b 1
    )
    echo       venv created successfully.
) else (
    echo [1/6] Python venv already exists, skipping creation.
)

echo [2/6] Activating venv...
call "%VENV_DIR%\Scripts\activate.bat"

echo [3/6] Installing/updating Python dependencies...
pip install --upgrade pip --quiet
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir
pip install -r requirements.txt
pip install --upgrade --no-cache-dir yt-dlp
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies.
    pause & exit /b 1
)
call deactivate

:: ── Frontend dependencies ────────────────────────────────────
echo [4/6] Installing frontend dependencies...
cd /d "%ROOT%dashboard"
call npm install
if errorlevel 1 (
    echo ERROR: Failed to install frontend dependencies.
    pause & exit /b 1
)

:: ── Render service dependencies ──────────────────────────────
echo [5/6] Installing render service dependencies...
cd /d "%ROOT%render-service"
call npm install
if errorlevel 1 (
    echo ERROR: Failed to install render service dependencies.
    pause & exit /b 1
)

echo [+] Installing remotion project dependencies...
cd /d "%ROOT%remotion"
call npm install
if errorlevel 1 (
    echo ERROR: Failed to install remotion dependencies.
    pause & exit /b 1
)

cd /d "%ROOT%"

:: ── Launch all services ──────────────────────────────────────
echo [6/6] Launching services...
echo.

start "OpenShorts Backend"  cmd /k "cd /d %ROOT% && call %VENV_DIR%\Scripts\activate.bat && set RENDER_SERVICE_URL=http://localhost:%RENDERER_PORT% && set MAX_CONCURRENT_JOBS=5 && echo Backend: http://localhost:%BACKEND_PORT% && echo Docs: http://localhost:%BACKEND_PORT%/docs && echo. && uvicorn app:app --host 0.0.0.0 --port %BACKEND_PORT%"

start "Remotion Renderer"   cmd /k "cd /d %ROOT%render-service && set PORT=%RENDERER_PORT% && set ""OUTPUT_DIR=%ROOT%output"" && set ""REMOTION_BUNDLE_PATH=%ROOT%remotion"" && echo Renderer: http://localhost:%RENDERER_PORT% && echo. && npx tsx src/server.ts"

start "OpenShorts Frontend" cmd /k "cd /d %ROOT%dashboard && echo Frontend: http://localhost:5173 && echo. && npm run dev"

echo ==========================================
echo   Services started in separate windows:
echo     Backend:   http://localhost:%BACKEND_PORT%
echo     Renderer:  http://localhost:%RENDERER_PORT%
echo     Frontend:  http://localhost:5173
echo   Close each window to stop that service.
echo ==========================================
echo.
echo   NOTE: The renderer requires Chromium installed.
echo   Install from https://www.chromium.org/
echo   and set PUPPETEER_EXECUTABLE_PATH if needed.
echo ==========================================

endlocal
