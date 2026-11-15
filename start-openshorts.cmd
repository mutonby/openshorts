@echo off
setlocal
cd /d "%~dp0"
title OpenShorts launcher

where ollama >nul 2>nul || (echo Ollama is not installed or not on PATH. & pause & exit /b 1)
where bun >nul 2>nul || (echo Bun is not installed or not on PATH. & pause & exit /b 1)
if not exist ".venv\Scripts\python.exe" (echo Python venv missing: .venv\Scripts\python.exe & pause & exit /b 1)

curl -s -o nul --max-time 3 http://localhost:11434/api/tags
if errorlevel 1 (
  echo Starting Ollama...
  start "OpenShorts - Ollama" /min cmd /c "ollama serve"
) else (
  echo Ollama already running.
)

curl -s -o nul --max-time 3 http://127.0.0.1:8000/api/config
if errorlevel 1 (
  echo Starting API...
  start "OpenShorts - API" cmd /k ".venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000"
) else (
  echo API already running.
)

curl -s -o nul --max-time 3 http://localhost:5173/
if errorlevel 1 (
  echo Starting dashboard...
  start "OpenShorts - Dashboard" cmd /k "cd /d dashboard && set "VITE_PROXY_TARGET=http://127.0.0.1:8000" && bun run dev"
) else (
  echo Dashboard already running.
)

echo Waiting for the API...
set /a tries=0
:wait_api
curl -s -o nul --max-time 3 http://127.0.0.1:8000/api/config
if not errorlevel 1 goto ready
set /a tries+=1
if %tries% geq 60 (echo The API did not start: check the "OpenShorts - API" window. & pause & exit /b 1)
ping -n 3 127.0.0.1 >nul
goto wait_api

:ready
echo Ready: opening http://localhost:5173/#app
start "" "http://localhost:5173/#app"
endlocal
