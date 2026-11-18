@echo off
rem Double-click: opens the livewatch dashboard and starts watching.
rem Keep this window open (minimise it). If livewatch ever crashes it restarts
rem by itself and resumes watching. Close the window to stop it.
cd /d "%~dp0"
title livewatch - keep this window open
set FIRST=1

:run
if "%FIRST%"=="1" (
  python -m livewatch ui --autostart
) else (
  python -m livewatch ui --autostart --no-browser
)
set CODE=%ERRORLEVEL%
if "%CODE%"=="0" goto end
if "%CODE%"=="3" goto busy
echo.
echo livewatch stopped unexpectedly (exit code %CODE%). Restarting in 5 seconds...
echo %date% %time% restart after exit code %CODE%>> livewatch_restarts.log
set FIRST=0
timeout /t 5 /nobreak >nul
goto run

:busy
pause
goto end

:end
