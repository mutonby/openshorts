@echo off
rem Double-click to open the livewatch dashboard in your browser.
cd /d "%~dp0"
title livewatch
python -m livewatch ui
if errorlevel 1 pause
