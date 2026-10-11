@echo off
rem Build the site into site\ with Zensical.
set PYTHONUTF8=1
cd /d %~dp0
.venv\Scripts\zensical.exe build --clean
pause
