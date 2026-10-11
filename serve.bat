@echo off
rem Preview at http://localhost:8001/e-shikaku/ with Zensical.
rem First-time setup (run once in this folder):
rem   python -m venv .venv
rem   .venv\Scripts\pip install -r requirements.txt
set PYTHONUTF8=1
cd /d %~dp0
.venv\Scripts\zensical.exe serve -a 127.0.0.1:8001
pause
