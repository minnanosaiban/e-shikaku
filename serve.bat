@echo off
cd /d %~dp0
python -m mkdocs serve --livereload -a 127.0.0.1:8001
pause
