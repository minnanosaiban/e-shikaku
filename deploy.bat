@echo off
setlocal
rem ============================================================
rem  e-shikaku deploy (GitHub Pages, built by GitHub Actions)
rem  Double-click after editing to publish the site.
rem    1. Build check with Zensical. Stops here if the build fails.
rem    2. Commit and push to GitHub (main).
rem    3. GitHub Actions builds and publishes the site (about 30 seconds).
rem  First-time setup (run once in this folder):
rem    python -m venv .venv
rem    .venv\Scripts\pip install -r requirements.txt
rem  Site:    https://minnanosaiban.github.io/e-shikaku/
rem  Actions: https://github.com/minnanosaiban/e-shikaku/actions
rem ============================================================
set PYTHONUTF8=1
cd /d "%~dp0"

if not exist ".venv\Scripts\zensical.exe" (
    echo [ERROR] .venv\Scripts\zensical.exe was not found.
    echo         Run once in this folder:
    echo           python -m venv .venv
    echo           .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo === Build check ===
.venv\Scripts\zensical.exe build --clean
if %errorlevel% neq 0 (
    echo [ERROR] Build failed. Nothing was pushed.
    pause
    exit /b 1
)

echo === Commit and push to GitHub, main ===
git add .
git commit -m "Update e-shikaku" || echo No changes to commit
git push origin main
if %errorlevel% neq 0 (
    echo [ERROR] Push failed.
    pause
    exit /b 1
)

echo === Done. GitHub Actions is publishing the site. ===
pause
