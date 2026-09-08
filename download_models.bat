@echo off
setlocal enabledelayedexpansion
rem Download zing-0.5 pretrained weights.
rem
rem A wrapper around _download_models_helper.py, which holds the actual logic --
rem shared with download_models.sh so the two platforms cannot drift apart. The
rem repository previously carried three separate copies of the same
rem snapshot_download call, each with different bugs.

cd /d "%~dp0"
if not defined MODELS_DIR set "MODELS_DIR=%cd%\pretrained_models"

echo ==========================================
echo Zing-0.5 model download
echo ==========================================
echo.

if not defined HF_TOKEN (
  echo ERROR: HF_TOKEN is not set.
  echo   set HF_TOKEN=^<token^>
  exit /b 1
)

rem The venv's interpreter when it exists, so huggingface_hub is found without
rem the caller having to activate anything first.
set "PYTHON=%cd%\.venv\Scripts\python.exe"
if not exist "!PYTHON!" set "PYTHON=python"

"!PYTHON!" "%cd%\_download_models_helper.py" "%MODELS_DIR%"
if errorlevel 1 exit /b 1

echo.
echo ==========================================
echo Next: run_rtx5090_windows.bat
echo ==========================================
endlocal
