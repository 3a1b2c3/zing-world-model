@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo Zing-0.5 on RTX 5090 (32GB)
echo ==========================================
echo.

cd /d "%~dp0"

REM Check if .venv exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: .venv not found
    echo Run: setup_venv.bat
    exit /b 1
)

REM Activate venv
call .venv\Scripts\activate.bat

REM Set environment variables
set PYTHONPATH=%cd%\src;%PYTHONPATH%
set CUDA_VISIBLE_DEVICES=0

set PRETRAINED_DIR=%cd%\pretrained_models
set OUTPUT_DIR=%cd%\outputs
set SNAPSHOTS_DIR=%PRETRAINED_DIR%\models--seedleap--zing-0.5\snapshots

if not exist "%PRETRAINED_DIR%" (
    echo ERROR: pretrained_models/ not found
    exit /b 1
)

REM Find the latest snapshot (first by directory sort)
for /d %%D in ("%SNAPSHOTS_DIR%\*") do (
    set SNAPSHOT_DIR=%%D
)

if not exist "%SNAPSHOT_DIR%\pretrained\text_encoder" (
    echo ERROR: Could not find Zing-0.5 models
    exit /b 1
)

set PRETRAINED_PATH=%SNAPSHOT_DIR%\pretrained
set CHECKPOINT_PATH=%SNAPSHOT_DIR%\generator\model.pt

echo Snapshot: %SNAPSHOT_DIR%
echo.

REM Define all example cases (lowres: 640x360)
set CASES=case4_action_ti2v_lowres case5_action_ti2v_lowres case6_prompt_switch_t2v_lowres case7_action_prompt_switch_t2v_lowres sample_input_lowres

for %%C in (%CASES%) do (
    if exist "examples\%%C.jsonl" (
        if not exist "%OUTPUT_DIR%\%%C" (
            echo.
            echo ==========================================
            echo Running: %%C
            echo ==========================================
            echo.

            python -m zing_v0_5 ^
              --pretrained-dir "%PRETRAINED_PATH%" ^
              --checkpoint "%CHECKPOINT_PATH%" ^
              --messages "examples\%%C.jsonl" ^
              --output-dir "%OUTPUT_DIR%\%%C" ^
              --seed 0

            if errorlevel 1 (
                echo.
                echo WARNING: %%C failed, continuing...
                echo.
            ) else (
                echo.
                echo OK: %%C complete
                echo.
            )
        ) else (
            echo Skipping %%C (already exists)
        )
    )
)

echo.
echo ==========================================
echo All examples complete!
echo Check outputs/ for results
echo ==========================================
echo.
pause
