@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo Zing-0.5 - Running New Examples Only
echo ==========================================
echo.

cd /d "%~dp0"

call .venv\Scripts\activate.bat
set PYTHONPATH=%cd%\src;%PYTHONPATH%
set CUDA_VISIBLE_DEVICES=0

set PRETRAINED_DIR=%cd%\pretrained_models
set OUTPUT_DIR=%cd%\outputs
set SNAPSHOTS_DIR=%PRETRAINED_DIR%\models--seedleap--zing-0.5\snapshots

for /d %%D in ("%SNAPSHOTS_DIR%\*") do (
    set SNAPSHOT_DIR=%%D
)

set PRETRAINED_PATH=%SNAPSHOT_DIR%\pretrained
set CHECKPOINT_PATH=%SNAPSHOT_DIR%\generator\model.pt

REM Define test cases
set CASES=case3_action_t2v case4_action_ti2v case5_action_ti2v case6_prompt_switch_t2v case7_action_prompt_switch_t2v sample_input

for %%C in (%CASES%) do (
    if exist "examples\%%C.jsonl" (
        if not exist "!OUTPUT_DIR!\%%C" (
            echo.
            echo ==========================================
            echo Running: %%C
            echo ==========================================

            python -m zing_v0_5 ^
              --pretrained-dir "!PRETRAINED_PATH!" ^
              --checkpoint "!CHECKPOINT_PATH!" ^
              --messages "examples\%%C.jsonl" ^
              --output-dir "!OUTPUT_DIR!\%%C" ^
              --seed 0

            if errorlevel 1 (
                echo.
                echo WARNING: %%C failed
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
echo Done! Check outputs/ for results
echo ==========================================
echo.
pause
