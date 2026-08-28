@echo off
if not exist ".env" (
    echo ERROR: .env file not found
    echo Copy .env.example to .env and fill in your HF_TOKEN
    exit /b 1
)

for /f "delims== tokens=1,2" %%a in (.env) do (
    if not "%%a"=="" if not "%%a:~0,1%"=="#" (
        set %%a=%%b
    )
)

echo HF_TOKEN loaded from .env
