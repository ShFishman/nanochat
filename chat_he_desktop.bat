@echo off
REM Windows launcher for the nanochat-hebrew desktop chat app.
REM First run: creates .venv-desktop, installs CPU torch + minimal deps.
REM Later runs: just activates the venv and launches the app.

setlocal
cd /d "%~dp0"

set VENV=.venv-desktop
set PYEXE=%VENV%\Scripts\python.exe

if not exist "%VENV%" (
    echo [setup] creating virtual environment in %VENV% ...
    where py >nul 2>nul
    if errorlevel 1 (
        python -m venv "%VENV%"
    ) else (
        py -3 -m venv "%VENV%"
    )
    if errorlevel 1 (
        echo ERROR: could not create venv. Make sure Python 3.10+ is installed.
        pause
        exit /b 1
    )

    echo [setup] upgrading pip ...
    "%PYEXE%" -m pip install --upgrade pip wheel

    echo [setup] installing CPU-only torch (this is the slowest step, ~250MB) ...
    "%PYEXE%" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.9.1"
    if errorlevel 1 (
        echo ERROR: torch install failed.
        pause
        exit /b 1
    )

    echo [setup] installing chat dependencies ...
    "%PYEXE%" -m pip install ^
        "huggingface_hub>=0.24" ^
        "tokenizers>=0.22.0" ^
        "tiktoken>=0.11.0" ^
        "rustbpe>=0.1.0" ^
        "filelock>=3.13"
    if errorlevel 1 (
        echo ERROR: dependency install failed.
        pause
        exit /b 1
    )
    echo [setup] done.
    echo.
)

echo [launch] starting desktop app ...
"%PYEXE%" scripts_he\chat_desktop_he.py
if errorlevel 1 pause
endlocal
