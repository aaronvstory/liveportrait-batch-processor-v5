@echo off
title Template Manager - Collect PKL Files

echo.
echo ========================================================================
echo  LivePortrait Template Manager v5.0 - PKL Template Collection
echo ========================================================================
echo.
echo This tool scans LivePortrait temp folders for .pkl template files
echo and copies them to the driving_templates folder for 10x faster processing.
echo Templates eliminate the need to process driving videos repeatedly.
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [WARNING] Virtual environment not found
    echo Please run setup.bat first to configure the environment
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    echo Please run setup.bat to fix the installation
    pause
    exit /b 1
)

REM Check if template manager exists
if not exist "template_manager.py" (
    echo [ERROR] Template manager script not found
    echo Please ensure template_manager.py exists in the project directory
    pause
    exit /b 1
)

REM Create driving_templates directory if it doesn't exist
if not exist "driving_templates" (
    mkdir "driving_templates"
    echo [OK] Created driving_templates directory
)

echo [OK] Environment ready
echo.
echo Starting template collection...
echo.

REM Run the template manager
python template_manager.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Template collection failed with exit code %errorlevel%
    echo Check the error messages above for details.
)

echo.
echo Press Enter to exit...
pause >nul
