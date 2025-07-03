@echo off
title LivePortrait Batch Processor v5.0

echo.
echo ========================================================================
echo  LivePortrait Batch Processor v5.0 - Professional Batch Animation
echo ========================================================================
echo.
echo Advanced batch processing with detailed progress tracking, template
echo support, and parallel processing. Process hundreds of images efficiently
echo with PKL templates for 10x faster processing speeds.
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if setup has been run
if not exist ".venv\Scripts\activate.bat" (
    echo [WARNING] Virtual environment not found
    echo Please run setup.bat first to configure the environment
    echo.
    pause
    exit /b 1
)

REM Check if config exists
if not exist "liveportrait_batch_config.ini" (
    if exist "config_template.ini" (
        echo [WARNING] Configuration file not found
        echo Creating default configuration from template...
        copy "config_template.ini" "liveportrait_batch_config.ini" >nul
        echo [IMPORTANT] Please edit liveportrait_batch_config.ini with your paths
        echo.
    ) else (
        echo [ERROR] Configuration template missing
        echo Please ensure config_template.ini exists and run setup.bat
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    echo Please run setup.bat to fix the installation
    pause
    exit /b 1
)

REM Verify Python in virtual environment
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not available in virtual environment
    echo Please run setup.bat to fix the installation
    pause
    exit /b 1
)

REM Check if main script exists
if not exist "enhanced_lp_batch_v5.py" (
    echo [ERROR] Main script enhanced_lp_batch_v5.py not found
    echo Please ensure all files are present in the project directory
    pause
    exit /b 1
)

echo [OK] Environment ready
echo.
echo [INFO] GUI dialogs may appear behind this window
echo [INFO] Press Ctrl+C to interrupt processing if needed
echo.
echo Starting LivePortrait Batch Processor v5.0...
echo.

REM Run the enhanced script
python enhanced_lp_batch_v5.py

REM Check exit code
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Script execution failed with exit code %errorlevel%
    echo Check the error messages above for details.
    echo If you need help, check the README.md file or the log files.
)

echo.
echo Press Enter to exit...
pause >nul
