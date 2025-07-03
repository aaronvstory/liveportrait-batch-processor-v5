@echo off
title LivePortrait Batch Processor v5.0 - Initial Setup

echo.
echo ========================================================================
echo  LivePortrait Batch Processor v5.0 - First Time Setup
echo ========================================================================
echo.
echo This script will:
echo 1. Create a Python virtual environment
echo 2. Install required dependencies
echo 3. Create your configuration file
echo 4. Verify the installation
echo.

REM Check admin privileges
net session >nul 2>&1
if NOT %errorLevel% == 0 (
    echo [WARNING] This script may require Administrator privileges for some operations.
    echo If you encounter permission errors, right-click and "Run as Administrator"
    echo.
)

REM Check if Python is available
echo [INFO] Checking Python installation...
where python >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Found Python: python
    set PYTHON_CMD=python
) else (
    where py >nul 2>&1
    if %errorlevel% == 0 (
        echo [OK] Found Python launcher: py
        set PYTHON_CMD=py
    ) else (
        echo [ERROR] Python not found in PATH
        echo Please install Python 3.8+ from https://python.org
        echo Make sure to check "Add Python to PATH" during installation
        pause
        exit /b 1
    )
)

REM Display Python version
echo [INFO] Python version:
%PYTHON_CMD% --version

REM Change to script directory
cd /d "%~dp0"

REM Check if virtual environment exists
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment already exists at .venv
    echo [INFO] Activating existing environment...
    call .venv\Scripts\activate.bat
) else (
    echo [INFO] Creating Python virtual environment...
    %PYTHON_CMD% -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        echo Make sure you have Python 3.8+ installed
        pause
        exit /b 1
    )
    
    echo [OK] Virtual environment created successfully
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

REM Install requirements
echo [INFO] Installing required packages...
if exist "requirements.txt" (
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install requirements
        pause
        exit /b 1
    )
    echo [OK] All packages installed successfully
) else (
    echo [WARNING] requirements.txt not found, installing Rich manually...
    python -m pip install rich>=13.0.0
)

REM Create config file if it doesn't exist
if not exist "liveportrait_batch_config.ini" (
    if exist "config_template.ini" (
        echo [INFO] Creating configuration file from template...
        copy "config_template.ini" "liveportrait_batch_config.ini" >nul
        echo [OK] Configuration file created: liveportrait_batch_config.ini
        echo [IMPORTANT] Please edit liveportrait_batch_config.ini with your paths before running
    ) else (
        echo [WARNING] No configuration template found
        echo You will need to configure paths on first run
    )
) else (
    echo [INFO] Configuration file already exists
)

REM Create driving_templates directory
if not exist "driving_templates" (
    mkdir "driving_templates"
    echo [OK] Created driving_templates directory
)

REM Test installation
echo.
echo [INFO] Testing installation...
python -c "import rich; print('[OK] Rich library is working properly')" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Installation test failed
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo  Setup Complete!
echo ========================================================================
echo.
echo Next steps:
echo 1. Edit liveportrait_batch_config.ini with your LivePortrait paths
echo 2. Run collect_templates.bat to gather PKL templates (optional)
echo 3. Run launch_v5.bat to start the batch processor
echo.
echo The virtual environment has been created at .venv
echo It will be automatically activated when you run launch_v5.bat
echo.
echo Press Enter to continue...
pause >nul
