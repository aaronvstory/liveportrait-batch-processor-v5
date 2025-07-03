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

REM --- Detect LivePortrait installation and Python like v3.5 ---
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Try to read LivePortrait path from config
set "LP_REPO_PATH="
if exist "liveportrait_batch_config.ini" (
    for /f "tokens=2 delims== " %%i in ('findstr "liveportrait_repo_path" liveportrait_batch_config.ini 2^>nul') do set "LP_REPO_PATH=%%i"
)

REM Find the best Python executable
set "PYTHON_EXE="
set "USING_LP_VENV=0"

REM First try LivePortrait's venv (like v3.5)
if not "%LP_REPO_PATH%"=="" if exist "%LP_REPO_PATH%\venv\Scripts\python.exe" (
    echo [INFO] Found LivePortrait virtual environment Python: %LP_REPO_PATH%\venv\Scripts\python.exe
    set "PYTHON_EXE=%LP_REPO_PATH%\venv\Scripts\python.exe"
    set "USING_LP_VENV=1"
    goto PYTHON_FOUND
)

REM Check if local setup has been run
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Found local virtual environment
    set "PYTHON_EXE=%SCRIPT_DIR%\.venv\Scripts\python.exe"
    goto PYTHON_FOUND
)

echo [WARNING] Virtual environment not found
echo Please run setup.bat first or ensure LivePortrait is properly installed
echo.

REM Try system Python as fallback
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Using system Python as fallback
    set "PYTHON_EXE=python"
    goto PYTHON_FOUND
)

echo [ERROR] No Python installation found
pause
exit /b 1

:PYTHON_FOUND

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

REM Setup environment like v3.5
if "%USING_LP_VENV%"=="1" (
    echo [INFO] Setting up CUDA environment for LivePortrait virtual environment...
    set "CUDA_MODULE_LOADING=LAZY"
    if exist "%LP_REPO_PATH%\venv\Lib\site-packages\torch\lib" (
        set "CUDA_PATH=%LP_REPO_PATH%\venv\Lib\site-packages\torch\lib"
        set "PATH=%LP_REPO_PATH%\venv\scripts;%LP_REPO_PATH%\venv\Lib\site-packages\torch\lib;%PATH%"
    )
    
    REM Activate LivePortrait virtual environment
    echo [INFO] Activating LivePortrait virtual environment...
    if exist "%LP_REPO_PATH%\venv\Scripts\activate.bat" (
        call "%LP_REPO_PATH%\venv\Scripts\activate.bat"
    )
) else if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating local virtual environment...
    call .venv\Scripts\activate.bat
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to activate local virtual environment
        echo Please run setup.bat to fix the installation
        pause
        exit /b 1
    )
)

REM Verify Python executable
echo [INFO] Testing Python executable: %PYTHON_EXE%
"%PYTHON_EXE%" --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python executable not working: %PYTHON_EXE%
    echo Please check your Python installation
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

:MAIN_LOOP

REM Run the enhanced script
echo [INFO] Using Python: %PYTHON_EXE%
"%PYTHON_EXE%" enhanced_lp_batch_v5.py

REM Check exit code
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Script execution failed with exit code %errorlevel%
    echo Check the error messages above for details.
    echo If you need help, check the README.md file or the log files.
)

echo.
echo =====================================================================
echo  Press ENTER to restart the application
echo  Press Ctrl+C to exit
echo =====================================================================
echo.
pause >nul

REM Clear screen and restart
cls
goto MAIN_LOOP
