@echo off
title LivePortrait Batch Processor v5.0 Enhanced Launcher

echo.
echo ========================================================================
echo  LivePortrait Batch Processor v5.0 - ENHANCED WITH DETAILED PROGRESS
echo ========================================================================
echo.
echo Found enhanced script: %~dp0enhanced_lp_batch_v5.py
echo.

REM Check if Python is available
where python >nul 2>&1
if %errorlevel% == 0 (
    echo Using system Python: python
    set PYTHON_CMD=python
) else (
    where py >nul 2>&1
    if %errorlevel% == 0 (
        echo Using Python launcher: py
        set PYTHON_CMD=py
    ) else (
        echo [ERROR] Python not found in PATH
        echo Please install Python or add it to your PATH
        pause
        exit /b 1
    )
)

echo.
echo Testing Python installation...
%PYTHON_CMD% --version

echo.
echo ========================================================================
echo  NEW FEATURES IN V5.0
echo ========================================================================
echo.
echo ENHANCEMENTS APPLIED:
echo - Fixed parallel processing issues [simplified approach]
echo - Added option to skip or reprocess existing folders
echo - Detailed progress showing individual file completion times
echo - Enhanced error handling and recovery
echo - Template [PKL] support for 10x faster processing
echo - Real-time progress with filenames and processing times
echo.
echo PROCESSING BENEFITS:
echo - Sequential mode: More stable, detailed progress per file
echo - Parallel mode: Faster but less detailed [optional]
echo - Template processing: 4-8 seconds vs 40+ seconds for video
echo - Skip processed folders or reprocess everything
echo.
echo If GUI dialogs appear, they might be behind this window.
echo Press Ctrl+C to interrupt processing if needed.

REM Change to script directory
cd /d "%~dp0"

REM Run the enhanced script
echo.
echo Starting Enhanced LivePortrait Batch Processor v5.0...
echo.
%PYTHON_CMD% enhanced_lp_batch_v5.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Script execution failed with exit code %errorlevel%
    echo Check the error messages above for details.
)

echo.
echo Press Enter to exit...
pause >nul
