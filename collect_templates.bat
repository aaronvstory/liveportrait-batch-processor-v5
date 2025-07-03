@echo off
title Template Manager - Collect PKL Files

echo.
echo ========================================================================
echo  LivePortrait Template Manager v5.0
echo ========================================================================
echo.
echo This tool will scan LivePortrait temp folders for .pkl template files
echo and copy them to the driving_templates folder for faster processing.
echo.

REM Check if Python is available
where python >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=python
) else (
    where py >nul 2>&1
    if %errorlevel% == 0 (
        set PYTHON_CMD=py
    ) else (
        echo [ERROR] Python not found in PATH
        pause
        exit /b 1
    )
)

REM Change to script directory
cd /d "%~dp0"

REM Run the template manager
%PYTHON_CMD% template_manager.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Template manager failed with exit code %errorlevel%
)

echo.
pause
