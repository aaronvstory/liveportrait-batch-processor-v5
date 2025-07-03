@echo off
setlocal enabledelayedexpansion

REM =======================================================================
REM LivePortrait Batch Processor Launcher - WORKING VERSION
REM =======================================================================

echo LivePortrait Batch Processor (Working Version)
echo ---------------------------------

REM --- Set absolute paths ---
set "AUTOMATION_DIR=%~dp0"
if "%AUTOMATION_DIR:~-1%"=="\" set "AUTOMATION_DIR=%AUTOMATION_DIR:~0,-1%"
for /F "delims=" %%i in ("%AUTOMATION_DIR%\..") do set "BASEDIR=%%~fi"
set "SCRIPT_PATH=%AUTOMATION_DIR%\LPbatchV3.5.py"

REM --- Validate script exists ---
if not exist "%SCRIPT_PATH%" (
    echo [ERROR] Cannot find the Python script at:
    echo   %SCRIPT_PATH%
    echo Aborting.
    pause
    goto END
)

REM --- Find Python executable ---
set "VENV_PYTHON=%BASEDIR%\venv\Scripts\python.exe"
set "PYTHON_EXE="

echo Detecting Python installation...

REM Check for virtual environment first
if exist "%VENV_PYTHON%" (
    echo Found virtual environment Python: %VENV_PYTHON%
    set "PYTHON_EXE=%VENV_PYTHON%"
    set "USING_VENV=1"
    goto PYTHON_FOUND
)

echo Virtual environment not found at: %VENV_PYTHON%
echo Searching for system Python...

REM Try to find python in PATH
python --version >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    for /f "delims=" %%i in ('where python 2^>nul') do (
        if "!PYTHON_EXE!"=="" (
            echo Found system Python: %%i
            set "PYTHON_EXE=%%i"
            set "USING_VENV=0"
            goto PYTHON_FOUND
        )
    )
)

REM Try py launcher
py --version >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    echo Found Python Launcher (py.exe)
    set "PYTHON_EXE=py"
    set "USING_VENV=0"
    goto PYTHON_FOUND
)

echo [ERROR] No Python installation found!
echo Please install Python or check your PATH environment variable.
pause
goto END

:PYTHON_FOUND

REM --- Set environment variables ---
if "%USING_VENV%"=="1" (
    echo Setting up CUDA environment for virtual environment...
    set "CUDA_MODULE_LOADING=LAZY"
    set "CUDA_PATH=%BASEDIR%\venv\Lib\site-packages\torch\lib"
    set "PATH=%BASEDIR%\git\cmd;%BASEDIR%\python;%BASEDIR%\venv\scripts;%BASEDIR%\ffmpeg;%BASEDIR%\venv\Lib\site-packages\torch\lib;%PATH%"

    REM Activate virtual environment
    echo Activating Python virtual environment...
    set "VENV_ACTIVATE=%BASEDIR%\venv\Scripts\activate.bat"
    if exist "%VENV_ACTIVATE%" (
        call "%VENV_ACTIVATE%"
    )
) else (
    echo Using system Python - some packages may need to be installed globally.
)

REM --- Check for critical dependencies ---
echo.
echo Checking for critical LivePortrait dependencies...

echo Checking onnxruntime...
"%PYTHON_EXE%" -c "import onnxruntime" 2>nul
if !ERRORLEVEL! NEQ 0 (
    echo Installing onnxruntime-gpu...
    "%PYTHON_EXE%" -m pip install onnxruntime-gpu
)

echo Checking cv2...
"%PYTHON_EXE%" -c "import cv2" 2>nul
if !ERRORLEVEL! NEQ 0 (
    echo Installing opencv-python...
    "%PYTHON_EXE%" -m pip install opencv-python==4.10.0.84
)

echo Checking torch...
"%PYTHON_EXE%" -c "import torch" 2>nul
if !ERRORLEVEL! NEQ 0 (
    echo Installing torch...
    "%PYTHON_EXE%" -m pip install torch torchvision torchaudio
)

echo Checking numpy...
"%PYTHON_EXE%" -c "import numpy" 2>nul
if !ERRORLEVEL! NEQ 0 (
    echo Installing numpy...
    "%PYTHON_EXE%" -m pip install numpy==1.26.4
)

echo Checking other essential packages...
"%PYTHON_EXE%" -c "import tqdm" 2>nul
if !ERRORLEVEL! NEQ 0 (
    echo Installing tqdm...
    "%PYTHON_EXE%" -m pip install tqdm==4.66.4
)

"%PYTHON_EXE%" -c "import rich" 2>nul
if !ERRORLEVEL! NEQ 0 (
    echo Installing rich...
    "%PYTHON_EXE%" -m pip install rich==13.7.1
)

"%PYTHON_EXE%" -c "import imageio" 2>nul
if !ERRORLEVEL! NEQ 0 (
    echo Installing imageio...
    "%PYTHON_EXE%" -m pip install imageio==2.34.2
)

"%PYTHON_EXE%" -c "import skimage" 2>nul
if !ERRORLEVEL! NEQ 0 (
    echo Installing scikit-image...
    "%PYTHON_EXE%" -m pip install scikit-image==0.24.0
)

REM Check for batch script dependencies
"%PYTHON_EXE%" -c "import pyfiglet" 2>nul
if !ERRORLEVEL! NEQ 0 (
    echo Installing pyfiglet...
    "%PYTHON_EXE%" -m pip install pyfiglet
)

REM --- Change to automation directory ---
cd /D "%AUTOMATION_DIR%"

REM --- Run the Python script ---
echo.
echo Starting LivePortrait batch processor...
echo Python executable: %PYTHON_EXE%
echo Script path: %SCRIPT_PATH%
echo.
echo You will see prompts below - please respond to them when asked.
echo.
echo [If a GUI dialog opens, it might appear behind this window]
echo.

REM --- Enhanced retry logic ---
set RETRY_COUNT=0
:RETRY_LPBATCH
"%PYTHON_EXE%" "%SCRIPT_PATH%" %*
if !ERRORLEVEL! EQU 0 goto SUCCESS

set /a RETRY_COUNT+=1
if !RETRY_COUNT! GEQ 2 goto FAILURE

echo.
echo [RETRY] Script failed, installing complete LivePortrait package set...

REM Install complete LivePortrait dependencies
"%PYTHON_EXE%" -m pip install --upgrade pip
"%PYTHON_EXE%" -m pip install onnxruntime-gpu
"%PYTHON_EXE%" -m pip install "tokenizers==0.19.1"
"%PYTHON_EXE%" -m pip install torch torchvision torchaudio
"%PYTHON_EXE%" -m pip install "numpy==1.26.4"
"%PYTHON_EXE%" -m pip install "pyyaml==6.0.1"
"%PYTHON_EXE%" -m pip install "opencv-python==4.10.0.84"
"%PYTHON_EXE%" -m pip install "scipy==1.13.1"
"%PYTHON_EXE%" -m pip install "imageio==2.34.2"
"%PYTHON_EXE%" -m pip install "lmdb==1.4.1"
"%PYTHON_EXE%" -m pip install "tqdm==4.66.4"
"%PYTHON_EXE%" -m pip install "rich==13.7.1"
"%PYTHON_EXE%" -m pip install "ffmpeg-python==0.2.0"
"%PYTHON_EXE%" -m pip install "onnx==1.16.1"
"%PYTHON_EXE%" -m pip install "scikit-image==0.24.0"
"%PYTHON_EXE%" -m pip install "albumentations==1.4.10"
"%PYTHON_EXE%" -m pip install "matplotlib==3.9.0"
"%PYTHON_EXE%" -m pip install "imageio-ffmpeg==0.5.1"
"%PYTHON_EXE%" -m pip install "tyro==0.8.5"
"%PYTHON_EXE%" -m pip install "gradio==4.37.1"
"%PYTHON_EXE%" -m pip install "pykalman==0.9.7"
"%PYTHON_EXE%" -m pip install transformers
"%PYTHON_EXE%" -m pip install "pillow>=10.2.0"
"%PYTHON_EXE%" -m pip install pyfiglet

echo Retrying script execution...
goto RETRY_LPBATCH

:SUCCESS
echo.
echo Batch processing completed successfully.
goto END

:FAILURE
echo.
echo [ERROR] Python script failed after retries. Exit code: !ERRORLEVEL!
echo.
echo Possible solutions:
echo 1. Check if you have administrative privileges to install packages
echo 2. Try running as administrator
echo 3. Check your internet connection
echo 4. Verify Python installation
echo.
echo Check the log file for details: %AUTOMATION_DIR%\liveportrait_batch_log.txt
goto END

:END
echo.
echo Press any key to close this window...
pause >nul
endlocal
