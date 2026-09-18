@echo off
setlocal
echo ========================================================
echo     FAMELab Matfile Web App - Desktop Compilation       
echo ========================================================

:: Set directories
set SCRIPT_DIR=%~dp0
set FRONTEND_DIR=%SCRIPT_DIR%frontend
set BACKEND_DIR=%SCRIPT_DIR%backend
set OUTPUT_DIR=%SCRIPT_DIR%dist_desktop

:: 0. Check and Install Required System Dependencies
echo.
echo [Step 0/3] Checking system dependencies (Node.js and Python)...

:: Check Node.js
set NODE_MAJOR=0
for /f "tokens=1 delims=v." %%a in ('node -v 2^>nul') do (
    set NODE_MAJOR=%%a
)

if %NODE_MAJOR% LSS 20 (
    echo [Notice] Node.js is missing or too old ^(Current: v%NODE_MAJOR%^). Required: v20+
    if %NODE_MAJOR% EQU 0 (
        echo Installing Node.js LTS via winget...
        winget install OpenJS.NodeJS.LTS --accept-source-agreements --accept-package-agreements
    ) else (
        echo Upgrading Node.js LTS via winget...
        winget upgrade OpenJS.NodeJS.LTS --accept-source-agreements --accept-package-agreements
    )
    echo ========================================================
    echo  Node.js has been installed or updated!
    echo  IMPORTANT: You must CLOSE this window and RUN this script again
    echo  so that the new Node.js is available in your PATH.
    echo ========================================================
    pause
    exit /b 1
)

:: Check Python
set PYTHON_MAJOR=0
for /f "tokens=2 delims=. " %%a in ('python --version 2^>nul') do (
    set PYTHON_MAJOR=%%a
)

if %PYTHON_MAJOR% LSS 3 (
    echo [Notice] Python 3 is missing.
    echo Installing Python 3 via winget...
    winget install Python.Python.3.12 --accept-source-agreements --accept-package-agreements
    echo ========================================================
    echo  Python 3 has been installed!
    echo  IMPORTANT: You must CLOSE this window and RUN this script again
    echo  so that Python is available in your PATH.
    echo ========================================================
    pause
    exit /b 1
)

echo [Step 0/3] System dependencies are up to date!

:: 1. Build the Frontend Production Bundle
echo.
echo [Step 1/3] Building frontend static bundle (Vite)...
cd /d "%FRONTEND_DIR%"

echo Installing frontend dependencies...
call npm install
if errorlevel 1 (
    echo [Error] npm install failed. Make sure Node.js is installed!
    exit /b 1
)

echo Building frontend...
call npm run build
if errorlevel 1 (
    echo [Error] Frontend build failed.
    exit /b 1
)

if not exist "%FRONTEND_DIR%\dist\index.html" (
    echo [Error] Frontend build failed: index.html not found in dist.
    exit /b 1
)
echo [Step 1/3] Frontend build successful!

:: 2. Check Nuitka Installation
echo.
echo [Step 2/3] Checking Nuitka compiler...
python -m nuitka --version >nul 2>&1
if errorlevel 1 (
    echo [Notice] Nuitka is not currently installed.
    echo          Installing nuitka via pip...
    python -m pip install nuitka
)
echo [Step 2/3] Nuitka is ready.

:: 3. Detect Platform & Compile
echo.
echo [Step 3/3] Compiling standalone desktop application with Nuitka...
cd /d "%BACKEND_DIR%"

echo Installing backend dependencies...
python -m pip install -r requirements.txt
python -m pip install zstandard
if errorlevel 1 (
    echo [Error] pip install failed.
    exit /b 1
)

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: Execute Nuitka compilation for Windows
python -m nuitka ^
    --onefile ^
    --windows-console-mode=disable ^
    --include-data-dir="%FRONTEND_DIR%\dist=frontend_dist" ^
    --include-package=uvicorn ^
    --include-package=fastapi ^
    --include-package=starlette ^
    --include-package=pydantic ^
    --include-package=pyarrow ^
    --include-package=h5py ^
    --include-package=scipy ^
    --include-package=pandas ^
    --include-package=numpy ^
    --include-package=xlsxwriter ^
    --include-package=psutil ^
    --nofollow-import-to=torch ^
    --nofollow-import-to=matplotlib ^
    --nofollow-import-to=IPython ^
    --nofollow-import-to=pytest ^
    --output-dir="%OUTPUT_DIR%" ^
    --remove-output ^
    desktop.py

echo.
echo ========================================================
echo  [SUCCESS] Desktop Compilation Complete!
echo  Output Location: %OUTPUT_DIR%
echo  Executable located in: %OUTPUT_DIR%\desktop.dist\desktop.exe
echo ========================================================
