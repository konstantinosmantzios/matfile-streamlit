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
if errorlevel 1 (
    echo [Error] pip install failed.
    exit /b 1
)

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: Execute Nuitka compilation for Windows
python -m nuitka ^
    --standalone ^
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
