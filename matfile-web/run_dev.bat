@echo off
setlocal

echo =========================================
echo  Starting Matfile Development Servers... 
echo =========================================

echo [System] Cleaning temporary session files...
if exist backend\temp\*.parquet del /q backend\temp\*.parquet 2>nul
if exist backend\temp\*.mat del /q backend\temp\*.mat 2>nul

echo [System] Freeing port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000 " ^| findstr "LISTENING"') do taskkill /f /pid %%a 2>nul

echo [Backend] Starting FastAPI on http://localhost:8000
start "Matfile Backend" cmd /k "cd backend && python -m uvicorn main:app --reload"

:: Give the backend a second to start
timeout /t 1 /nobreak >nul

echo [Frontend] Starting Vite server...
start "Matfile Frontend" cmd /k "cd frontend && npm run dev"

echo =========================================
echo  Both servers are running in new windows!
echo  Close those windows to stop the servers.
echo =========================================
