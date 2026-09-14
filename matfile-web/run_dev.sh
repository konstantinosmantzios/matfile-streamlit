#!/bin/bash

echo "========================================="
echo " Starting Matfile Development Servers... "
echo "========================================="

# Trap SIGINT, SIGTERM, and EXIT to clean up temp files and kill child processes
cleanup() {
    echo -e "\n[System] Stopping servers and cleaning temporary session files..."
    rm -f backend/temp/*.parquet backend/temp/*.mat 2>/dev/null || true
    kill 0 2>/dev/null || true
}
trap cleanup SIGINT SIGTERM EXIT

# Ensure port 8000 is free before starting
echo "[System] Freeing port 8000..."
lsof -t -i:8000 | xargs kill -9 2>/dev/null || true

# Start FastAPI backend in the background
echo "[Backend] Starting FastAPI on http://localhost:8000"
cd backend
python3 -m uvicorn main:app --reload &
cd ..

# Give the backend a second to start
sleep 1

# Start Vite frontend in the background
echo "[Frontend] Starting Vite server..."
cd frontend
npm run dev &
cd ..

echo "========================================="
echo " Both servers are running!"
echo " Press Ctrl+C to stop both."
echo "========================================="

# Wait indefinitely until the user presses Ctrl+C
wait
