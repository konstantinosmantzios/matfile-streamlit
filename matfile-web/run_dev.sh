#!/bin/bash

echo "========================================="
echo " Starting Matfile Development Servers... "
echo "========================================="

# Trap SIGINT and SIGTERM so that when you press Ctrl+C, 
# it kills both background processes properly.
trap 'kill 0' SIGINT SIGTERM EXIT

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
