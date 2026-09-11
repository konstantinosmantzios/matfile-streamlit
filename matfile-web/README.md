# Matfile Web App

This project consists of a FastAPI backend (`backend/`) and a React/Vite frontend (`frontend/`). 

It is designed to be runnable both as a standard web application (for development) and as a local desktop application using PyWebView. Ultimately, it can be compiled into a single executable to hide the source code using Nuitka.

---

## 1. Development & Testing Workflow

When actively developing the application, you want hot-reloading for the frontend so that changes appear instantly in the browser. You can run both the frontend and backend servers at the same time using the provided bash script:

```bash
./run_dev.sh
```
*(This starts both the FastAPI backend on port 8000 and the Vite frontend on port 5173. Press `Ctrl+C` to stop both).*

If you prefer to run them separately:
- **Backend:** `cd backend && python3 -m uvicorn main:app --reload`
- **Frontend:** `cd frontend && npm run dev`


---

## 2. Desktop View Workflow (Production Preview)

When you are finished making changes and want to test how the app feels as a native "Desktop" application (and ensure the built frontend works correctly with FastAPI):

**Step 1: Build the Frontend**
You must compile the frontend code into static files so the backend can serve them.
```bash
cd frontend
npm run build
```

**Step 2: Run the Desktop App**
Start the application using the `desktop.py` script. This will start the FastAPI server in the background and open a PyWebView native window.
```bash
cd backend
python3 desktop.py
```
*(A standalone application window will open automatically, displaying your app).*

---

## 3. Final Compilation (Future Step)

When the application is 100% complete and you are ready to distribute it to others without sharing the source code, you will use **Nuitka** to compile the `desktop.py` and `main.py` (along with the frontend `dist` folder) into a standalone binary file. 

*Details for the Nuitka build command will be added here when the app is ready for distribution.*
