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

## 3. Final Standalone Compilation (Nuitka)

When the application is ready for distribution without sharing source code or requiring users to install Python and Node.js, you can compile the entire backend and frontend into a standalone native application using **Nuitka**.

Nuitka compiles Python code into native C machine code (using your system's C compiler like Apple Clang or GCC), creating a high-performance, tamper-resistant standalone binary.

---

### Option A: Automated Build Script (Recommended)

From the `matfile-web` folder, run:

```bash
./build_desktop.sh
```

This automated script will:
1. Compile the frontend into static assets (`frontend/dist`) via `npm run build`.
2. Verify that `nuitka` is installed (installing it via `pip` if needed).
3. Compile `desktop.py` into a native standalone application inside `matfile-web/dist_desktop/`:
   - On **macOS**: Creates a self-contained `.app` bundle (`desktop.app` / `Matfile Viewer.app`) that users can double-click or drag into `/Applications`.
   - On **Windows/Linux**: Creates a standalone directory containing the executable and all bundled shared libraries.

---

### Option B: Manual Step-by-Step Compilation

#### 1. Prerequisites
Ensure you have a working C compiler installed:
- **macOS**: Install Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```
- **Windows**: Install Microsoft Visual C++ Build Tools or MinGW64.
- **Linux**: Install GCC/Clang: `sudo apt install build-essential`.

Install Nuitka:
```bash
python3 -m pip install nuitka
```

#### 2. Build the Frontend
```bash
cd frontend
npm run build
cd ..
```

#### 3. Run the Nuitka Build Command

**On macOS (Creates a Native `.app` Bundle):**
```bash
cd backend
python3 -m nuitka \
    --standalone \
    --macos-create-app-bundle \
    --macos-app-name="Matfile Viewer" \
    --macos-app-icon=none \
    --include-data-dir=../frontend/dist=frontend_dist \
    --include-package=uvicorn \
    --include-package=fastapi \
    --include-package=starlette \
    --include-package=pydantic \
    --include-package=pyarrow \
    --include-package=h5py \
    --include-package=scipy \
    --include-package=pandas \
    --include-package=numpy \
    --include-package=xlsxwriter \
    --include-package=psutil \
    --output-dir=../dist_desktop \
    --remove-output \
    desktop.py
```

**On Windows / Linux (Standalone Executable):**
```bash
cd backend
python3 -m nuitka \
    --standalone \
    --include-data-dir=../frontend/dist=frontend_dist \
    --include-package=uvicorn \
    --include-package=fastapi \
    --include-package=starlette \
    --include-package=pydantic \
    --include-package=pyarrow \
    --include-package=h5py \
    --include-package=scipy \
    --include-package=pandas \
    --include-package=numpy \
    --include-package=xlsxwriter \
    --include-package=psutil \
    --output-dir=../dist_desktop \
    --remove-output \
    desktop.py
```

---

### Key Nuitka Flag Explanations

| Flag | Purpose |
| :--- | :--- |
| `--standalone` | Bundles the Python runtime and all required `.so` / `.dylib` / `.dll` shared libraries so end-users don't need Python installed. |
| `--macos-create-app-bundle` | Packages the output into a standard double-clickable macOS `.app` bundle. |
| `--include-data-dir=...=frontend_dist` | Packages the compiled Vite frontend (`index.html`, JS, CSS) directly into the app bundle. |
| `--include-package=...` | Forces complete bundling for dynamic frameworks like FastAPI, Uvicorn, SciPy, and PyArrow. |
| `--enable-plugin=pywebview` | Integrates native Cocoa/WebKit desktop window bindings. |
| `--remove-output` | Cleans up intermediate C build artifacts after the build succeeds. |

> [!TIP]
> **Why `--standalone` instead of `--onefile`?**
> Scientific computing packages (SciPy, Pandas, NumPy, PyArrow) are hundreds of megabytes in size. In `--onefile` mode, the entire archive must decompress into a temporary folder every time the user launches the application, causing a 10–20 second delay on startup. In `--standalone` / `.app` bundle mode, all binaries are already unpacked, launching in under one second.

