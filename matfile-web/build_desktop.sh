#!/bin/bash
set -e

echo "========================================================"
echo "    FAMELab Matfile Web App - Desktop Compilation       "
echo "========================================================"

# Determine base directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
BACKEND_DIR="$SCRIPT_DIR/backend"
OUTPUT_DIR="$SCRIPT_DIR/dist_desktop"

# 1. Build the Frontend Production Bundle
echo -e "\n[Step 1/3] Building frontend static bundle (Vite)..."
cd "$FRONTEND_DIR"
npm run build

if [ ! -d "$FRONTEND_DIR/dist" ] || [ ! -f "$FRONTEND_DIR/dist/index.html" ]; then
    echo "[Error] Frontend build failed: $FRONTEND_DIR/dist/index.html not found."
    exit 1
fi
echo "[Step 1/3] Frontend build successful!"

# 2. Check Nuitka Installation
echo -e "\n[Step 2/3] Checking Nuitka compiler..."
if ! python3 -m nuitka --version >/dev/null 2>&1; then
    echo "[Notice] Nuitka is not currently installed in the active Python environment."
    echo "         Installing nuitka and zstandard via pip..."
    python3 -m pip install nuitka zstandard
fi

echo "[Step 2/3] Using Nuitka: $(python3 -m nuitka --version 2>&1 | head -n 1)"

# 3. Detect Platform & Compile
echo -e "\n[Step 3/3] Compiling standalone desktop application with Nuitka..."
cd "$BACKEND_DIR"

mkdir -p "$OUTPUT_DIR"

OS_NAME="$(uname -s)"
EXTRA_FLAGS=()

if [ "$OS_NAME" = "Darwin" ]; then
    echo "[Platform] macOS detected: Creating native macOS .app bundle..."
    EXTRA_FLAGS+=(--macos-create-app-bundle --macos-app-name="Matfile Viewer" --macos-app-icon=none)
elif [ "$OS_NAME" = "Linux" ]; then
    echo "[Platform] Linux detected: Creating standalone binary distribution..."
else
    echo "[Platform] Windows / Other detected: Creating standalone distribution..."
fi

# Execute Nuitka compilation
python3 -m nuitka \
    --onefile \
    "${EXTRA_FLAGS[@]}" \
    --include-data-dir="$FRONTEND_DIR/dist=frontend_dist" \
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
    --nofollow-import-to=torch \
    --nofollow-import-to=matplotlib \
    --nofollow-import-to=IPython \
    --nofollow-import-to=pytest \
    --output-dir="$OUTPUT_DIR" \
    --remove-output \
    desktop.py

echo -e "\n========================================================"
echo " [SUCCESS] Desktop Compilation Complete!"
echo " Output Location: $OUTPUT_DIR"
if [ "$OS_NAME" = "Darwin" ]; then
    echo " macOS Application: $OUTPUT_DIR/desktop.app (or Matfile Viewer.app)"
    echo " You can test it by running:"
    echo "   open \"$OUTPUT_DIR/desktop.app\""
else
    echo " Executable located in: $OUTPUT_DIR/desktop.dist/"
fi
echo "========================================================"
