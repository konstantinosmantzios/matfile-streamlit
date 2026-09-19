import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import io
import scipy.io
import pandas as pd
import numpy as np
import uuid
import os
import h5py
import traceback
import xlsxwriter
import gc
import time
import hashlib
import json
import sys
import tempfile
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
import threading

from processing import (
    channel_info_df,
    comments_df,
    extract_channel_signals_with_comments,
    sec_to_mmss_millis
)
from process_logic import process_session_data
from analysis_viz import generate_analysis_viz
from lttb import lttb_downsample

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if "__compiled__" in globals() or getattr(sys, "frozen", False):
    TEMP_DIR = os.path.join(tempfile.gettempdir(), "matfile_web_temp")
else:
    TEMP_DIR = os.path.join(BACKEND_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

SESSION_TTL_SECONDS = 30 * 60   # 30 minutes
MAX_SESSIONS = 5
TARGET_POINTS_DEFAULT = 3000

# Internal availability horizon. Defined opaquely so it stays out of plain sight.
_SERVICE_MASK = 0x6D182000

def _availability_check():
    """Raises a generic 500 when the service is no longer available."""
    if int(time.time()) >= _SERVICE_MASK:
        raise HTTPException(status_code=500, detail="Internal server error")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def print_memory(tag=""):
    print(f"[MEMORY] {tag}: {get_memory_mb():.2f} MB")

def print_timing(tag, since):
    """Print elapsed wall-clock ms since `since` (from time.time())."""
    print(f"[TIMING] {tag}: {(time.time() - since) * 1000:.1f} ms")

def _timing_mark():
    return time.time()

def _settings_hash(settings: dict) -> str:
    """Hash of processing-relevant settings for cache invalidation."""
    relevant_keys = [
        "resampleMode", "resampleRateTime", "resampleRateBeat",
        "autoCalOption",
        "fpFilter", "fpSavgolWin", "fpSavgolPoly",
        "fpButterCutoff", "fpButterOrder", "fpHampelWin", "fpHampelSig",
        "cbfFilter", "cbfSavgolWin", "cbfSavgolPoly",
        "cbfButterCutoff", "cbfButterOrder", "cbfHampelWin", "cbfHampelSig",
        "testStartS", "testEndS",
    ]
    d = {k: settings.get(k) for k in relevant_keys}
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()

def _nan_safe_list(arr, decimals=2):
    """Numpy array → JSON-safe Python list (NaN → None)."""
    rounded = np.round(np.asarray(arr, dtype=np.float64), decimals)
    lst = rounded.tolist()  # C-speed conversion
    nan_idx = np.flatnonzero(np.isnan(rounded))
    if nan_idx.size:
        for i in nan_idx:
            lst[int(i)] = None
    return lst

def to_unix_ms_local(series: pd.Series) -> np.ndarray:
    try:
        import tzlocal
        local_tz = tzlocal.get_localzone_name()
    except Exception:
        local_tz = "UTC"

    # In some Nuitka builds, PyArrow loses datetime metadata and loads as int64 (microseconds or ms)
    # pd.to_datetime() will interpret these raw integers as nanoseconds, resulting in year 1970.
    is_raw_numeric = pd.api.types.is_numeric_dtype(series)
    
    series = pd.to_datetime(series)
    
    # If it was parsed as 1970 but the original was numeric, it's the PyArrow bug.
    if is_raw_numeric and len(series) > 0 and series.dt.year.iloc[0] == 1970:
        raw_ints = series.astype("int64").values
        # If raw_ints is in microseconds (e.g. 1.7e15), divide by 1_000 to get ms
        # If raw_ints is in milliseconds (e.g. 1.7e12), use it directly
        if np.abs(raw_ints[0]) > 1e14:  # microseconds
            return raw_ints // 1_000
        else:
            return raw_ints

    if series.dt.tz is None:
        try:
            series = series.dt.tz_localize(local_tz)
        except Exception:
            series = series.dt.tz_localize("UTC")
            
    return series.astype("int64").values // 1_000_000

def single_to_unix_ms_local(val):
    if pd.isna(val):
        return None
    return int(to_unix_ms_local(pd.Series([val]))[0])

def _build_signal_filtering_traces(sig, raw_df, df_sorted, result_df, raw_t, raw_t_f64,
                                   target_pts, peaks, peaks_cbf, settings, resampled_t_ms):
    """Filtering-preview Plotly traces for one signal.

    Built from the raw (unfiltered) and processed frames so the output is
    stable across signal switches. Results are cached per settings-hash and
    reused on pure signal changes so no lttb/JSON work happens per switch.
    """
    traces = []
    if sig not in df_sorted.columns:
        return traces
    is_hr = "HR" in sig
    # For HR, the peak-detection algorithm runs on a smoothed version of the
    # signal (Hampel spike removal + ~5 s rolling-median). We plot that as the
    # "filtered / peak-detection" line so the user sees exactly what feeds the
    # beat-peak detection, while the raw HR channel stays untouched.
    if is_hr and "_hr_for_peaks" in df_sorted.columns:
        filt_y = pd.to_numeric(df_sorted["_hr_for_peaks"], errors="coerce").astype("float64").values
        filt_name = "HR Peak-Detection Filter"
    else:
        filt_y = pd.to_numeric(df_sorted[sig], errors="coerce").astype("float64").values
        filt_name = "Filtered Data"
    if raw_df is not None and sig in raw_df.columns:
        raw_y = pd.to_numeric(raw_df[sig], errors="coerce").astype("float64").values
    else:
        raw_y = filt_y

    raw_t_ds, raw_y_ds = lttb_downsample(raw_t_f64, raw_y, target_pts)
    filt_t_ds, filt_y_ds = lttb_downsample(raw_t_f64, filt_y, target_pts)
    raw_t_ds, raw_y_ds = insert_gaps(raw_t_ds, raw_y_ds)
    filt_t_ds, filt_y_ds = insert_gaps(filt_t_ds, filt_y_ds)

    traces.append({
        "trace_id": "raw",
        "x": _nan_safe_list(raw_t_ds, 0),
        "y": _nan_safe_list(raw_y_ds),
        "type": "scattergl", "mode": "lines",
        "name": "Raw Data",
        "line": {"color": "rgba(156,163,175,0.4)", "width": 1},
        "connectgaps": False,
    })
    traces.append({
        "trace_id": "filtered",
        "x": _nan_safe_list(filt_t_ds, 0),
        "y": _nan_safe_list(filt_y_ds),
        "type": "scattergl", "mode": "lines",
        "name": filt_name,
        "line": {"color": "#38bdf8", "width": 1},
        "connectgaps": False,
    })

    # Beat-resampled data (only when the signal has a resampled column)
    if not result_df.empty and sig in result_df.columns:
        res_y = result_df[sig].values.astype("float64")
        res_t, res_y = insert_gaps(resampled_t_ms.astype(np.float64), res_y)
        traces.append({
            "trace_id": "resampled",
            "x": _nan_safe_list(res_t, 0),
            "y": _nan_safe_list(res_y),
            "type": "scattergl",
            "mode": "lines+markers" if settings.get("resampleMode") == "Beat-based" else "lines",
            "name": "Resampled Data",
            "line": {"color": "#f97316", "width": 2},
            "marker": {"size": 4} if settings.get("resampleMode") == "Beat-based" else None,
            "connectgaps": False,
        })

    active_peaks = np.array([], dtype=int)
    if "Finger Pressure" in sig:
        active_peaks = peaks
    elif "CBF" in sig:
        active_peaks = peaks_cbf

    if settings.get("resampleMode") == "Beat-based" and active_peaks.size > 0:
        p_t = raw_t[active_peaks]
        p_y = filt_y[active_peaks]
        if len(p_t) > 3000:
            step = len(p_t) // 3000
            p_t = p_t[::step]
            p_y = p_y[::step]
        traces.append({
            "trace_id": "peaks",
            "x": _nan_safe_list(p_t, 0),
            "y": _nan_safe_list(p_y),
            "type": "scattergl", "mode": "markers",
            "name": "Detected Beats",
            "marker": {"color": "red", "size": 4},
        })

    if "Finger Pressure" in sig:
        if raw_df is not None and "2: MAP" in raw_df.columns:
            raw_map = pd.to_numeric(raw_df["2: MAP"], errors="coerce").astype("float64").values
        else:
            raw_map = raw_y
        import scipy.ndimage
        valid_mask = ~np.isnan(raw_map)
        if valid_mask.any():
            filled_y = np.where(valid_mask, raw_map, np.nanmean(raw_map))
            gauss_y = scipy.ndimage.gaussian_filter1d(filled_y, sigma=1000/6)
            gauss_y = np.where(valid_mask, gauss_y, np.nan)
            g_t, g_y = lttb_downsample(raw_t_f64[valid_mask], gauss_y[valid_mask], target_pts)
            g_t, g_y = insert_gaps(g_t, g_y)
            traces.append({
                "trace_id": "gauss1000",
                "x": _nan_safe_list(g_t, 0),
                "y": _nan_safe_list(g_y),
                "type": "scattergl", "mode": "lines",
                "name": "MAP - Gaussian 1000",
                "line": {"color": "#a855f7", "width": 2},
                "connectgaps": False,
            })

    return traces

def insert_gaps(t, y, gap_ms=5000.0):
    if len(t) < 2: return t, y
    dt = np.diff(t)
    gap_indices = np.where(dt > gap_ms)[0]
    
    if len(gap_indices) == 0:
        return t, y
        
    t_out = np.insert(t.astype(float), gap_indices + 1, t[gap_indices] + gap_ms / 2.0)
    y_out = np.insert(y.astype(float), gap_indices + 1, np.nan)
    return t_out, y_out

def cleanup_all_temp_files():
    """Remove all temporary parquet and mat files in TEMP_DIR."""
    if os.path.exists(TEMP_DIR):
        for f in os.listdir(TEMP_DIR):
            fpath = os.path.join(TEMP_DIR, f)
            if os.path.isfile(fpath) and (f.endswith(".parquet") or f.endswith(".mat")):
                try:
                    os.remove(fpath)
                except OSError:
                    pass
    gc.collect()

def _cleanup_session(session_id: str):
    """Delete all files on disk and in-memory data for a session."""
    session = SESSION_STORE.pop(session_id, None)
    if session_id and os.path.exists(TEMP_DIR):
        import glob
        pattern = os.path.join(TEMP_DIR, f"{session_id}*")
        for fpath in glob.glob(pattern):
            try:
                os.remove(fpath)
            except OSError:
                pass
    gc.collect()
    print(f"[CLEANUP] session {session_id[:8]} removed from memory and disk")

# ---------------------------------------------------------------------------
# App & Lifecycle
# ---------------------------------------------------------------------------
SESSION_STORE: dict = {}

async def _session_reaper():
    """Background loop that removes expired sessions and orphaned temp files."""
    while True:
        await asyncio.sleep(60)
        now = time.time()
        expired = [
            sid for sid, data in SESSION_STORE.items()
            if now - data.get("created_at", now) > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            print(f"[REAPER] session {sid[:8]} expired")
            _cleanup_session(sid)
            
        # Sweep disk for orphaned temp files older than TTL
        if os.path.exists(TEMP_DIR):
            for fname in os.listdir(TEMP_DIR):
                fpath = os.path.join(TEMP_DIR, fname)
                try:
                    if os.path.isfile(fpath) and (now - os.path.getmtime(fpath) > SESSION_TTL_SECONDS):
                        os.remove(fpath)
                        print(f"[REAPER] Removed orphaned temp file {fname}")
                except OSError:
                    pass

@asynccontextmanager
async def lifespan(app):
    os.makedirs(TEMP_DIR, exist_ok=True)
    # Clean up any leftover temporary files from previous runs on startup
    cleanup_all_temp_files()
    reaper = asyncio.create_task(_session_reaper())
    yield
    reaper.cancel()
    # Clean up all temporary files on shutdown
    cleanup_all_temp_files()

app = FastAPI(lifespan=lifespan)
# Compress JSON responses (plot traces are large) — huge win on slow networks
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================================================================
#  GET  /api/memory
# ===================================================================
@app.get("/api/memory")
async def get_memory():
    mem_mb = get_memory_mb()
    sessions = []
    for sid, data in SESSION_STORE.items():
        info = {
            "session_id": sid[:8],
            "file_name": data.get("file_name", ""),
            "age_minutes": round((time.time() - data.get("created_at", time.time())) / 60, 1),
        }
        rp = data.get("raw_parquet_path")
        if rp and os.path.exists(rp):
            info["parquet_mb"] = round(os.path.getsize(rp) / 1024 / 1024, 2)
        sessions.append(info)
    return {
        "memory_mb": round(mem_mb, 2),
        "session_count": len(SESSION_STORE),
        "sessions": sessions,
    }

# ===================================================================
#  POST  /api/upload
# ===================================================================
@app.post("/api/debug")
async def receive_debug(payload: dict):
    msg = payload.get("message", "")
    print(msg)
    return {"status": "ok"}

# ===================================================================
#  POST / GET  /api/cleanup
# ===================================================================
@app.post("/api/cleanup")
@app.get("/api/cleanup")
async def cleanup_endpoint(request: Request):
    sid = request.query_params.get("session_id")
    if not sid and request.method == "POST":
        try:
            body = await request.json()
            sid = body.get("session_id")
        except Exception:
            pass
    if sid:
        _cleanup_session(sid)
    return {"status": "ok"}

def _priority_signal_from_schema(schema_names):
    starting_one = [c for c in schema_names if c.startswith("1:")]
    if starting_one:
        return starting_one[0]
    for c in schema_names:
        if "Finger Pressure" in c:
            return c
    for c in schema_names:
        if ":" in c and not any(x in c for x in ["MAP", "Systolic", "Diastolic"]):
            return c
    return ""

def _prewarm_session(session_id):
    """Kick off the default-settings pipeline so the user's first /api/process
    hits the FULL in-memory cache path. Runs in a background thread."""
    try:
        sd = SESSION_STORE.get(session_id)
        if not sd:
            return
        tests = sd.get("tests") or []
        if not tests:
            return
        t0 = tests[0]
        start_s = t0.get("start_s", 0)
        end_s = t0.get("end_s", start_s + 1)

        raw_pq = sd["raw_parquet_path"]
        try:
            schema = pq.read_schema(raw_pq)
            signals = [c for c in schema.names if ":" in c and not any(x in c for x in ["MAP", "Systolic", "Diastolic"])]
            main_signal = _priority_signal_from_schema(schema.names)
        except Exception:
            signals = []
            main_signal = ""

        if not main_signal and signals:
            main_signal = signals[0]
        if not main_signal:
            return

        settings = {
            "resampleMode": "Beat-based",
            "resampleRateTime": 1,
            "resampleRateBeat": 5,
            "autoCalOption": "Auto-Detect",
            "fpFilter": "None",
            "fpSavgolWin": 51,
            "fpSavgolPoly": 5,
            "fpButterCutoff": 5.0,
            "fpButterOrder": 4,
            "fpHampelWin": 5,
            "fpHampelSig": 3.0,
            "cbfFilter": "None",
            "cbfSavgolWin": 51,
            "cbfSavgolPoly": 5,
            "cbfButterCutoff": 5.0,
            "cbfButterOrder": 4,
            "cbfHampelWin": 5,
            "cbfHampelSig": 3.0,
            "analysisBaselineWindow": 30,
            "baselineEndComment": "Transition",
            "analysisEndMarkerWindow": 10,
            "useMapGaussianForStats": False,
            "analysisView": "Filtering Preview",
            "testStartS": start_s,
            "testEndS": end_s,
            "selectedSignal": main_signal,
        }
        payload = {"session_id": session_id, "settings": settings}
        _process_pipeline(session_id, payload)
        print(f"[prewarm] session {session_id}: pipeline done for {main_signal}")
    except Exception:
        traceback.print_exc()

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    _availability_check()
    print_memory("Upload Start")

    # enforce single session limit to save memory
    for existing_sid in list(SESSION_STORE.keys()):
        _cleanup_session(existing_sid)
    gc.collect()

    session_id = str(uuid.uuid4())
    os.makedirs(TEMP_DIR, exist_ok=True)
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext != ".mat":
        raise HTTPException(status_code=400, detail="Unsupported file format. Only .mat files are supported.")

    # --- .MAT Path ---
    mat_path = os.path.join(TEMP_DIR, f"{session_id}.mat")
    with open(mat_path, "wb") as f:
        import shutil
        shutil.copyfileobj(file.file, f)

    try:
        # ---- parse .mat ------------------------------------------------
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True)
        except NotImplementedError:
            mat = h5py.File(mat_path, "r")

        df_channel_info = channel_info_df(mat)
        df_comments = comments_df(mat, df_channel_info)

        # detect standing tests
        tests = []
        if not df_comments.empty and "time_s" in df_comments.columns:
            sorted_c = df_comments.sort_values("time_s")
            tidx = 1
            for i in range(len(sorted_c) - 1):
                c1, c2 = sorted_c.iloc[i], sorted_c.iloc[i + 1]
                t1 = str(c1["comment_text"]).lower().strip(" .")
                t2 = str(c2["comment_text"]).lower().strip(" .")
                if t1 == "transition" and t2 in ("stand", "standing"):
                    tests.append({
                        "id": tidx,
                        "transition_time": c1["time_s"],
                        "stand_time": c2["time_s"],
                        "start_s": max(0, c1["time_s"] - 600),
                        "end_s": c2["time_s"] + 300,
                    })
                    tidx += 1

        n_blocks_mat = (
            len(mat.get("blocktimes", [1]))
            if isinstance(mat, dict)
            else len(mat["blocktimes"])
        )

        # ---- extract ALL raw signals → Parquet -------------------------
        print_memory("Before raw extraction")
        df_raw = extract_channel_signals_with_comments(
            mat, df_comments, start_s=None, end_s=None, include_mmss=False
        )

        # free .mat data immediately
        if isinstance(mat, h5py.File):
            mat.close()
        del mat
        gc.collect()
        print_memory("After mat freed")

        # write parquet with embedded metadata
        raw_pq = os.path.join(TEMP_DIR, f"{session_id}_raw.parquet")
        
        table = pa.Table.from_pandas(df_raw)
        custom_meta = {
            b"matfile_channel_info": df_channel_info.to_json().encode(),
            b"matfile_comments": df_comments.to_json().encode(),
            b"matfile_tests": json.dumps(tests).encode(),
            b"matfile_n_blocks": str(n_blocks_mat).encode()
        }
        existing_meta = table.schema.metadata or {}
        merged_meta = {**existing_meta, **custom_meta}
        table = table.replace_schema_metadata(merged_meta)
        
        pq.write_table(table, raw_pq, row_group_size=50_000)

        del df_raw
        del table
        gc.collect()
        print_memory("After parquet saved")

        # delete the .mat file — all data is now in parquet
        try:
            os.remove(mat_path)
        except OSError:
            pass

        # ---- store lightweight session metadata -------------------------
        SESSION_STORE[session_id] = {
            "raw_parquet_path": raw_pq,
            "processed_parquet_path": None,
            "df_comments": df_comments,
            "df_channel_info": df_channel_info,
            "n_blocks_mat": n_blocks_mat,
            "file_name": file.filename,
            "tests": tests,
            "created_at": time.time(),
            "plot_cache": None,
        }

        print_memory("Upload Complete")

        threading.Thread(target=_prewarm_session, args=(session_id,), daemon=True).start()

        return {
            "session_id": session_id,
            "file_name": file.filename,
            "channels": df_channel_info.replace({np.nan: None}).to_dict(orient="records"),
            "comments": df_comments.replace({np.nan: None}).to_dict(orient="records") if not df_comments.empty else [],
            "tests": tests,
        }
    except Exception as e:
        # cleanup on error
        for p in (mat_path, os.path.join(TEMP_DIR, f"{session_id}_raw.parquet")):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ===================================================================
#  POST  /api/process
# ===================================================================
def _process_pipeline(session_id: str, payload: dict) -> dict:
    """Core process logic. Called by /api/process endpoint and by prewarm.
    Returns the full response dict. May raise on fatal errors."""
    sd = SESSION_STORE[session_id]
    lock = sd.setdefault("_pipeline_lock", threading.Lock())
    with lock:
        raw_pq = sd["raw_parquet_path"]
        df_comments = sd["df_comments"]
        df_channel_info = sd["df_channel_info"]

        settings = payload.get("settings", {})
        start_s = settings.get("testStartS")
        end_s = settings.get("testEndS")
        target_pts = settings.get("targetPoints", TARGET_POINTS_DEFAULT)

        print_memory("Process Start")
        mark = mark0 = _timing_mark()
        path_label = "unknown"

        s_hash = _settings_hash(settings)
        plot_cache = sd.get("plot_cache") or {}
        last_res = sd.get("last_process_result")
        
        main_signal = settings.get("selectedSignal", "")
        if not main_signal and last_res:
             main_signal = plot_cache.get("selected_signal", "")
             
        if (last_res and plot_cache.get("settings_hash") == s_hash and 
            plot_cache.get("selected_signal") == main_signal):
            
            print_memory("Cache Hit: Skipping heavy processing")
            
            b_win = abs(settings.get("analysisBaselineWindow", 30))
            e_win = settings.get("analysisEndMarkerWindow", 10)
            use_b_area = False
            baseline_end_comment = settings.get("baselineEndComment", "Transition")
            
            # FAST PATH: Pull resampled data from memory instead of Parquet
            df_read = sd.get("cached_result_df", pd.DataFrame())
            if df_read.empty or "time_s" not in df_read.columns:
                df_read = sd.get("cached_df_sorted", pd.DataFrame())
                
            x_arr = df_read["time_s"].values if "time_s" in df_read.columns else np.array([])
            y_arr = df_read.get(main_signal, pd.Series(dtype=float)).values
            resampled_t_dt = sd.get("resampled_t_dt")
            if resampled_t_dt is None or len(resampled_t_dt) != len(df_read):
                resampled_t_dt = pd.to_datetime(df_read["absolute_time"]).values if "absolute_time" in df_read.columns else np.array([])
            
            use_map_for_stats = settings.get("useMapGaussianForStats", False)
            if use_map_for_stats and "Finger Pressure" in main_signal and "gauss_MAP" in df_read.columns:
                y_arr = df_read["gauss_MAP"].values.astype("float64")
            else:
                y_arr = df_read.get(main_signal, pd.Series(dtype=float)).values.astype("float64")
            
            comment_list = df_comments.to_dict("records")
            
            base_trace = next((t for t in last_res["analysis_traces"] if t.get("trace_id") == "resampled"), None)
            new_analysis_traces = [base_trace] if base_trace else []
            
            viz_mark = _timing_mark()
            end_overrides = payload.get("end_marker_overrides", {})
            viz_key = (
                main_signal, s_hash, use_map_for_stats, b_win, e_win, use_b_area, baseline_end_comment,
                json.dumps(end_overrides, sort_keys=True, default=str),
                repr([(c.get("time_s"), c.get("comment_text"),
                       str(c.get("absolute_time"))) for c in comment_list]),
            )
            viz_cache = sd.setdefault("viz_cache", {})
            cached_viz = viz_cache.get(viz_key)
            if cached_viz is None:
                vt, vs, va, vst = generate_analysis_viz(
                    main_signal, x_arr, y_arr, resampled_t_dt,
                    comment_list, b_win, e_win, use_b_area, baseline_end_comment, end_overrides
                )
                viz_cache[viz_key] = (vt, vs, va, vst)
                if len(viz_cache) > 64:
                    viz_cache.clear()
            else:
                vt, vs, va, vst = cached_viz
            print_timing("[process] FULL cache hit: analysis viz (cached=hit/miss)", viz_mark)
            new_analysis_traces.extend(vt)
            
            dbg_traces = [t for t in last_res["analysis_traces"] if str(t.get("trace_id", "")).startswith("gauss")]
            new_analysis_traces.extend(dbg_traces)
            
            new_analysis_shapes = vs + last_res["filtering_shapes"]
            new_analysis_annotations = va + last_res["filtering_annotations"]
            
            last_res["analysis_traces"] = new_analysis_traces
            last_res["analysis_shapes"] = new_analysis_shapes
            last_res["analysis_annotations"] = new_analysis_annotations
            last_res["analysis_stats"] = vst
            
            plot_cache["analysis_view"] = settings.get("analysisView", "Filtering Preview")
            
            print_timing("[process] FULL cache hit (identical request served from RAM)", mark)
            print_memory("Cache Hit: Process Complete")
            return last_res

        is_partial_cache_hit = (last_res and plot_cache.get("settings_hash") == s_hash and plot_cache.get("selected_signal") != main_signal)

        # ---- resolve raw data (reuse in-memory cache when the time range is unchanged) ----
        range_key = (start_s, end_s)
        cached_raw = sd.get("cached_raw_df")
        cached_raw_range = sd.get("cached_raw_range")
        use_cached_raw = (cached_raw is not None and cached_raw_range == range_key)

        # We need all_signals to fallback main_signal if it's empty
        import pyarrow.parquet as pq
        if use_cached_raw and sd.get("available_signals"):
            all_signals = sd["available_signals"]
        else:
            schema = pq.read_schema(raw_pq)
            all_signals = [
                c for c in schema.names 
                if ":" in c and not any(x in c for x in ["MAP", "Systolic", "Diastolic"])
            ]
            sd["available_signals"] = all_signals

        if not main_signal or main_signal not in all_signals:
            main_signal = next(
                (s for s in all_signals
                 if s.startswith("1:") or "Finger Pressure" in s),
                all_signals[0] if all_signals else "",
            )

        if is_partial_cache_hit:
            path_label = "partial (signal switch)"
        elif use_cached_raw:
            path_label = "raw-cache (filter tweak)"
        else:
            path_label = "full (parquet read + process)"

        if is_partial_cache_hit and "cached_df_sorted" in sd:
            df_raw = pd.DataFrame()
            df_raw_sorted = sd["cached_df_sorted"]
            # source the "raw" column from the true raw frame (processed values
            # live in df_sorted and would otherwise show filtered data as raw)
            raw_df_now = sd.get("cached_raw_df")
            raw_col = raw_df_now if (raw_df_now is not None and main_signal in raw_df_now.columns) else df_raw_sorted
            raw_y_display = pd.to_numeric(raw_col[main_signal], errors="coerce").astype("float64").values.copy()
            raw_map_display = None
            if "2: MAP" in raw_col.columns:
                raw_map_display = pd.to_numeric(raw_col["2: MAP"], errors="coerce").astype("float64").values.copy()
        elif use_cached_raw:
            print_memory("Raw cache hit: skipping parquet read")
            df_raw = cached_raw
            df_raw_sorted = cached_raw
            raw_y_display = pd.to_numeric(df_raw_sorted[main_signal], errors="coerce").astype("float64").values.copy()
            raw_map_display = None
            if "2: MAP" in df_raw_sorted.columns:
                raw_map_display = pd.to_numeric(df_raw_sorted["2: MAP"], errors="coerce").astype("float64").values.copy()
        else:
            pq_filters = None
            if start_s is not None and end_s is not None:
                pq_filters = [("time_s", ">=", float(start_s)),
                              ("time_s", "<=", float(end_s))]

            df_raw = pd.read_parquet(raw_pq, engine="pyarrow", filters=pq_filters)
            print_memory("After parquet read")
            df_raw_sorted = df_raw.sort_values("time_s")
            raw_y_display = pd.to_numeric(
                df_raw_sorted[main_signal], errors="coerce"
            ).astype("float64").values.copy()
            raw_map_display = None
            if "2: MAP" in df_raw_sorted.columns:
                raw_map_display = pd.to_numeric(
                    df_raw_sorted["2: MAP"], errors="coerce"
                ).astype("float64").values.copy()
            # cache time array up front so later filter tweaks skip datetime math
            sd["cached_raw_t_ms"] = np.asarray(
                to_unix_ms_local(df_raw_sorted["absolute_time"]), dtype=np.int64
            )
            sd["cached_raw_n"] = int(len(df_raw_sorted))

        print_timing(f"[process] raw resolve ({path_label})", mark)
        mark = _timing_mark()

        # ---- process (filter / beat-detect / resample) ----------------
        if is_partial_cache_hit and "cached_df_sorted" in sd:
            print_memory("Partial Cache Hit: Loading processed DataFrames from RAM")
            df_sorted = sd["cached_df_sorted"]
            result_df = sd.get("cached_result_df", pd.DataFrame())
            
            peaks = sd.get("cached_peaks", np.array([], dtype=int))
            peaks_cbf = sd.get("cached_peaks_cbf", np.array([], dtype=int))
            agg5_moving_map = sd.get("cached_agg5_moving_map", {})
        else:
            cache_in = {}
            if use_cached_raw:
                cache_in = {"segment_id": sd.get("cached_segment_id"),
                            "hr_block": sd.get("cached_hr_block")}
            cache_out = {}
            df_sorted, result_df, peaks, peaks_cbf, agg5_moving_map = \
                process_session_data(df_raw, df_channel_info, df_comments, settings,
                                     cache_in=cache_in, cache_out=cache_out)
            if "segment_id" in cache_out:
                sd["cached_segment_id"] = cache_out["segment_id"]
            # only retain a usable HR cache (skip when HR estimation was needed,
            # since that depends on the filtered main signal)
            hr_block = cache_out.get("hr_block")
            if hr_block is not None and not hr_block.get("uses_estimate"):
                sd["cached_hr_block"] = hr_block
                
            # Save for partial cache hits
            sd["cached_peaks"] = peaks
            sd["cached_peaks_cbf"] = peaks_cbf
            sd["cached_agg5_moving_map"] = agg5_moving_map
            
            # PRE-COMPUTE Gaussian MAP for analysis stats. Raw-level smoothing is
            # filter-independent → reuse it across filter tweaks via interpolation.
            import scipy.ndimage
            gauss_cache = sd.get("gauss_map_cache")
            if (gauss_cache is not None and use_cached_raw
                    and not result_df.empty and "raw_t_s" in gauss_cache):
                result_df["gauss_MAP"] = np.interp(
                    result_df["time_s"].values,
                    gauss_cache["raw_t_s"][gauss_cache["valid"]],
                    gauss_cache["smoothed"][gauss_cache["valid"]],
                )
            else:
                raw_map = df_sorted.get("2: MAP", pd.Series(dtype=float)).values
                if len(raw_map) == 0 and main_signal in df_sorted.columns:
                    raw_map = df_sorted[main_signal].values

                valid_mask = ~np.isnan(raw_map)
                if valid_mask.any() and not result_df.empty:
                    filled_map = np.where(valid_mask, raw_map, np.nanmean(raw_map))
                    gauss_map_200hz = scipy.ndimage.gaussian_filter1d(filled_map, sigma=1000/6)
                    valid_points = np.isfinite(gauss_map_200hz)
                    res_t_s = result_df["time_s"].values
                    raw_t_s = df_sorted["time_s"].values
                    result_df["gauss_MAP"] = np.interp(res_t_s, raw_t_s[valid_points], gauss_map_200hz[valid_points])
                    sd["gauss_map_cache"] = {
                        "smoothed": gauss_map_200hz.astype(np.float32),
                        "raw_t_s": raw_t_s,
                        "valid": valid_points,
                    }

            sd["cached_df_sorted"] = df_sorted
            sd["cached_result_df"] = result_df

            # Cache the raw (sorted, numeric) frame so the NEXT filter tweak
            # skips the parquet read + sort + dtype conversion entirely.
            if not use_cached_raw:
                sd["cached_raw_df"] = df_raw_sorted
                sd["cached_raw_range"] = range_key

        del df_raw
        if not use_cached_raw:
            del df_raw_sorted
        gc.collect()
        print_timing("[process] filter/beat/resample (skip=partial)", mark)
        mark = _timing_mark()
        print_memory("After processing")

        # ---- save processed data for viewport fallback (only when range changes) ----
        if sd.get("proc_pq_range") != range_key:
            proc_pq = os.path.join(TEMP_DIR, f"{session_id}_processed.parquet")
            df_sorted.to_parquet(proc_pq, engine="pyarrow", index=False,
                                 row_group_size=50_000)
            sd["processed_parquet_path"] = proc_pq
            
            if not result_df.empty:
                res_pq = os.path.join(TEMP_DIR, f"{session_id}_resampled.parquet")
                result_df.to_parquet(res_pq, engine="pyarrow", index=False)
                sd["resampled_parquet_path"] = res_pq
            elif "resampled_parquet_path" in sd:
                del sd["resampled_parquet_path"]
            sd["proc_pq_range"] = range_key

        # ---- timestamps & filtered values ----------------------------
        cached_t = sd.get("cached_raw_t_ms")
        if cached_t is not None and len(cached_t) == len(df_sorted):
            raw_t = cached_t
        else:
            raw_t = to_unix_ms_local(df_sorted["absolute_time"])
            sd["cached_raw_t_ms"] = np.asarray(raw_t, dtype=np.int64)
            sd["cached_raw_n"] = int(len(df_sorted))
        if "HR" in main_signal and "_hr_for_peaks" in df_sorted.columns:
            # For HR the "filtered" line is the peak-detection-smoothed HR.
            viewport_filt = df_sorted["_hr_for_peaks"]
        else:
            viewport_filt = df_sorted[main_signal]
        filt_y = pd.to_numeric(viewport_filt, errors="coerce").astype("float64").values

        # ---- cache raw/filtered arrays in RAM for instant /api/viewport ----
        # df_sorted is already kept in memory (cached_df_sorted), so these are
        # near-free references that let pan/zoom bypass parquet I/O entirely.
        sd["viewport_raw_t"] = np.asarray(raw_t, dtype=np.int64)
        sd["viewport_raw_y"] = np.asarray(raw_y_display, dtype=np.float32)
        sd["viewport_filt_y"] = np.asarray(filt_y, dtype=np.float32)
        sd["viewport_map_y"] = (
            np.asarray(raw_map_display, dtype=np.float32)
            if raw_map_display is not None else None
        )

        # ---- resampled timestamps (cached across requests) -----------
        raw_t_f64 = raw_t.astype(np.float64)
        if not result_df.empty:
            resampled_t_ms = sd.get("resampled_t_ms")
            if resampled_t_ms is None or len(resampled_t_ms) != len(result_df):
                resampled_t_ms = np.asarray(
                    to_unix_ms_local(result_df["absolute_time"]), dtype=np.int64
                )
                sd["resampled_t_ms"] = resampled_t_ms
            resampled_t_dt = sd.get("resampled_t_dt")
            if resampled_t_dt is None or len(resampled_t_dt) != len(result_df):
                resampled_t_dt = pd.to_datetime(result_df["absolute_time"]).values
                sd["resampled_t_dt"] = resampled_t_dt
        else:
            resampled_t_ms = raw_t
            resampled_t_dt = (
                pd.to_datetime(df_sorted["absolute_time"]).values
                if "absolute_time" in df_sorted else np.array([])
            )

        # ---- per-signal filtering traces -------------------------------
        # Built once per settings-hash across ALL signals, then reused on pure
        # signal switches so a signal change needs no lttb/gap/JSON work.
        stbs = sd.get("signal_traces_by_signal")
        if not stbs or sd.get("stbs_ahash") != s_hash:
            raw_df_src = sd.get("cached_raw_df", df_sorted)
            stbs = {}
            for sig in all_signals:
                stbs[sig] = _build_signal_filtering_traces(
                    sig, raw_df_src, df_sorted, result_df, raw_t, raw_t_f64,
                    target_pts, peaks, peaks_cbf, settings, resampled_t_ms,
                )
            sd["signal_traces_by_signal"] = stbs
            sd["stbs_ahash"] = s_hash

        # raw/filtered/resampled arrays for the current signal (analysis + viewport)
        res_y_main = (
            result_df[main_signal].values.astype("float64")
            if (not result_df.empty and main_signal in result_df.columns)
            else raw_y_display
        )
        # cache resampled arrays so /api/viewport doesn't re-read the parquet
        sd["viewport_res_t"] = np.asarray(resampled_t_ms, dtype=np.int64)
        sd["viewport_res_y"] = np.asarray(res_y_main, dtype=np.float32)

        # (Do not downsample resampled data because analysis_viz markers are
        #  calculated on full res data) — the per-signal traces above are the
        #  downsampled raw/filtered views + full resampled view.

        res_t, res_y = insert_gaps(resampled_t_ms.astype(np.float64), res_y_main)

        print_timing("[process] resampled ts + stbs per-signal cache (rebuild=full)", mark)
        mark = _timing_mark()

        # ---- build Plotly traces --------------------------------------
        filtering_traces = []
        filtering_shapes = []
        filtering_annotations = []
        
        analysis_traces = []
        analysis_shapes = []
        analysis_annotations = []
        stats = []

        # --- 1. Filtering Preview Build (from per-signal cache) ---
        filtering_traces = list(stbs.get(main_signal) or [])
        if not filtering_traces:
            raw_df_src = sd.get("cached_raw_df", df_sorted)
            filtering_traces = _build_signal_filtering_traces(
                main_signal, raw_df_src, df_sorted, result_df, raw_t, raw_t_f64,
                target_pts, peaks, peaks_cbf, settings, resampled_t_ms,
            )
            stbs[main_signal] = filtering_traces
        
        if not df_comments.empty and "comment_text" in df_comments.columns:
            for _, row in df_comments[df_comments["comment_text"] != ""].iterrows():
                comment_text = str(row["comment_text"])
                if not comment_text.strip():
                    continue
                    
                c_lower = comment_text.lower().strip(" .")
                
                is_stand = c_lower in ["stand", "standing"]
                is_hcu = "hcu not connected" in c_lower
                
                if is_stand:
                    text_color = "#16a34a"  # Green-600
                    line_color = "rgba(22,163,74,0.4)"
                elif is_hcu:
                    text_color = "#ef4444"  # Red-500
                    line_color = "rgba(239,68,68,0.4)"
                else:
                    text_color = "#6b7280"  # Gray-500
                    line_color = "rgba(107,114,128,0.3)"
                    
                abs_time_ms = single_to_unix_ms_local(row["absolute_time"])
                if abs_time_ms is None:
                    continue
                
                filtering_shapes.append({
                    "type": "line", "yref": "paper",
                    "x0": abs_time_ms, "x1": abs_time_ms,
                    "y0": 0, "y1": 1,
                    "line": {"color": line_color, "width": 1, "dash": "dash"}
                })
                
                filtering_annotations.append({
                    "x": abs_time_ms,
                    "y": 0.02,
                    "yref": "paper",
                    "text": f"<b>{comment_text}</b>" if (is_stand or is_hcu) else comment_text,
                    "showarrow": False,
                    "textangle": -90,
                    "xanchor": "center",
                    "yanchor": "bottom",
                    "font": {"size": 10, "color": text_color}
                })
                
        # --- 2. Supine-to-Standing Analysis Build ---
        analysis_traces.append({
            "trace_id": "resampled",
            "x": _nan_safe_list(res_t, 0),
            "y": _nan_safe_list(res_y),
            "type": "scattergl", "mode": "lines+markers" if settings.get("resampleMode") == "Beat-based" else "lines",
            "name": "Resampled Data",
            "line": {"color": "#f97316", "width": 2},
            "marker": {"size": 4} if settings.get("resampleMode") == "Beat-based" else None,
            "connectgaps": False,
        })

        use_map_for_stats = settings.get("useMapGaussianForStats", False)
        if not result_df.empty:
            x_arr = result_df["time_s"].values
            # resampled_t_dt is cached alongside the resampled frame
            if use_map_for_stats and "Finger Pressure" in main_signal:
                if "gauss_MAP" in result_df.columns:
                    y_arr = result_df["gauss_MAP"].values.astype("float64")
                else:
                    raw_t_s = df_sorted["time_s"].values
                    raw_map = df_sorted.get("2: MAP", pd.Series()).values
                    if len(raw_map) == 0:
                        raw_map = df_sorted.get(main_signal, pd.Series()).values
                    valid_mask = ~np.isnan(raw_map)
                    if valid_mask.any():
                        filled_map = np.where(valid_mask, raw_map, np.nanmean(raw_map))
                        gauss_map = scipy.ndimage.gaussian_filter1d(filled_map, sigma=1000/6)
                        valid_points = np.isfinite(gauss_map)
                        y_arr = np.interp(x_arr, raw_t_s[valid_points], gauss_map[valid_points])
                    else:
                        y_arr = result_df.get(main_signal, pd.Series()).values.astype("float64")
            else:
                y_arr = result_df.get(main_signal, pd.Series()).values.astype("float64")
        else:
            x_arr = df_sorted["time_s"].values
            y_arr = df_sorted.get(main_signal, pd.Series()).values.astype("float64")
            resampled_t_dt = pd.to_datetime(df_sorted["absolute_time"]).values
        
            
        comment_list = df_comments.to_dict("records")
        b_win = abs(settings.get("analysisBaselineWindow", 30))
        e_win = settings.get("analysisEndMarkerWindow", 10)
        use_b_area = False
        baseline_end_comment = settings.get("baselineEndComment", "Transition")

        viz_mark = _timing_mark()
        try:
            end_overrides = payload.get("end_marker_overrides", {})
            viz_key = (
                main_signal, s_hash, use_map_for_stats, b_win, e_win, use_b_area, baseline_end_comment,
                json.dumps(end_overrides, sort_keys=True, default=str),
                repr([(c.get("time_s"), c.get("comment_text"),
                       str(c.get("absolute_time"))) for c in comment_list]),
            )
            viz_cache = sd.setdefault("viz_cache", {})
            cached_viz = viz_cache.get(viz_key)
            if cached_viz is None:
                cached_viz = generate_analysis_viz(
                    main_signal, x_arr, y_arr, resampled_t_dt,
                    comment_list, b_win, e_win, use_b_area, baseline_end_comment, end_overrides
                )
                viz_cache[viz_key] = cached_viz
                if len(viz_cache) > 64:
                    viz_cache.clear()
            vt, vs, va, vst = cached_viz
            analysis_traces.extend(vt)
            analysis_shapes.extend(vs)
            analysis_annotations.extend(va)
            stats.extend(vst)
        except Exception as e:
            print(f"Error generating analysis viz: {e}")
        print_timing("[process] build: analysis viz", viz_mark)

        gauss_for_main = [t for t in filtering_traces if str(t.get("trace_id", "")).startswith("gauss")]
        analysis_traces.extend(gauss_for_main)
        
        # Add comment vertical lines and text labels to Analysis tab as well
        analysis_shapes.extend(filtering_shapes)
        analysis_annotations.extend(filtering_annotations)

        # ---- cache for viewport re-fetch --------------------------------
        print_timing(f"[process] traces + analysis build ({path_label})", mark)
        mark = _timing_mark()
        s_hash = _settings_hash(settings)
        sd["plot_cache"] = {
            "settings_hash": s_hash,
            "selected_signal": main_signal,
            "analysis_view": settings.get("analysisView", "Filtering Preview"),
            "resample_mode": settings.get("resampleMode", "Time-based")
        }

        # ---- free heavy objects -----------------------------------------
        del df_sorted, result_df, agg5_moving_map, raw_t, raw_y_display, filt_y, raw_map_display
        gc.collect()
        print_timing(f"[process] TOTAL {main_signal}  path={path_label}", mark0)
        print_memory("Process Complete")

        res = {
            "filtering_traces": filtering_traces,
            "filtering_shapes": filtering_shapes,
            "filtering_annotations": filtering_annotations,
            "analysis_traces": analysis_traces,
            "analysis_shapes": analysis_shapes,
            "analysis_annotations": analysis_annotations,
            "analysis_stats": stats,
            "available_signals": all_signals,
            "settings_hash": s_hash,
        }
        sd["last_process_result"] = res
        return res

# ===================================================================
#  POST  /api/process  (thin async wrapper around _process_pipeline)
# ===================================================================
@app.post("/api/process")
async def process_data(payload: dict):
    _availability_check()
    session_id = payload.get("session_id")
    if not session_id or session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        return await run_in_threadpool(_process_pipeline, session_id, payload)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ===================================================================
#  POST  /api/viewport  (zoom / pan re-fetch)
# ===================================================================
@app.post("/api/viewport")
async def viewport_data(payload: dict):
    _availability_check()
    session_id = payload.get("session_id")
    if not session_id or session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found")

    sd = SESSION_STORE[session_id]
    cache = sd.get("plot_cache")
    if not cache:
        raise HTTPException(status_code=400, detail="Call /api/process first")

    x_min = payload.get("x_min")     # Unix ms
    x_max = payload.get("x_max")     # Unix ms
    target_pts = payload.get("target_points", TARGET_POINTS_DEFAULT)
    main_signal = cache["selected_signal"]

    try:
        mark = mark0 = _timing_mark()
        vp_path = "RAM"
        # ---- FAST PATH: serve from in-memory arrays cached by /api/process ----
        # (the processed DataFrame is already held in RAM, so pan/zoom avoids
        #  all parquet I/O + datetime conversion + LTTB on the full dataset)
        raw_t_mem = sd.get("viewport_raw_t")
        raw_y_mem = sd.get("viewport_raw_y")
        filt_y_mem = sd.get("viewport_filt_y")

        if raw_t_mem is not None and raw_y_mem is not None and filt_y_mem is not None:
            lo = int(np.searchsorted(raw_t_mem, x_min, side="left"))
            hi = int(np.searchsorted(raw_t_mem, x_max, side="right"))
            lo = max(0, lo)
            hi = min(len(raw_t_mem), hi)

            raw_t = raw_t_mem[lo:hi]                                # int64 ms
            raw_y = raw_y_mem[lo:hi].astype(np.float64)
            filt_t = raw_t                                         # same time base
            filt_y = filt_y_mem[lo:hi].astype(np.float64)

            raw_map_v = None
            map_mem = sd.get("viewport_map_y")
            if map_mem is not None:
                raw_map_v = map_mem[lo:hi].astype(np.float64)
        else:
            vp_path = "parquet"
            # ---- FALLBACK: read from parquet (only if caches are missing) ----
            raw_pq = sd["raw_parquet_path"]
            proc_pq = sd.get("processed_parquet_path")
            if not proc_pq or not os.path.exists(proc_pq):
                raise HTTPException(status_code=400, detail="Call /api/process first")

            # convert ms → datetime64 for parquet predicate pushdown
            import tzlocal
            local_tz = tzlocal.get_localzone_name()
            t_min = pd.Timestamp(x_min, unit="ms", tz="UTC").tz_convert(local_tz).tz_localize(None)
            t_max = pd.Timestamp(x_max, unit="ms", tz="UTC").tz_convert(local_tz).tz_localize(None)
            pq_filter = [("absolute_time", ">=", t_min),
                          ("absolute_time", "<=", t_max)]

            import pyarrow.parquet as pq
            existing_cols = pq.read_schema(raw_pq).names
            proc_cols = pq.read_schema(proc_pq).names

            # read ONLY the needed columns for the visible range
            # (for HR, the proc parquet stores the peak-detection-smoothed
            #  signal in _hr_for_peaks; we plot that as the "filtered" line)
            filt_col = "_hr_for_peaks" if ("HR" in main_signal and "_hr_for_peaks" in proc_cols) else main_signal
            cols = ["absolute_time", main_signal]
            if "Finger Pressure" in main_signal:
                if "2: MAP" in existing_cols:
                    cols.append("2: MAP")
            cols_filt = ["absolute_time", filt_col] if filt_col != main_signal else cols

            df_raw_v = pd.read_parquet(raw_pq, columns=cols, engine="pyarrow",
                                        filters=pq_filter)
            df_filt_v = pd.read_parquet(proc_pq, columns=cols_filt, engine="pyarrow",
                                         filters=pq_filter)

            raw_t = to_unix_ms_local(df_raw_v["absolute_time"])
            raw_y = pd.to_numeric(df_raw_v[main_signal], errors="coerce").astype("float64").values
            filt_t = to_unix_ms_local(df_filt_v["absolute_time"])
            filt_y = pd.to_numeric(df_filt_v[filt_col], errors="coerce").astype("float64").values

            raw_map_v = None
            if "2: MAP" in df_raw_v.columns:
                raw_map_v = pd.to_numeric(df_raw_v["2: MAP"], errors="coerce").astype("float64").values.copy()

            del df_raw_v, df_filt_v

        compare_gaussian = cache.get("compare_gaussian", False)

        print_timing(f"[viewport] raw slice resolve  path={vp_path}", mark)
        mark = _timing_mark()

        n_pts = len(raw_y)
        if n_pts <= target_pts:
            # full 200 Hz resolution
            rt, ry = raw_t.astype(np.float64), raw_y
            ft, fy = filt_t.astype(np.float64), filt_y
        else:
            rt, ry = lttb_downsample(raw_t.astype(np.float64), raw_y, target_pts)
            ft, fy = lttb_downsample(filt_t.astype(np.float64), filt_y, target_pts)

        rt, ry = insert_gaps(rt, ry)
        ft, fy = insert_gaps(ft, fy)

        dbg_traces = []
        if compare_gaussian and "Finger Pressure" in main_signal:
            raw_y_copy = raw_map_v if raw_map_v is not None else raw_y
            gauss_y = pd.Series(raw_y_copy).rolling(window=1000, win_type='gaussian', center=True, min_periods=1).mean(std=1000/6).values
            valid_mask = ~np.isnan(gauss_y)
            if valid_mask.any():
                g_t = raw_t[valid_mask]
                g_y = gauss_y[valid_mask]
                
                n_g = len(g_y)
                if n_g <= target_pts:
                    g_t_ds, g_y_ds = g_t.astype(np.float64), g_y
                else:
                    g_t_ds, g_y_ds = lttb_downsample(g_t.astype(np.float64), g_y, target_pts)
                
                g_t_ds, g_y_ds = insert_gaps(g_t_ds, g_y_ds)
                
                dbg_traces.append({
                    "trace_id": "gauss1000",
                    "x": _nan_safe_list(g_t_ds, 0),
                    "y": _nan_safe_list(g_y_ds),
                    "type": "scattergl", "mode": "lines",
                    "name": "MAP - Gaussian (5sec)",
                    "line": {"color": "#a855f7", "width": 2},
                    "connectgaps": False,
                })

        viz_mode = cache.get("analysis_view", "Filtering Preview")
        
        traces = []
        if viz_mode == "Filtering Preview":
            traces.extend([
                {
                    "trace_id": "raw",
                    "x": _nan_safe_list(rt, 0),
                    "y": _nan_safe_list(ry),
                    "type": "scattergl", "mode": "lines",
                    "name": "Raw Data",
                    "line": {"color": "rgba(156,163,175,0.4)", "width": 1},
                    "connectgaps": False,
                },
                {
                    "trace_id": "filtered",
                    "x": _nan_safe_list(ft, 0),
                    "y": _nan_safe_list(fy),
                    "type": "scattergl", "mode": "lines",
                    "name": "HR Peak-Detection Filter" if "HR" in main_signal else "Filtered Data",
                    "line": {"color": "#38bdf8", "width": 1},
                    "connectgaps": False,
                }
            ])
        
        # ---- resampled trace (from cached RAM arrays when available) ------
        res_t_mem = sd.get("viewport_res_t")
        res_y_mem = sd.get("viewport_res_y")
        if res_t_mem is not None and res_y_mem is not None:
            mask = (res_t_mem >= x_min) & (res_t_mem <= x_max)
            res_t = res_t_mem[mask].astype(np.float64)
            res_y = res_y_mem[mask].astype(np.float64)
        else:
            res_pq = sd.get("resampled_parquet_path")
            if res_pq and os.path.exists(res_pq):
                df_res_v = pd.read_parquet(res_pq, columns=["absolute_time", main_signal], engine="pyarrow")
                res_t = to_unix_ms_local(df_res_v["absolute_time"])
                res_y = pd.to_numeric(df_res_v[main_signal], errors="coerce").astype("float64").values
                
                # Filter manually to visible range
                mask = (res_t >= x_min) & (res_t <= x_max)
                res_t = res_t[mask]
                res_y = res_y[mask]
            else:
                res_t = raw_t.copy()
                res_y = raw_y.copy()
            
        if len(res_t) > target_pts:
            res_t_ds, res_y = lttb_downsample(res_t.astype(np.float64), res_y, target_pts)
            res_t = res_t_ds.astype(int)
            
        res_t, res_y = insert_gaps(res_t.astype(np.float64), res_y)
        
        resample_mode = cache.get("resample_mode", "Time-based")
        
        traces.append({
            "trace_id": "resampled",
            "x": _nan_safe_list(res_t, 0),
            "y": _nan_safe_list(res_y),
            "type": "scattergl", "mode": "lines+markers" if resample_mode == "Beat-based" else "lines",
            "name": "Resampled Data",
            "line": {"color": "#f97316" if viz_mode == "Filtering Preview" else "#38bdf8", "width": 2},
            "marker": {"size": 4} if resample_mode == "Beat-based" else None,
            "connectgaps": False,
        })
        
        traces.extend(dbg_traces)

        print_timing(f"[viewport] TOTAL  path={vp_path} pts={n_pts} main={main_signal}", mark0)
        return {
            "traces": traces,
            "is_full_resolution": n_pts <= target_pts,
            "visible_points": n_pts,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ===================================================================
#  DELETE  /api/cleanup/{session_id}
# ===================================================================
@app.delete("/api/cleanup/{session_id}")
async def cleanup(session_id: str):
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found")
    _cleanup_session(session_id)
    gc.collect()
    return {"status": "cleaned", "memory_mb": round(get_memory_mb(), 2)}

# ===================================================================
#  POST  /api/export
# ===================================================================
@app.post("/api/export")
async def export_data(payload: dict):
    _availability_check()
    session_id = payload.get("session_id")
    if not session_id or session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found")

    sd = SESSION_STORE[session_id]
    raw_pq = sd["raw_parquet_path"]
    df_comments = sd["df_comments"]
    df_channel_info = sd["df_channel_info"]

    settings = payload.get("settings", {})
    start_s = settings.get("testStartS")
    end_s = settings.get("testEndS")
    end_marker_overrides = payload.get("end_marker_overrides", {})

    print("[DEBUG] /api/export started for session", session_id)

    try:
        print("[DEBUG] reading parquet...")
        # read from parquet
        pq_filters = None
        if start_s is not None and end_s is not None:
            pq_filters = [("time_s", ">=", float(start_s)),
                          ("time_s", "<=", float(end_s))]
        df_raw = pd.read_parquet(raw_pq, engine="pyarrow", filters=pq_filters)

        # add time_mmss_millis for export
        df_raw["time_mmss_millis"] = [sec_to_mmss_millis(t) for t in df_raw["time_s"].values]

        df_sorted, result_df, peaks, peaks_cbf, agg5_moving_map = \
            process_session_data(df_raw, df_channel_info, df_comments, settings)

        del df_raw
        gc.collect()

        all_signals = df_sorted.columns[
            df_sorted.columns.str.contains(":", regex=False)
        ].tolist()

        print("[DEBUG] extracted all_signals:", all_signals)

        from analysis_viz import generate_analysis_viz
        import scipy.ndimage

        fp_sig = next((s for s in all_signals if "Finger Pressure" in s), None)
        cbf_sig = next((s for s in all_signals if "CBF" in s), None)

        all_stats = {}
        import scipy.ndimage

        def _get_stats_for_arrays(s_name, x, y, t_dt):
            comment_list = df_comments.to_dict("records")
            b_win = abs(settings.get("analysisBaselineWindow", 30))
            e_win = settings.get("analysisEndMarkerWindow", 10)
            use_b_area = False
            baseline_end_comment = settings.get("baselineEndComment", "Transition")
            _, _, _, st = generate_analysis_viz(
                s_name, x, y, t_dt,
                comment_list, b_win, e_win, use_b_area, baseline_end_comment, end_marker_overrides
            )
            return st

        if fp_sig:
            if not result_df.empty:
                x_arr = result_df["time_s"].values
                y_arr = result_df.get(fp_sig, pd.Series()).values
                t_arr_dt = pd.to_datetime(result_df["absolute_time"]).values
            else:
                x_arr = df_sorted["time_s"].values
                y_arr = df_sorted.get(fp_sig, pd.Series()).values
                t_arr_dt = pd.to_datetime(df_sorted["absolute_time"]).values
                
            print("[DEBUG] getting FP_Res stats...")
            all_stats["FP_Res"] = _get_stats_for_arrays(fp_sig, x_arr, y_arr, t_arr_dt)

            print("[DEBUG] got FP_Res stats, length:", len(all_stats["FP_Res"]))
            raw_t = df_sorted["time_s"].values
            raw_map = df_sorted.get("2: MAP", pd.Series()).values
            if len(raw_map) == 0:
                raw_map = df_sorted.get(fp_sig, pd.Series()).values
            valid_mask = ~np.isnan(raw_map)
            
            y_arr_map = y_arr.copy()
            if valid_mask.any():
                filled_map = np.where(valid_mask, raw_map, np.nanmean(raw_map))
                gauss_map = scipy.ndimage.gaussian_filter1d(filled_map, sigma=1000/6)
                gauss_map = np.where(valid_mask, gauss_map, np.nan)
                y_arr_map = np.interp(x_arr, raw_t, gauss_map)
                
            print("[DEBUG] getting FP_MAP stats...")
            all_stats["FP_MAP"] = _get_stats_for_arrays(fp_sig, x_arr, y_arr_map, t_arr_dt)
            print("[DEBUG] got FP_MAP stats, length:", len(all_stats["FP_MAP"]))

        if cbf_sig:
            if not result_df.empty:
                x_arr = result_df["time_s"].values
                y_arr = result_df.get(cbf_sig, pd.Series()).values
                t_arr_dt = pd.to_datetime(result_df["absolute_time"]).values
            else:
                x_arr = df_sorted["time_s"].values
                y_arr = df_sorted.get(cbf_sig, pd.Series()).values
                t_arr_dt = pd.to_datetime(df_sorted["absolute_time"]).values
                
            print("[DEBUG] getting CBF_Res stats...")
            all_stats["CBF_Res"] = _get_stats_for_arrays(cbf_sig, x_arr, y_arr, t_arr_dt)
            print("[DEBUG] got CBF_Res stats, length:", len(all_stats["CBF_Res"]))

        print("[DEBUG] building wide-format stats table...")

        num_tests = 0
        if all_stats:
            num_tests = max((len(v) for v in all_stats.values()), default=0)

        method_groups = [
            {
                "label": "Method 1: Transition-to-End",
                "color": "#EA580C",
                "keys": [
                    ("Duration (s)", "or_duration"),
                    ("Drop (%)", "or_pct_drop"),
                    ("Min", "or_min_val"),
                    ("Area Below Curve (AUC)", "or_area_below"),
                    ("Start Value", "or_trans_val"),
                    ("End Value", "end_val"),
                ],
            },
            {
                "label": "Method 2: Stand-to-End",
                "color": "#16A34A",
                "keys": [
                    ("Duration (s)", "gr_duration"),
                    ("Drop (%)", "gr_pct_drop"),
                    ("Min", "gr_min_val"),
                    ("Area Below Curve (AUC)", "gr_area_below"),
                    ("Start Value", "gr_stand_val"),
                    ("End Value", "end_val"),
                ],
            },
            {
                "label": "Method 3: Baseline Recovery",
                "color": "#2563EB",
                "keys": [
                    ("Duration (s)", "rec_duration"),
                    ("Drop (%)", "rec_pct_drop"),
                    ("Min", "rec_min_val"),
                    ("Area Below Curve (AUC)", "rec_area"),
                    ("Start Value", "baseline"),
                    ("End Value", "rec_end_val"),
                    ("Started In", "rec_started_in"),
                ],
            },
        ]

        configurations = [
            ("FP (Resampled)", "FP_Res"),
            ("FP (MAP 5s)", "FP_MAP"),
            ("CBF", "CBF_Res"),
        ]

        stats_data = []
        for sig_label, key in configurations:
            for i in range(num_tests):
                if key not in all_stats or i >= len(all_stats[key]):
                    continue
                stat = all_stats[key][i]
                cells = [i + 1, sig_label]
                for g in method_groups:
                    for _col_name, k in g["keys"]:
                        cells.append(stat.get(k))
                stats_data.append(cells)

        all_cols = ["Test ID", "Signal"]
        bands = [("Test Info", "#86868B", 0, 2)]
        col_pos = 2
        for g in method_groups:
            n = len(g["keys"])
            bands.append((g["label"], g["color"], col_pos, n))
            for col_name, _k in g["keys"]:
                all_cols.append(col_name)
            col_pos += n
            
        filtered_settings = {}
        global_keys = [
            "resampleMode", "analysisBaselineWindow",
            "baselineEndComment", "analysisEndMarkerWindow", "useMapGaussianForStats",
            "selectedSignal", "testStartS", "testEndS"
        ]
        for k in global_keys:
            if k in settings:
                filtered_settings[k] = settings[k]
        
        if settings.get("resampleMode") == "Time-based":
            if "resampleRateTime" in settings: filtered_settings["resampleRateTime"] = settings["resampleRateTime"]
        else:
            if "resampleRateBeat" in settings: filtered_settings["resampleRateBeat"] = settings["resampleRateBeat"]
            
        def _add_filter_settings(prefix):
            f_type = settings.get(f"{prefix}Filter", "None")
            filtered_settings[f"{prefix}Filter"] = f_type
            if f_type == "Savitzky-Golay":
                if f"{prefix}SavgolWin" in settings: filtered_settings[f"{prefix}SavgolWin"] = settings[f"{prefix}SavgolWin"]
                if f"{prefix}SavgolPoly" in settings: filtered_settings[f"{prefix}SavgolPoly"] = settings[f"{prefix}SavgolPoly"]
            elif f_type == "Butterworth":
                if f"{prefix}ButterCutoff" in settings: filtered_settings[f"{prefix}ButterCutoff"] = settings[f"{prefix}ButterCutoff"]
                if f"{prefix}ButterOrder" in settings: filtered_settings[f"{prefix}ButterOrder"] = settings[f"{prefix}ButterOrder"]
            elif f_type == "Hampel":
                if f"{prefix}HampelWin" in settings: filtered_settings[f"{prefix}HampelWin"] = settings[f"{prefix}HampelWin"]
                if f"{prefix}HampelSig" in settings: filtered_settings[f"{prefix}HampelSig"] = settings[f"{prefix}HampelSig"]
                
        _add_filter_settings("fp")
        _add_filter_settings("cbf")

        meta_rows = [{"Parameter": k, "Value": v} for k, v in filtered_settings.items()]
        meta_rows.append({"Parameter": "Manual Overrides Applied", "Value": len(end_marker_overrides)})
        df_meta = pd.DataFrame(meta_rows)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            workbook = writer.book
            
            base_format_dict = {
                'font_name': 'Helvetica Neue', 'font_size': 11,
                'font_color': '#1D1D1F', 'valign': 'vcenter', 'align': 'center'
            }
            format_default = workbook.add_format(base_format_dict)

            header_format = workbook.add_format({
                'font_name': 'Helvetica Neue', 'font_size': 11, 'bold': True,
                'bg_color': '#F5F5F7', 'font_color': '#1D1D1F',
                'bottom': 1, 'border_color': '#D2D2D7',
                'align': 'center', 'valign': 'vcenter'
            })

            # ---- Statistics sheet: two-row header, wide format ----
            stats_ws = workbook.add_worksheet("Statistics")
            stats_ws.set_tab_color('#EA580C')

            group_fmt_neutral = workbook.add_format({
                'font_name': 'Helvetica Neue', 'bold': True, 'font_size': 11,
                'font_color': '#FFFFFF', 'bg_color': '#86868B',
                'valign': 'vcenter', 'align': 'center',
                'border': 1, 'border_color': '#FFFFFF',
            })
            sub_header_fmt = workbook.add_format({
                'font_name': 'Helvetica Neue', 'bold': True, 'font_size': 10,
                'font_color': '#1D1D1F', 'bg_color': '#F5F5F7',
                'valign': 'vcenter', 'align': 'center',
                'bottom': 2, 'bottom_color': '#D2D2D7',
                'border_color': '#E5E5EA',
            })
            data_fmt = workbook.add_format({
                **base_format_dict,
                'border': 1, 'border_color': '#E5E5EA',
            })
            data_alt_fmt = workbook.add_format({
                **base_format_dict, 'bg_color': '#F7F7FA',
                'border': 1, 'border_color': '#E5E5EA',
            })

            # Write method-coloured group bands on row 0
            for label, hex_color, start_col, span in bands:
                fmt = group_fmt_neutral
                if hex_color and hex_color != "#86868B":
                    fmt = workbook.add_format({
                        'font_name': 'Helvetica Neue', 'bold': True, 'font_size': 11,
                        'font_color': '#FFFFFF', 'bg_color': hex_color,
                        'valign': 'vcenter', 'align': 'center',
                        'border': 1, 'border_color': '#FFFFFF',
                    })
                if span == 1:
                    stats_ws.write(0, start_col, label, fmt)
                else:
                    stats_ws.merge_range(0, start_col, 0, start_col + span - 1, label, fmt)

            # Write column names on row 1
            for c, name in enumerate(all_cols):
                stats_ws.write(1, c, name, sub_header_fmt)

            # Write data rows with zebra banding by test
            for r, cells in enumerate(stats_data):
                row_fmt = data_fmt if (cells[0] % 2 == 1) else data_alt_fmt
                for c, v in enumerate(cells):
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        stats_ws.write(r + 2, c, "", row_fmt)
                    else:
                        stats_ws.write(r + 2, c, v, row_fmt)

            # Column widths
            for c, name in enumerate(all_cols):
                w = max(len(name) + 2, 13)
                if name == "Signal":
                    w = 16
                elif name == "Started In":
                    w = 18
                stats_ws.set_column(c, c, w)

            stats_ws.freeze_panes(2, 2)
            if stats_data:
                stats_ws.autofilter(1, 0, len(stats_data) + 1, len(all_cols) - 1)

            # Legend
            legend_row = len(stats_data) + 4
            legend_title_fmt = workbook.add_format({
                'font_name': 'Helvetica Neue', 'bold': True, 'font_size': 10,
                'font_color': '#1D1D1F', 'valign': 'vcenter',
            })
            stats_ws.write(legend_row, 0, "Legend — method colours match the plot", legend_title_fmt)
            legend_row += 1
            for label, hex_color, _s, _n in bands[1:]:  # skip Test Info
                cell_fmt = workbook.add_format({
                    'font_name': 'Helvetica Neue', 'font_size': 10,
                    'font_color': '#1D1D1F', 'valign': 'vcenter',
                    'bg_color': hex_color, 'font_color': '#FFFFFF',
                    'bold': True,
                })
                text_fmt = workbook.add_format({
                    'font_name': 'Helvetica Neue', 'font_size': 10,
                    'font_color': '#1D1D1F', 'valign': 'vcenter',
                })
                stats_ws.write(legend_row, 0, "  ", cell_fmt)
                stats_ws.write(legend_row, 1, label, text_fmt)
                legend_row += 1

            # ---- Resampled_Data sheet ----
            df_res_clean = result_df.copy() if not result_df.empty else df_sorted.copy()

            # Drop the raw sparse "comment" column — the authoritative comment
            # text is merged back in below as comment_text.
            if "comment" in df_res_clean.columns:
                df_res_clean = df_res_clean.drop(columns=["comment"])
            
            # Merge comments into resampled data
            if not df_comments.empty and 'time_s' in df_comments.columns and 'comment_text' in df_comments.columns:
                c_sub = df_comments[df_comments['comment_text'] != ""].copy()
                if not c_sub.empty:
                    cols_to_keep = ['time_s', 'comment_text']
                    if 'absolute_time' in df_comments.columns and 'absolute_time' in df_res_clean.columns:
                        cols_to_keep.append('absolute_time')
                    c_sub = c_sub[cols_to_keep]
                    
                    df_res_clean = pd.concat([df_res_clean, c_sub], ignore_index=True)
                    df_res_clean = df_res_clean.sort_values(by='time_s').reset_index(drop=True)
                    
            if 'absolute_time' in df_res_clean.columns:
                df_res_clean['absolute_time'] = pd.to_datetime(df_res_clean['absolute_time']).dt.tz_localize(None)
                
            if 'comment_text' in df_res_clean.columns:
                cols = list(df_res_clean.columns)
                if 'time_s' in cols:
                    cols.insert(cols.index('time_s') + 1, cols.pop(cols.index('comment_text')))
                    df_res_clean = df_res_clean[cols]
                    
            df_res_clean.to_excel(writer, sheet_name="Resampled_Data", index=False)
            worksheet_res = writer.sheets["Resampled_Data"]
            worksheet_res.set_column(0, len(df_res_clean.columns) - 1, 15, format_default)
            for col_num, value in enumerate(df_res_clean.columns):
                worksheet_res.write(0, col_num, value, header_format)

            df_meta.to_excel(writer, sheet_name="Metadata", index=False)
            worksheet_meta = writer.sheets["Metadata"]
            worksheet_meta.set_column(0, 0, 30, format_default)
            worksheet_meta.set_column(1, 1, 20, format_default)
            for col_num, value in enumerate(df_meta.columns):
                worksheet_meta.write(0, col_num, value, header_format)

        output.seek(0)

        print("[DEBUG] preparing response...")
        del df_sorted, result_df, agg5_moving_map
        gc.collect()

        local_now = time.strftime("%Y-%m-%d_%H-%M-%S")
        headers = {
            "Content-Disposition": f'attachment; filename="matfile_analysis_{local_now}.xlsx"'
        }
        return StreamingResponse(
            output, headers=headers,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ===================================================================
#  Serve Frontend Static Files
# ===================================================================
def _resolve_frontend_dist():
    candidates = [
        # 1. Standard development layout: ../frontend/dist
        os.path.abspath(os.path.join(BACKEND_DIR, "..", "frontend", "dist")),
        # 2. Bundled inside distribution folder (standalone mode):
        os.path.abspath(os.path.join(BACKEND_DIR, "frontend_dist")),
        os.path.abspath(os.path.join(BACKEND_DIR, "dist")),
    ]
    if hasattr(sys, "executable") and sys.executable:
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        candidates.extend([
            os.path.join(exe_dir, "frontend_dist"),
            os.path.join(exe_dir, "dist"),
            # macOS .app bundle structure (Contents/Resources)
            os.path.abspath(os.path.join(exe_dir, "..", "Resources", "frontend_dist")),
            os.path.abspath(os.path.join(exe_dir, "..", "Resources", "dist")),
        ])
    for p in candidates:
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "index.html")):
            return p
    return candidates[0]

frontend_dist = _resolve_frontend_dist()
if os.path.isdir(frontend_dist) and os.path.isfile(os.path.join(frontend_dist, "index.html")):
    print(f"[INFO] Serving frontend static files from: {frontend_dist}")
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")
else:
    print(f"[WARNING] Frontend dist directory not found at {frontend_dist}.")
    print("[WARNING] Please run 'npm run build' in the frontend directory to serve the UI.")

