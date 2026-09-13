from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
import asyncio
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTasks

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
TEMP_DIR = "temp"
SESSION_TTL_SECONDS = 30 * 60   # 30 minutes
MAX_SESSIONS = 5
TARGET_POINTS_DEFAULT = 3000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def print_memory(tag=""):
    print(f"[MEMORY] {tag}: {get_memory_mb():.2f} MB")

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
    rounded = np.round(arr, decimals)
    return [None if np.isnan(v) else float(v) for v in rounded]

def to_unix_ms_local(series: pd.Series) -> np.ndarray:
    import tzlocal
    local_tz = tzlocal.get_localzone_name()
    # Force to datetime in case it's an object/string type
    series = pd.to_datetime(series)
    if series.dt.tz is None:
        series = series.dt.tz_localize(local_tz)
    return series.astype("int64").values // 1_000_000

def single_to_unix_ms_local(val):
    if pd.isna(val):
        return None
    return int(to_unix_ms_local(pd.to_datetime(pd.Series([val])))[0])

def insert_gaps(t, y, gap_ms=5000.0):
    if len(t) < 2: return t, y
    dt = np.diff(t)
    gap_indices = np.where(dt > gap_ms)[0]
    
    if len(gap_indices) == 0:
        return t, y
        
    t_out = np.insert(t.astype(float), gap_indices + 1, t[gap_indices] + gap_ms / 2.0)
    y_out = np.insert(y.astype(float), gap_indices + 1, np.nan)
    return t_out, y_out

def _cleanup_session(session_id: str):
    """Delete all files and in-memory data for a session."""
    session = SESSION_STORE.pop(session_id, None)
    if session is None:
        return
    for key in ("raw_parquet_path", "processed_parquet_path", "file_path"):
        path = session.get(key)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    gc.collect()
    print(f"[CLEANUP] session {session_id[:8]} removed")

# ---------------------------------------------------------------------------
# App & Lifecycle
# ---------------------------------------------------------------------------
SESSION_STORE: dict = {}

async def _session_reaper():
    """Background loop that removes expired sessions."""
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

@asynccontextmanager
async def lifespan(app):
    os.makedirs(TEMP_DIR, exist_ok=True)
    reaper = asyncio.create_task(_session_reaper())
    yield
    reaper.cancel()
    for sid in list(SESSION_STORE.keys()):
        _cleanup_session(sid)

app = FastAPI(lifespan=lifespan)
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

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    print_memory("Upload Start")

    # enforce single session limit to save memory
    for existing_sid in list(SESSION_STORE.keys()):
        _cleanup_session(existing_sid)
    gc.collect()

    session_id = str(uuid.uuid4())
    os.makedirs(TEMP_DIR, exist_ok=True)
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext == ".parquet":
        raw_pq = os.path.join(TEMP_DIR, f"{session_id}_raw.parquet")
        with open(raw_pq, "wb") as f:
            import shutil
            shutil.copyfileobj(file.file, f)
            
        # read embedded metadata
        schema = pq.read_schema(raw_pq)
        meta = schema.metadata or {}
        
        df_channel_info = pd.read_json(io.StringIO(meta.get(b"matfile_channel_info", b"{}").decode()))
        df_comments = pd.read_json(io.StringIO(meta.get(b"matfile_comments", b"{}").decode()))
        tests = json.loads(meta.get(b"matfile_tests", b"[]").decode())
        n_blocks_mat = int(meta.get(b"matfile_n_blocks", b"1").decode())
        
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
        return {"session_id": session_id, "file_name": file.filename, "tests": tests}
        
    # --- .MAT Fallback Path ---
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
#  POST  /api/convert
# ===================================================================
@app.post("/api/convert")
async def convert_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    print_memory("Convert Start")
    session_id = str(uuid.uuid4())
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    mat_path = os.path.join(TEMP_DIR, f"{session_id}.mat")
    raw_pq = os.path.join(TEMP_DIR, f"{session_id}_raw.parquet")
    
    def cleanup_temp_files():
        try:
            if os.path.exists(mat_path):
                os.remove(mat_path)
            if os.path.exists(raw_pq):
                os.remove(raw_pq)
        except Exception:
            pass
            
    background_tasks.add_task(cleanup_temp_files)
    
    # save uploaded file to disk
    with open(mat_path, "wb") as f:
        import shutil
        shutil.copyfileobj(file.file, f)

    try:
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True)
        except NotImplementedError:
            mat = h5py.File(mat_path, "r")

        df_channel_info = channel_info_df(mat)
        df_comments = comments_df(mat, df_channel_info)

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

        df_raw = extract_channel_signals_with_comments(
            mat, df_comments, start_s=None, end_s=None, include_mmss=False
        )

        if isinstance(mat, h5py.File):
            mat.close()
        del mat
        gc.collect()

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
        print_memory("Convert Done")
        
        download_name = os.path.splitext(file.filename)[0] + ".parquet"
        return FileResponse(raw_pq, media_type="application/octet-stream", filename=download_name)
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to convert file: {str(e)}")

# ===================================================================
#  POST  /api/process
# ===================================================================
@app.post("/api/process")
async def process_data(payload: dict):
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
    target_pts = settings.get("targetPoints", TARGET_POINTS_DEFAULT)

    try:
        print_memory("Process Start")

        s_hash = _settings_hash(settings)
        plot_cache = sd.get("plot_cache", {})
        last_res = sd.get("last_process_result")
        
        main_signal = settings.get("selectedSignal", "")
        if not main_signal and last_res:
             main_signal = plot_cache.get("selected_signal", "")
             
        if (last_res and plot_cache.get("settings_hash") == s_hash and 
            plot_cache.get("selected_signal") == main_signal):
            
            print_memory("Cache Hit: Skipping heavy processing")
            
            b_win = abs(settings.get("analysisBaselineWindow", 30))
            e_win = settings.get("analysisEndMarkerWindow", 10)
            use_b_area = settings.get("useBaselineArea", False)
            baseline_end_comment = settings.get("baselineEndComment", "Transition")
            
            # FAST PATH: Pull resampled data from memory instead of Parquet
            df_read = sd.get("cached_result_df", pd.DataFrame())
            if df_read.empty or "time_s" not in df_read.columns:
                df_read = sd.get("cached_df_sorted", pd.DataFrame())
                
            x_arr = df_read["time_s"].values if "time_s" in df_read.columns else np.array([])
            y_arr = df_read.get(main_signal, pd.Series(dtype=float)).values
            resampled_t_dt = pd.to_datetime(df_read["absolute_time"]).values if "absolute_time" in df_read.columns else np.array([])
            
            use_map_for_stats = settings.get("useMapGaussianForStats", False)
            if use_map_for_stats and "Finger Pressure" in main_signal and "gauss_MAP" in df_read.columns:
                y_arr = df_read["gauss_MAP"].values.astype("float64")
            else:
                y_arr = df_read.get(main_signal, pd.Series(dtype=float)).values.astype("float64")
            
            comment_list = df_comments.to_dict("records")
            
            base_trace = next((t for t in last_res["analysis_traces"] if t.get("trace_id") == "resampled"), None)
            new_analysis_traces = [base_trace] if base_trace else []
            
            end_overrides = payload.get("end_marker_overrides", {})
            vt, vs, va, vst = generate_analysis_viz(
                main_signal, x_arr, y_arr, resampled_t_dt,
                comment_list, b_win, e_win, use_b_area, baseline_end_comment, end_overrides
            )
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
            
            print_memory("Cache Hit: Process Complete")
            return last_res

        is_partial_cache_hit = (last_res and plot_cache.get("settings_hash") == s_hash and plot_cache.get("selected_signal") != main_signal)

        # ---- read raw data from parquet (with optional time filter) -----
        pq_filters = None
        if start_s is not None and end_s is not None:
            pq_filters = [("time_s", ">=", float(start_s)),
                          ("time_s", "<=", float(end_s))]
                          
        # We need all_signals to fallback main_signal if it's empty
        import pyarrow.parquet as pq
        schema = pq.read_schema(raw_pq)
        all_signals = [
            c for c in schema.names 
            if ":" in c and not any(x in c for x in ["MAP", "Systolic", "Diastolic"])
        ]
        if not main_signal or main_signal not in all_signals:
            main_signal = next(
                (s for s in all_signals
                 if s.startswith("1:") or "Finger Pressure" in s),
                all_signals[0] if all_signals else "",
            )

        if is_partial_cache_hit and "cached_df_sorted" in sd:
            df_raw = pd.DataFrame()
            df_raw_sorted = sd["cached_df_sorted"]
            raw_y_display = pd.to_numeric(df_raw_sorted[main_signal], errors="coerce").astype("float64").values.copy()
            raw_map_display = None
            if "2: MAP" in df_raw_sorted.columns:
                raw_map_display = pd.to_numeric(df_raw_sorted["2: MAP"], errors="coerce").astype("float64").values.copy()
        else:
            try:
                df_raw = pd.read_parquet(raw_pq, engine="pyarrow", filters=pq_filters)
            except Exception:
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

        # ---- process (filter / beat-detect / resample) ----------------
        if is_partial_cache_hit and "cached_df_sorted" in sd:
            print_memory("Partial Cache Hit: Loading processed DataFrames from RAM")
            df_sorted = sd["cached_df_sorted"]
            result_df = sd.get("cached_result_df", pd.DataFrame())
            
            peaks = sd.get("cached_peaks", np.array([], dtype=int))
            peaks_cbf = sd.get("cached_peaks_cbf", np.array([], dtype=int))
            agg5_moving_map = sd.get("cached_agg5_moving_map", {})
        else:
            df_sorted, result_df, peaks, peaks_cbf, agg5_moving_map = \
                process_session_data(df_raw, df_channel_info, df_comments, settings)
                
            # Save for partial cache hits
            sd["cached_peaks"] = peaks
            sd["cached_peaks_cbf"] = peaks_cbf
            sd["cached_agg5_moving_map"] = agg5_moving_map
            
            # PRE-COMPUTE Gaussian MAP for analysis stats
            import scipy.ndimage
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

            sd["cached_df_sorted"] = df_sorted
            sd["cached_result_df"] = result_df

        del df_raw, df_raw_sorted
        gc.collect()
        print_memory("After processing")

        # ---- save processed data for viewport -------------------------
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

        # ---- timestamps & filtered values ----------------------------
        raw_t = to_unix_ms_local(df_sorted["absolute_time"])
        filt_y = pd.to_numeric(
            df_sorted[main_signal], errors="coerce"
        ).astype("float64").values

        # ---- LTTB downsample for initial view -------------------------
        raw_t_f64 = raw_t.astype(np.float64)
        raw_t_ds, raw_y_ds = lttb_downsample(raw_t_f64, raw_y_display, target_pts)
        filt_t_ds, filt_y_ds = lttb_downsample(raw_t_f64, filt_y, target_pts)

        dbg_traces = []
        if "Finger Pressure" in main_signal:
            raw_y_copy = raw_map_display if raw_map_display is not None else raw_y_display
            
            # Use scipy's gaussian_filter1d for massive speedup over pandas rolling
            import scipy.ndimage
            valid_mask = ~np.isnan(raw_y_copy)
            if valid_mask.any():
                # Fill NaNs with mean to prevent NaN propagation in scipy filter
                filled_y = np.where(valid_mask, raw_y_copy, np.nanmean(raw_y_copy))
                gauss_y = scipy.ndimage.gaussian_filter1d(filled_y, sigma=1000/6)
                # Restore NaNs
                gauss_y = np.where(valid_mask, gauss_y, np.nan)
            if valid_mask.any():
                g_t = raw_t_f64[valid_mask]
                g_y = gauss_y[valid_mask]
                g_t_ds, g_y_ds = lttb_downsample(g_t, g_y, target_pts)
                g_t_ds, g_y_ds = insert_gaps(g_t_ds, g_y_ds)
                
                dbg_traces.append({
                    "trace_id": "gauss1000",
                    "x": _nan_safe_list(g_t_ds, 0),
                    "y": _nan_safe_list(g_y_ds),
                    "type": "scattergl", "mode": "lines",
                    "name": "MAP - Gaussian 1000",
                    "line": {"color": "#a855f7", "width": 2},
                    "connectgaps": False,
                })

        # ---- resampled data (already sparse → send in full) -----------
        if not result_df.empty:
            res_t_raw = to_unix_ms_local(result_df["absolute_time"])
            res_y = result_df[main_signal].values.astype("float64")
        else:
            res_t_raw = raw_t
            res_y = raw_y_display

        # (Do not downsample resampled data because analysis_viz markers are calculated on full res data)

        raw_t_ds, raw_y_ds = insert_gaps(raw_t_ds, raw_y_ds)
        filt_t_ds, filt_y_ds = insert_gaps(filt_t_ds, filt_y_ds)
        res_t, res_y = insert_gaps(res_t_raw.astype(np.float64), res_y)

        # ---- build Plotly traces --------------------------------------
        filtering_traces = []
        filtering_shapes = []
        filtering_annotations = []
        
        analysis_traces = []
        analysis_shapes = []
        analysis_annotations = []
        stats = []

        priority_signal = next((s for s in all_signals if s.startswith("1:") or "Finger Pressure" in s), all_signals[0] if all_signals else "")
        fallback_signal = next((s for s in all_signals if s.startswith("6:") or "CBF" in s), None)
        is_hr_signal = "HR" in main_signal

        # --- 1. Filtering Preview Build ---
        if is_hr_signal:
            # For HR: only show the smoothed signal (stored in df_sorted after process_logic cleaned it).
            # The raw and resampled traces are meaningless for HR (not beat-resampled).
            hr_smoothed_y = pd.to_numeric(df_sorted[main_signal], errors="coerce").astype("float64").values
            hr_t_ds, hr_y_ds = lttb_downsample(raw_t_f64, hr_smoothed_y, target_pts)
            hr_t_ds, hr_y_ds = insert_gaps(hr_t_ds, hr_y_ds)
            filtering_traces.append({
                "trace_id": "filtered",
                "x": _nan_safe_list(hr_t_ds, 0),
                "y": _nan_safe_list(hr_y_ds),
                "type": "scattergl", "mode": "lines",
                "name": "HR (smoothed)",
                "line": {"color": "#f97316", "width": 2},
                "connectgaps": False,
            })
        else:
            filtering_traces.append({
                    "trace_id": "raw",
                    "x": _nan_safe_list(raw_t_ds, 0),
                    "y": _nan_safe_list(raw_y_ds),
                    "type": "scattergl", "mode": "lines",
                    "name": "Raw Data",
                    "line": {"color": "rgba(156,163,175,0.4)", "width": 1},
                    "connectgaps": False,
                })
            filtering_traces.append({
                "trace_id": "filtered",
                "x": _nan_safe_list(filt_t_ds, 0),
                "y": _nan_safe_list(filt_y_ds),
                "type": "scattergl", "mode": "lines",
                "name": "Filtered Data",
                "line": {"color": "#38bdf8", "width": 1},
                "connectgaps": False,
            })
            filtering_traces.append({
                    "trace_id": "resampled",
                    "x": _nan_safe_list(res_t, 0),
                    "y": _nan_safe_list(res_y),
                    "type": "scattergl", "mode": "lines+markers" if settings.get("resampleMode") == "Beat-based" else "lines",
                    "name": "Resampled Data",
                    "line": {"color": "#f97316", "width": 2},
                    "marker": {"size": 4} if settings.get("resampleMode") == "Beat-based" else None,
                    "connectgaps": False,
                })

        active_peaks = np.array([], dtype=int)
        if main_signal == priority_signal:
            active_peaks = peaks
        elif fallback_signal and main_signal == fallback_signal:
            active_peaks = peaks_cbf

        if not is_hr_signal and settings.get("resampleMode") == "Beat-based" and active_peaks.size > 0:
            p_t = raw_t[active_peaks]
            p_y = filt_y[active_peaks]
            if len(p_t) > 3000:
                step = len(p_t) // 3000
                p_t = p_t[::step]
                p_y = p_y[::step]
            filtering_traces.append({
                "trace_id": "peaks",
                "x": _nan_safe_list(p_t, 0),
                "y": _nan_safe_list(p_y),
                "type": "scattergl", "mode": "markers",
                "name": "Detected Beats",
                "marker": {"color": "red", "size": 4},
            })
        
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
        if is_hr_signal:
            # Use the same smoothed HR trace (already built above)
            analysis_traces.append({
                "trace_id": "filtered",
                "x": _nan_safe_list(hr_t_ds, 0),
                "y": _nan_safe_list(hr_y_ds),
                "type": "scattergl", "mode": "lines",
                "name": "HR (smoothed)",
                "line": {"color": "#f97316", "width": 2},
                "connectgaps": False,
            })
        else:
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
            resampled_t_dt = pd.to_datetime(result_df["absolute_time"]).values
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
        use_b_area = settings.get("useBaselineArea", False)
        baseline_end_comment = settings.get("baselineEndComment", "Transition")

        try:
            end_overrides = payload.get("end_marker_overrides", {})
            vt, vs, va, vst = generate_analysis_viz(
                main_signal, x_arr, y_arr, resampled_t_dt,
                comment_list, b_win, e_win, use_b_area, baseline_end_comment, end_overrides
            )
            analysis_traces.extend(vt)
            analysis_shapes.extend(vs)
            analysis_annotations.extend(va)
            stats.extend(vst)
        except Exception as e:
            print(f"Error generating analysis viz: {e}")

        filtering_traces.extend(dbg_traces)
        analysis_traces.extend(dbg_traces)
        
        # Add comment vertical lines and text labels to Analysis tab as well
        analysis_shapes.extend(filtering_shapes)
        analysis_annotations.extend(filtering_annotations)

        # ---- cache for viewport re-fetch --------------------------------
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

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ===================================================================
#  POST  /api/viewport  (zoom / pan re-fetch)
# ===================================================================
@app.post("/api/viewport")
async def viewport_data(payload: dict):
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

    raw_pq = sd["raw_parquet_path"]
    proc_pq = sd.get("processed_parquet_path")
    if not proc_pq or not os.path.exists(proc_pq):
        raise HTTPException(status_code=400, detail="Call /api/process first")

    try:
        # convert ms → datetime64 for parquet predicate pushdown
        import tzlocal
        local_tz = tzlocal.get_localzone_name()
        t_min = pd.Timestamp(x_min, unit="ms", tz="UTC").tz_convert(local_tz).tz_localize(None)
        t_max = pd.Timestamp(x_max, unit="ms", tz="UTC").tz_convert(local_tz).tz_localize(None)
        pq_filter = [("absolute_time", ">=", t_min),
                      ("absolute_time", "<=", t_max)]

        import pyarrow.parquet as pq
        existing_cols = pq.read_schema(raw_pq).names
        
        # read ONLY the needed columns for the visible range
        cols = ["absolute_time", main_signal]
        if "Finger Pressure" in main_signal:
            if "2: MAP" in existing_cols:
                cols.append("2: MAP")
                    
        df_raw_v = pd.read_parquet(raw_pq, columns=cols, engine="pyarrow",
                                    filters=pq_filter)
        df_filt_v = pd.read_parquet(proc_pq, columns=cols, engine="pyarrow",
                                     filters=pq_filter)

        raw_t = to_unix_ms_local(df_raw_v["absolute_time"])
        raw_y = pd.to_numeric(df_raw_v[main_signal], errors="coerce").astype("float64").values
        filt_t = to_unix_ms_local(df_filt_v["absolute_time"])
        filt_y = pd.to_numeric(df_filt_v[main_signal], errors="coerce").astype("float64").values

        raw_map_v = None
        if "2: MAP" in df_raw_v.columns:
            raw_map_v = pd.to_numeric(df_raw_v["2: MAP"], errors="coerce").astype("float64").values.copy()

        compare_gaussian = cache.get("compare_gaussian", False)

        del df_raw_v, df_filt_v

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
                    "name": "Filtered Data",
                    "line": {"color": "#38bdf8", "width": 1},
                    "connectgaps": False,
                }
            ])
        
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
            use_b_area = settings.get("useBaselineArea", False)
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

        print("[DEBUG] combined_stats starting...")

        combined_stats = []
        num_tests = 0
        if all_stats:
            num_tests = max((len(v) for v in all_stats.values()), default=0)

        for i in range(num_tests):
            configurations = [
                ("FP (Resampled)", "FP_Res"),
                ("FP (MAP 5s)", "FP_MAP"),
                ("CBF", "CBF_Res")
            ]

            for sig_label, key in configurations:
                if key not in all_stats or i >= len(all_stats[key]):
                    continue
                stat = all_stats[key][i]

                trans_dur = stat.get("transition_time")

                # Row 1: Method 1 (Orange)
                combined_stats.append({
                    "Test ID": i + 1,
                    "Signal": sig_label,
                    "Method": "Method 1: Transition-to-End",
                    "Transition Duration (s)": trans_dur,
                    "Duration (s)": stat.get("or_duration"),
                    "Drop (%)": stat.get("or_pct_drop"),
                    "Min": stat.get("or_min_val"),
                    "Area Above Curve (AUC)": stat.get("or_area_below"),
                    "Start / Baseline Value": stat.get("or_trans_val"),
                    "End Value": stat.get("end_val"),
                    "Started In": None
                })

                # Row 2: Method 2 (Green)
                combined_stats.append({
                    "Test ID": i + 1,
                    "Signal": sig_label,
                    "Method": "Method 2: Standing-to-End",
                    "Transition Duration (s)": trans_dur,
                    "Duration (s)": stat.get("gr_duration"),
                    "Drop (%)": stat.get("gr_pct_drop"),
                    "Min": stat.get("gr_min_val"),
                    "Area Above Curve (AUC)": stat.get("gr_area_below"),
                    "Start / Baseline Value": stat.get("gr_stand_val"),
                    "End Value": stat.get("end_val"),
                    "Started In": None
                })

                # Row 3: Method 3 (Blue)
                combined_stats.append({
                    "Test ID": i + 1,
                    "Signal": sig_label,
                    "Method": "Method 3: Baseline Recovery",
                    "Transition Duration (s)": trans_dur,
                    "Duration (s)": stat.get("rec_duration"),
                    "Drop (%)": stat.get("rec_pct_drop"),
                    "Min": stat.get("rec_min_val"),
                    "Area Above Curve (AUC)": stat.get("rec_area"),
                    "Start / Baseline Value": stat.get("baseline"),
                    "End Value": stat.get("rec_end_val"),
                    "Started In": stat.get("rec_started_in")
                })

        print("[DEBUG] writing dataframe to excel...")
        df_stats = pd.DataFrame(combined_stats)
        if not df_stats.empty:
            df_stats = df_stats.sort_values(by=["Signal", "Method", "Test ID"]).reset_index(drop=True)
            
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
            
            # Apple-like styling formats
            header_format = workbook.add_format({
                'font_name': 'Helvetica Neue', 'font_size': 11, 'bold': True,
                'bg_color': '#F5F5F7', 'font_color': '#1D1D1F',
                'bottom': 1, 'border_color': '#D2D2D7',
                'align': 'center', 'valign': 'vcenter'
            })
            
            base_format_dict = {
                'font_name': 'Helvetica Neue', 'font_size': 11,
                'font_color': '#1D1D1F', 'valign': 'vcenter', 'align': 'center'
            }
            format_default = workbook.add_format(base_format_dict)
            
            # Method specific row background colors
            format_m1 = workbook.add_format({**base_format_dict, 'bg_color': '#E8F2FF'}) # Light blue
            format_m2 = workbook.add_format({**base_format_dict, 'bg_color': '#F3E8FF'}) # Light purple
            format_m3 = workbook.add_format({**base_format_dict, 'bg_color': '#E8F5E9'}) # Light green
            
            # Signal specific colors
            format_sig_fp = workbook.add_format({**base_format_dict, 'bg_color': '#FFEFE5', 'bold': True}) # Light orange
            format_sig_cbf = workbook.add_format({**base_format_dict, 'bg_color': '#E5F6FF', 'bold': True}) # Light cyan

            df_stats.to_excel(writer, sheet_name="Statistics", index=False)
            worksheet = writer.sheets["Statistics"]
            
            # Overwrite header with style
            for col_num, value in enumerate(df_stats.columns):
                worksheet.write(0, col_num, value, header_format)
                
            # Write data rows with style
            for row_num in range(len(df_stats)):
                row_data = df_stats.iloc[row_num]
                method = str(row_data["Method"])
                signal = str(row_data["Signal"])
                
                if "Method 1" in method: row_fmt = format_m1
                elif "Method 2" in method: row_fmt = format_m2
                elif "Method 3" in method: row_fmt = format_m3
                else: row_fmt = format_default
                    
                for col_num, value in enumerate(row_data):
                    fmt = row_fmt
                    if col_num == list(df_stats.columns).index("Signal"):
                        if "FP" in signal or "Finger Pressure" in signal: fmt = format_sig_fp
                        elif "CBF" in signal: fmt = format_sig_cbf
                            
                    if pd.isna(value):
                        worksheet.write(row_num + 1, col_num, "", fmt)
                    else:
                        worksheet.write(row_num + 1, col_num, value, fmt)

            for idx, col in enumerate(df_stats.columns):
                worksheet.set_column(idx, idx, max(len(col) + 2, 12))

            df_res_clean = result_df.copy() if not result_df.empty else df_sorted.copy()
            
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

        headers = {
            "Content-Disposition": 'attachment; filename="matfile_analysis.xlsx"'
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
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.isdir(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")
else:
    print(f"[WARNING] Frontend dist directory not found at {frontend_dist}.")
    print("[WARNING] Please run 'npm run build' in the frontend directory to serve the UI.")

