from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
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
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    print_memory("Upload Start")

    # enforce single session limit to save memory
    for existing_sid in list(SESSION_STORE.keys()):
        _cleanup_session(existing_sid)
    gc.collect()

    session_id = str(uuid.uuid4())
    os.makedirs(TEMP_DIR, exist_ok=True)
    mat_path = os.path.join(TEMP_DIR, f"{session_id}.mat")

    # save uploaded file to disk
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

        # write parquet (with small row groups for efficient range reads)
        raw_pq = os.path.join(TEMP_DIR, f"{session_id}_raw.parquet")
        df_raw.to_parquet(raw_pq, engine="pyarrow", index=False,
                          row_group_size=50_000)

        del df_raw
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

        # ---- read raw data from parquet (with optional time filter) -----
        pq_filters = None
        if start_s is not None and end_s is not None:
            pq_filters = [("time_s", ">=", float(start_s)),
                          ("time_s", "<=", float(end_s))]

        df_raw = pd.read_parquet(raw_pq, engine="pyarrow",
                                 filters=pq_filters)
        print_memory("After parquet read")

        # ---- signal list & selected signal ----------------------------
        all_signals = [
            c for c in df_raw.columns 
            if ":" in c and not any(x in c for x in ["MAP", "Systolic", "Diastolic"])
        ]
        main_signal = settings.get("selectedSignal", "")
        if not main_signal or main_signal not in all_signals:
            main_signal = next(
                (s for s in all_signals
                 if s.startswith("1:") or "Finger Pressure" in s),
                all_signals[0] if all_signals else "",
            )

        # keep a copy of the raw (unfiltered) values for the selected sig
        df_raw_sorted = df_raw.sort_values("time_s")
        raw_y_display = pd.to_numeric(
            df_raw_sorted[main_signal], errors="coerce"
        ).astype("float64").values.copy()
        
        raw_map_display = None
        if "2: MAP" in df_raw_sorted.columns:
            raw_map_display = pd.to_numeric(
                df_raw_sorted["2: MAP"], errors="coerce"
            ).astype("float64").values.copy()

        compare_gaussian = settings.get("compareGaussian", False)

        # ---- process (filter / beat-detect / resample) ----------------
        df_sorted, result_df, peaks, peaks_cbf, agg5_moving_map = \
            process_session_data(df_raw, df_channel_info, df_comments, settings)

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
        raw_t = df_sorted["absolute_time"].astype("int64").values // 1_000_000
        filt_y = pd.to_numeric(
            df_sorted[main_signal], errors="coerce"
        ).astype("float64").values

        # ---- LTTB downsample for initial view -------------------------
        raw_t_f64 = raw_t.astype(np.float64)
        raw_t_ds, raw_y_ds = lttb_downsample(raw_t_f64, raw_y_display, target_pts)
        filt_t_ds, filt_y_ds = lttb_downsample(raw_t_f64, filt_y, target_pts)

        dbg_traces = []
        if compare_gaussian and "Finger Pressure" in main_signal:
            raw_y_copy = raw_map_display if raw_map_display is not None else raw_y_display
            gauss_y = pd.Series(raw_y_copy).rolling(window=1000, win_type='gaussian', center=True, min_periods=1).mean(std=1000/6).values
            valid_mask = ~np.isnan(gauss_y)
            if valid_mask.any():
                g_t = raw_t_f64[valid_mask]
                g_y = gauss_y[valid_mask]
                g_t_ds, g_y_ds = lttb_downsample(g_t, g_y, target_pts)
                g_t_ds, g_y_ds = insert_gaps(g_t_ds, g_y_ds)
                
                dbg_traces.append({
                    "trace_id": "gauss1000",
                    "x": g_t_ds.astype(int).tolist(),
                    "y": _nan_safe_list(g_y_ds),
                    "type": "scattergl", "mode": "lines",
                    "name": "MAP - Gaussian 1000",
                    "line": {"color": "#a855f7", "width": 2},
                    "connectgaps": False,
                })

        # ---- resampled data (already sparse → send in full) -----------
        if not result_df.empty:
            res_t = result_df["absolute_time"].astype("int64").values // 1_000_000
            res_y = result_df[main_signal].values.astype("float64")
        else:
            res_t = raw_t
            res_y = raw_y_display

        if len(res_t) > target_pts:
            res_t_ds, res_y = lttb_downsample(res_t.astype(np.float64), res_y, target_pts)
            res_t = res_t_ds.astype(int)

        raw_t_ds, raw_y_ds = insert_gaps(raw_t_ds, raw_y_ds)
        filt_t_ds, filt_y_ds = insert_gaps(filt_t_ds, filt_y_ds)
        res_t, res_y = insert_gaps(res_t.astype(np.float64), res_y)

        # ---- build Plotly traces --------------------------------------
        traces = []
        shapes = []
        annotations = []
        stats = []
        viz_mode = settings.get("analysisView", "Filtering Preview")

        if viz_mode == "Filtering Preview":
            traces.append({
                "trace_id": "raw",
                "x": raw_t_ds.astype(int).tolist(),
                "y": _nan_safe_list(raw_y_ds),
                "type": "scattergl", "mode": "lines",
                "name": "Raw Data",
                "line": {"color": "rgba(156,163,175,0.4)", "width": 1},
                "connectgaps": False,
            })
            traces.append({
                "trace_id": "filtered",
                "x": filt_t_ds.astype(int).tolist(),
                "y": _nan_safe_list(filt_y_ds),
                "type": "scattergl", "mode": "lines",
                "name": "Filtered Data",
                "line": {"color": "#38bdf8", "width": 1},
                "connectgaps": False,
            })
            traces.append({
                "trace_id": "resampled",
                "x": res_t.astype(int).tolist(),
                "y": _nan_safe_list(res_y),
                "type": "scattergl", "mode": "lines+markers" if settings.get("resampleMode") == "Beat-based" else "lines",
                "name": "Resampled Data",
                "line": {"color": "#f97316", "width": 2},
                "marker": {"size": 4} if settings.get("resampleMode") == "Beat-based" else None,
                "connectgaps": False,
            })

            if settings.get("resampleMode") == "Beat-based" and peaks.size > 0:
                p_t = raw_t[peaks]
                p_y = filt_y[peaks]
                if len(p_t) > 3000:
                    step = len(p_t) // 3000
                    p_t = p_t[::step]
                    p_y = p_y[::step]
                traces.append({
                    "trace_id": "peaks",
                    "x": p_t.astype(int).tolist(),
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
                        
                    abs_time_ms = int(pd.to_datetime(row["absolute_time"]).value) // 1_000_000
                    
                    shapes.append({
                        "type": "line", "yref": "paper",
                        "x0": abs_time_ms, "x1": abs_time_ms,
                        "y0": 0, "y1": 1,
                        "line": {"color": line_color, "width": 1, "dash": "dash"}
                    })
                    
                    annotations.append({
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
                    
        else:
            # Supine-to-Standing Analysis view
            traces.append({
                "trace_id": "resampled",
                "x": res_t.astype(int).tolist(),
                "y": _nan_safe_list(res_y),
                "type": "scattergl", "mode": "lines+markers" if settings.get("resampleMode") == "Beat-based" else "lines",
                "name": "Resampled Data",
                "line": {"color": "#f97316", "width": 2},
                "marker": {"size": 4} if settings.get("resampleMode") == "Beat-based" else None,
                "connectgaps": False,
            })

            sig_viz = settings.get("selectedSignal", "Finger Pressure")
            
            if not result_df.empty:
                x_arr = result_df["time_s"].values
                y_arr = result_df.get(sig_viz, pd.Series()).values
                resampled_t_dt = pd.to_datetime(result_df["absolute_time"]).values
            else:
                x_arr = df_sorted["time_s"].values
                y_arr = df_sorted.get(sig_viz, pd.Series()).values
                resampled_t_dt = pd.to_datetime(df_sorted["absolute_time"]).values
                
            comment_list = df_comments.to_dict("records")
            b_win = abs(settings.get("analysisBaselineWindow", 60))
            e_win = settings.get("analysisEndMarkerWindow", 10)
            use_b_area = settings.get("useBaselineArea", False)
            baseline_end_comment = settings.get("baselineEndComment", "Transition")

            try:
                vt, vs, va, vst = generate_analysis_viz(
                    sig_viz, x_arr, y_arr, resampled_t_dt,
                    comment_list, b_win, e_win, use_b_area, baseline_end_comment
                )
                traces.extend(vt)
                shapes.extend(vs)
                annotations.extend(va)
                stats.extend(vst)
            except Exception as e:
                print(f"Error generating analysis viz: {e}")

        traces.extend(dbg_traces)

        # ---- cache for viewport re-fetch --------------------------------
        s_hash = _settings_hash(settings)
        sd["plot_cache"] = {
            "settings_hash": s_hash,
            "selected_signal": main_signal,
            "analysis_view": settings.get("analysisView", "Filtering Preview"),
            "resample_mode": settings.get("resampleMode", "Time-based"),
            "compare_gaussian": settings.get("compareGaussian", False),
            "dev_mode": settings.get("devMode", False),
        }

        # ---- free heavy objects -----------------------------------------
        del df_sorted, result_df, agg5_moving_map, raw_t, raw_y_display, filt_y
        gc.collect()
        print_memory("Process Complete")

        return {
            "traces": traces,
            "shapes": shapes,
            "annotations": annotations,
            "analysis_stats": stats,
            "available_signals": all_signals,
            "settings_hash": s_hash,
        }

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
        t_min = pd.Timestamp(x_min, unit="ms")
        t_max = pd.Timestamp(x_max, unit="ms")
        pq_filter = [("absolute_time", ">=", t_min),
                      ("absolute_time", "<=", t_max)]

        import pyarrow.parquet as pq
        existing_cols = pq.read_schema(raw_pq).names
        
        # read ONLY the needed columns for the visible range
        cols = ["absolute_time", main_signal]
        if cache.get("compare_gaussian", False) and "Finger Pressure" in main_signal:
            if "2: MAP" in existing_cols:
                cols.append("2: MAP")
                    
        df_raw_v = pd.read_parquet(raw_pq, columns=cols, engine="pyarrow",
                                    filters=pq_filter)
        df_filt_v = pd.read_parquet(proc_pq, columns=cols, engine="pyarrow",
                                     filters=pq_filter)

        raw_t = df_raw_v["absolute_time"].astype("int64").values // 1_000_000
        raw_y = pd.to_numeric(df_raw_v[main_signal], errors="coerce").astype("float64").values
        filt_t = df_filt_v["absolute_time"].astype("int64").values // 1_000_000
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
                    "x": g_t_ds.astype(int).tolist(),
                    "y": _nan_safe_list(g_y_ds),
                    "type": "scattergl", "mode": "lines",
                    "name": "MAP - Gaussian 1000",
                    "line": {"color": "#a855f7", "width": 2},
                    "connectgaps": False,
                })

        viz_mode = cache.get("analysis_view", "Filtering Preview")
        
        traces = []
        if viz_mode == "Filtering Preview":
            traces.extend([
                {
                    "trace_id": "raw",
                    "x": rt.astype(int).tolist(),
                    "y": _nan_safe_list(ry),
                    "type": "scattergl", "mode": "lines",
                    "name": "Raw Data",
                    "line": {"color": "rgba(156,163,175,0.4)", "width": 1},
                    "connectgaps": False,
                },
                {
                    "trace_id": "filtered",
                    "x": ft.astype(int).tolist(),
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
            res_t = df_res_v["absolute_time"].astype("int64").values // 1_000_000
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
            "x": res_t.astype(int).tolist(),
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

    try:
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

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_sorted.head(1000).to_excel(writer, sheet_name="Raw_Preview", index=False)

            if not df_comments.empty:
                df_comments.to_excel(writer, sheet_name="Comments", index=False)

            if not df_comments.empty:
                from export import extract_stats_for_signal
                comment_list = df_comments.to_dict("records")
                stats = []
                for sig in all_signals:
                    if settings.get("resampleMode") == "Beat-based":
                        x_arr = df_sorted["time_s"].values
                        y_arr = agg5_moving_map.get(sig, np.array([]))
                    else:
                        x_arr = result_df["time_s"].values
                        y_arr = result_df[sig].values

                    sig_stats = extract_stats_for_signal(
                        sig, x_arr, y_arr, comment_list,
                        baseline_window=abs(settings.get("baselineWinStart", -60)),
                        end_marker_window=(
                            settings.get("standingWinStart", 10)
                            + settings.get("standingWinDuration", 30)
                        ),
                        use_baseline_area=False,
                    )
                    stats.extend(sig_stats)

                if stats:
                    pd.DataFrame(stats).to_excel(
                        writer, sheet_name="Analysis_Stats", index=False
                    )

        output.seek(0)

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

