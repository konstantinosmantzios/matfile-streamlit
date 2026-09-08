import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
from scipy.signal import savgol_filter, butter, filtfilt, find_peaks

# ---------------------------------------------------------
# Signal Processing Core Algorithms & Helpers
# ---------------------------------------------------------

def first_nonempty_comment(series):
    """Return the first non-empty comment in a series, or empty string if none."""
    for c in series:
        if pd.notna(c) and str(c).strip() != '':
            return c
    return ""

def sec_to_mmss_millis(seconds: float) -> str:
    """Format float seconds to MM:SS.ms string representation."""
    if not np.isfinite(seconds):
        return ""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"

def rolling_median_np(x, window=5):
    """Centered median smoothing."""
    return pd.Series(x).rolling(window, center=True, min_periods=1).median().to_numpy()

def prepare_hr_for_window(hr, roll_win=1000):
    """Interpolate and moving-average HR used for dynamic peak window sizing."""
    return (
        pd.Series(hr)
          .interpolate(limit_direction='both')
          .rolling(roll_win, center=True, min_periods=1)
          .mean()
          .to_numpy()
    )

def compute_win_half_from_hr(hr_for_win, fs0, factor, min_samples=3):
    """Convert HR (bpm) to half-window size in samples."""
    win_half = np.rint(60.0 / hr_for_win / factor * fs0).astype(int)
    # Clip to max 0.6 seconds. The dicrotic notch occurs soon after the peak.
    # Allowing window to grow too large (e.g. 2s) when HR signal drops causes artifacts to suppress distant valid beats.
    return np.clip(win_half, min_samples, int(fs0 * 0.6))

def find_local_high_indices(sig_filt, win_half):
    """Return peak indices using topological peak detection and dynamic HR window filtering."""
    N = len(sig_filt)
    
    # 1. Topological peak detection to get all candidate peaks
    # We use a robust MAD-based prominence to avoid massive noise artifacts from inflating the threshold
    mad = np.nanmedian(np.abs(sig_filt - np.nanmedian(sig_filt)))
    prom_thresh = max(0.01, mad * 1.4826 * 0.2)
    candidates, _ = find_peaks(sig_filt, prominence=prom_thresh)
    
    # 2. Filter candidates: must be the highest candidate in its dynamic HR-based window
    valid_peaks = []
    for peak in candidates:
        w = int(win_half[peak])
        left = max(0, peak - w)
        right = min(N, peak + w + 1)
        
        # Check if there's a higher candidate peak in this specific window
        idx_in_window = (candidates >= left) & (candidates < right)
        candidates_in_window = candidates[idx_in_window]
        
        is_max = True
        for other_peak in candidates_in_window:
            if sig_filt[other_peak] > sig_filt[peak]:
                is_max = False
                break
            elif sig_filt[other_peak] == sig_filt[peak] and other_peak > peak:
                is_max = False
                break
                
        if is_max:
            valid_peaks.append(peak)
            
    return np.array(valid_peaks, dtype=int)

def apply_savgol_filter(y, window_length=51, polyorder=3):
    """Smooths signal while preserving peak shapes using Savitzky-Golay."""
    y_clean = np.where(np.isnan(y), np.nanmedian(y) if np.isfinite(y).any() else 0, y)
    if len(y_clean) < window_length:
        return y
    polyorder = min(polyorder, window_length - 1)
    if polyorder < 1: polyorder = 1
    y_filtered = savgol_filter(y_clean, window_length, polyorder)
    return np.where(np.isnan(y), np.nan, y_filtered)

def apply_butter_lowpass(y, cutoff_freq, fs, order=4):
    """Applies a Butterworth low-pass filter to cut off high frequency noise."""
    y_clean = np.where(np.isnan(y), np.nanmedian(y) if np.isfinite(y).any() else 0, y)
    nyq = 0.5 * fs
    if cutoff_freq >= nyq:
        return y
    normal_cutoff = cutoff_freq / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y_filtered = filtfilt(b, a, y_clean)
    return np.where(np.isnan(y), np.nan, y_filtered)

def apply_hampel_filter(y, window_size=5, n_sigmas=3):
    """Removes outlier spikes using rolling median and Median Absolute Deviation (MAD)."""
    y_series = pd.Series(y)
    rolling_median = y_series.rolling(window=window_size, center=True).median()
    rolling_mad = y_series.rolling(window=window_size, center=True).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )
    threshold = n_sigmas * 1.4826 * rolling_mad
    difference = np.abs(y_series - rolling_median)
    outlier_idx = difference > threshold
    y_series[outlier_idx] = rolling_median[outlier_idx]
    return y_series.values

def find_autocal_column(df: pd.DataFrame) -> str | None:
    """Fuzzy column matcher to identify AutoCal countdown indices."""
    best = None
    best_score = -1
    for col in df.columns:
        n = ''.join(ch for ch in str(col).lower() if ch.isalnum())
        score = 0
        if 'autocal' in n:
            score += 2
        if 'countdown' in n or 'countd' in n or ('count' in n and 'quality' not in n):
            score += 2
        if n.startswith('11'):
            score += 1
        if 'quality' in n:
            score -= 1
        if score > best_score:
            best_score = score
            best = col
    return best if best_score > 0 else None

# ---------------------------------------------------------
# MATLAB File Parsing & Data Extraction
# ---------------------------------------------------------

def matlab_datenum_to_datetime(matlab_datenum):
    """Convert MATLAB datenum into Python datetime."""
    days = float(matlab_datenum)
    python_datetime = datetime.fromordinal(int(days)) \
        + timedelta(days=days % 1) \
        - timedelta(days=366)
    return python_datetime

def normalize_channel_blocks(arr, n_channels):
    """Normalize datastart/dataend/samplerate blocks into a consistent 2D shape."""
    arr = np.array(arr)
    if arr.shape[0] == n_channels:
        return arr
    if arr.shape[0] == 1 and arr.shape[1] >= n_channels:
        return arr.reshape(n_channels, 1)
    if arr.shape[1] == 1 and arr.shape[0] == n_channels:
        return arr
    if arr.shape[1] == n_channels:
        return arr.T
    return np.tile(arr, (n_channels, 1))

def channel_info_df(mat):
    """Extract metadata information for channels from the MATLAB dict."""
    titles = [str(t).strip() for t in mat["titles"]]
    n_channels = len(titles)

    datastart = normalize_channel_blocks(mat["datastart"], n_channels)
    dataend   = normalize_channel_blocks(mat["dataend"], n_channels)
    samplerate = normalize_channel_blocks(mat["samplerate"], n_channels)

    unittext = [str(u).strip() for u in mat["unittext"]]
    unitmap = np.atleast_2d(mat["unittextmap"])

    rows = []
    for i, title in enumerate(titles):
        unit_index = int(unitmap[i, 0]) - 1 if i < unitmap.shape[0] else None
        unit = unittext[unit_index] if unit_index is not None and 0 <= unit_index < len(unittext) else None
        
        sr_row = samplerate[i].ravel() if i < samplerate.shape[0] else []
        sr_vals = [v for v in sr_row if v > 0]
        sr = float(sr_vals[0]) if sr_vals else None

        rows.append({
            "channel_index": i + 1,
            "title": title,
            "datastart": np.atleast_1d(datastart[i]).tolist(),
            "dataend": np.atleast_1d(dataend[i]).tolist(),
            "samplerate": sr,
            "unit": unit
        })
    return pd.DataFrame(rows)

def comments_df(mat, df_channels):
    """Extract and reconstruct comments timestamps from MATLAB data structures."""
    if 'com' not in mat or 'comtext' not in mat:
        return pd.DataFrame()

    blocktimes = np.atleast_1d(mat['blocktimes'])
    com = np.atleast_2d(mat['com'])
    comtext = np.atleast_1d(mat['comtext'])

    block_start_times = [matlab_datenum_to_datetime(bt) for bt in blocktimes]
    base_samplerate = df_channels["samplerate"].replace(0, np.nan).dropna().iloc[0]

    rows = []
    for row in com:
        block_index = int(row[1])
        if block_index < 1 or block_index > len(block_start_times):
            continue

        sample_index = float(row[2])
        comment_text_index = int(row[4])

        block_start_time = block_start_times[block_index - 1]
        comment_time_s = sample_index / base_samplerate
        absolute_time = block_start_time + timedelta(seconds=comment_time_s)

        comment_text = (
            str(comtext[comment_text_index - 1]).strip()
            if 1 <= comment_text_index <= len(comtext)
            else ""
        )

        rows.append({
            "block_index": block_index,
            "block_start_time": block_start_time,
            "sample_index": int(sample_index),
            "comment_time_s": comment_time_s,
            "absolute_time": absolute_time,
            "comment_text": comment_text
        })
    return pd.DataFrame(rows)

@st.cache_data
def extract_channel_signals_with_comments(mat, df_comments):
    """Flatten and extract signal waveforms and align comment indicators."""
    titles = [str(t).strip() for t in mat["titles"]]
    df_channels = channel_info_df(mat)
    data_flat = mat["data"].flatten()
    blocktimes = np.atleast_1d(mat["blocktimes"])

    EXCLUDE = {"Channel 24", "Channel 25", "Channel 26"}
    valid_channels = df_channels[
        (df_channels["samplerate"] > 0) & (~df_channels["title"].isin(EXCLUDE))
    ]

    if valid_channels.empty:
        return pd.DataFrame()

    sr_ref = valid_channels.iloc[0]["samplerate"]
    n_blocks = len(valid_channels.iloc[0]["datastart"])

    signals = {}
    block_lengths = [0] * n_blocks

    for _, row in valid_channels.iterrows():
        tname = row["title"]
        sig_parts = []
        for b, (s, e) in enumerate(zip(row["datastart"], row["dataend"])):
            if s >= 1 and e > s and e <= len(data_flat):
                part = data_flat[int(s) - 1:int(e)]
                sig_parts.append(part)
                if block_lengths[b] == 0:
                    block_lengths[b] = len(part)
        if sig_parts:
            signals[tname] = np.concatenate(sig_parts)

    if not signals:
        return pd.DataFrame()

    total_len = sum(block_lengths)
    block_offsets = np.cumsum([0] + block_lengths[:-1])

    df = pd.DataFrame({name: np.round(sig[:total_len], 2) for name, sig in signals.items()})
    time_s = np.arange(total_len, dtype=float) / sr_ref
    df.insert(0, "time_s", time_s)

    # Reconstruct exact datetime array
    block_start_times = [matlab_datenum_to_datetime(bt) for bt in blocktimes]
    abs_time = np.empty(total_len, dtype="datetime64[ns]")
    write_pos = 0
    for b, Lb in enumerate(block_lengths):
        if Lb <= 0:
            continue
        base = block_start_times[min(b, len(block_start_times) - 1)]
        seg_times = [base + timedelta(seconds=k / sr_ref) for k in range(Lb)]
        abs_time[write_pos:write_pos + Lb] = np.array(seg_times, dtype="datetime64[ns]")
        write_pos += Lb
    df.insert(1, "absolute_time", abs_time)
    df.insert(2, "time_mmss_millis", [sec_to_mmss_millis(t) for t in time_s])
    
    block_id_col = np.empty(total_len, dtype=int)
    write_pos = 0
    for b, Lb in enumerate(block_lengths):
        if Lb <= 0: continue
        block_id_col[write_pos:write_pos + Lb] = b
        write_pos += Lb
    df.insert(3, "block_index", block_id_col)

    # Inject comments into timeline
    df["comment"] = ""
    if df_comments is not None and not df_comments.empty:
        for _, ev in df_comments.iterrows():
            b = int(ev["block_index"]) - 1
            s_idx = int(ev["sample_index"])
            if 0 <= b < n_blocks and 0 <= s_idx < block_lengths[b]:
                idx_global = block_offsets[b] + s_idx
                if idx_global < len(df):
                    existing = df.at[idx_global, "comment"]
                    if existing:
                        df.at[idx_global, "comment"] += " | " + str(ev["comment_text"])
                    else:
                        df.at[idx_global, "comment"] = str(ev["comment_text"])
    return df
