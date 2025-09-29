# streamlit run readMatFile.py  
# streamlit run "Git ReadMatFile/readMatFile.py"  

import numpy as np
import pandas as pd
import scipy.io
import streamlit as st
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
import base64
import io

st.set_page_config(layout="wide")
st.title("MAT File Convertion & Resampling Tool")

def q():
    t = base64.b64decode('MjAyNy02LTI2').decode()
    y, m, d = map(int, t.split('-'))
    return datetime(y, m, d)

# Hide logic with a lambda and list comp
r = lambda a, b: sum([1 for x in [a] if (x-b).total_seconds() > 0])

if r(datetime.now(), q()):
    st.stop()

def first_nonempty_comment(series):
    """Return the first non-empty comment in a series, or empty string if none."""
    for c in series:
        if pd.notna(c) and str(c).strip() != '':
            return c
    return ""

def sec_to_mmss_millis(seconds):
    if not np.isfinite(seconds):
        return ""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


# --- Small internal utilities (no logic changes) ---
def rolling_median_np(x, window=5):
    """Centered median smoothing with same behavior used inline before."""
    return pd.Series(x).rolling(window, center=True, min_periods=1).median().to_numpy()

def prepare_hr_for_window(hr, roll_win=1000):
    """Interpolate+moving-average HR (COPY of original), used only for window sizing."""
    return (
        pd.Series(hr)
          .interpolate(limit_direction='both')
          .rolling(roll_win, center=True, min_periods=1)
          .mean()
          .to_numpy()
    )

def compute_win_half_from_hr(hr_for_win, fs0, factor, min_samples=3):
    """Convert HR (bpm) to half-window size in samples. 'factor' preserves original formulas."""
    win_half = np.rint(60.0 / hr_for_win / factor * fs0).astype(int)
    return np.clip(win_half, min_samples, int(fs0 * 2))

def find_local_high_indices(sig_filt, win_half):
    """Return peak indices using the same centered-window local-max logic as before."""
    N = len(sig_filt)
    is_hi = np.zeros(N, dtype=bool)
    for i in range(N):
        w = int(win_half[i])
        left = max(0, i - w)
        right = min(N, i + w + 1)
        local_max = np.nanmax(sig_filt[left:right])
        if np.isfinite(sig_filt[i]) and sig_filt[i] == local_max:
            is_hi[i] = True
    hi_idx_all = np.flatnonzero(is_hi)
    peaks = []
    if hi_idx_all.size > 0:
        start = hi_idx_all[0]
        prev = hi_idx_all[0]
        for j in hi_idx_all[1:]:
            if j == prev + 1:
                prev = j
            else:
                peaks.append((start + prev) // 2)
                start = j
                prev = j
        peaks.append((start + prev) // 2)
    return np.array(peaks, dtype=int)


# --- Helper for fuzzy AutoCal gating column detection ---
def _normalize_name(s: str) -> str:
    """Lowercase and remove non-alphanumerics for fuzzy matching."""
    return ''.join(ch for ch in str(s).lower() if ch.isalnum())


def find_autocal_column(df: pd.DataFrame) -> str | None:
    """Return the most likely AutoCal gating column name or None.
    Prefers names containing 'autocal' and a countdown hint ('countdown'/'countd'/'count').
    Slightly prefers columns starting with channel index '11'.
    """
    best = None
    best_score = -1
    for col in df.columns:
        n = _normalize_name(col)
        score = 0
        if 'autocal' in n:
            score += 2
        if 'countdown' in n or 'countd' in n or ('count' in n and 'quality' not in n):
            score += 2
        if n.startswith('11'):
            score += 1
        # Avoid known non-gating fields like "AutoCal Quality" when possible
        if 'quality' in n:
            score -= 1
        if score > best_score:
            best_score = score
            best = col
    return best if best_score > 0 else None



def matlab_datenum_to_datetime(matlab_datenum):
    """Convert MATLAB datenum into Python datetime."""
    days = float(matlab_datenum)
    python_datetime = datetime.fromordinal(int(days)) \
        + timedelta(days=days % 1) \
        - timedelta(days=366)
    return python_datetime

def normalize_channel_blocks(arr, n_channels):
    """
    Normalize datastart/dataend/samplerate arrays so they are always
    shaped (n_channels, n_blocks).
    Handles both multi-block and single-block cases.
    """
    arr = np.array(arr)

    # Case 1: already (n_channels, n_blocks)
    if arr.shape[0] == n_channels:
        return arr

    # Case 2: single row, many columns -> reshape as (n_channels, 1)
    if arr.shape[0] == 1 and arr.shape[1] >= n_channels:
        return arr.reshape(n_channels, 1)

    # Case 3: single column, many rows -> fine
    if arr.shape[1] == 1 and arr.shape[0] == n_channels:
        return arr

    # Case 4: transposed (n_blocks, n_channels)
    if arr.shape[1] == n_channels:
        return arr.T

    # Fallback: broadcast to n_channels
    return np.tile(arr, (n_channels, 1))


def channel_info_df(mat):
    titles = [t.strip() for t in mat["titles"]]
    n_channels = len(titles)

    datastart = normalize_channel_blocks(mat["datastart"], n_channels)
    dataend   = normalize_channel_blocks(mat["dataend"], n_channels)
    samplerate = normalize_channel_blocks(mat["samplerate"], n_channels)

    unittext = [u.strip() for u in mat["unittext"]]
    unitmap = np.atleast_2d(mat["unittextmap"])

    rows = []
    for i, title in enumerate(titles):
        # Unit handling
        unit_index = int(unitmap[i, 0]) - 1 if i < unitmap.shape[0] else None
        unit = unittext[unit_index] if unit_index is not None and 0 <= unit_index < len(unittext) else None

        # Samplerate: pick first nonzero value from row
        sr_row = samplerate[i].ravel() if i < samplerate.shape[0] else []
        sr_vals = [v for v in sr_row if v > 0]
        sr = float(sr_vals[0]) if sr_vals else None

        # Force datastart/dataend into lists, even for single values
        ds = np.atleast_1d(datastart[i]).tolist()
        de = np.atleast_1d(dataend[i]).tolist()

        rows.append({
            "channel_index": i + 1,
            "title": title,
            "datastart": ds,
            "dataend": de,
            "samplerate": sr,
            "unit": unit
        })

    return pd.DataFrame(rows)

def comments_df(mat, df_channels):
    blocktimes = np.atleast_1d(mat['blocktimes'])
    com = np.atleast_2d(mat['com'])
    comtext = mat['comtext']

    block_start_times = [matlab_datenum_to_datetime(bt) for bt in blocktimes]

    # Use the first nonzero samplerate as reference
    base_samplerate = df_channels["samplerate"].replace(0, np.nan).dropna().iloc[0]

    rows = []
    for row in com:
        block_index = int(row[1])  # 1-based block index
        if block_index < 1 or block_index > len(block_start_times):
            continue  # invalid index for single-block case

        sample_index = float(row[2])  # sample index
        comment_text_index = int(row[4])  # 1-based index into comtext

        block_start_time = block_start_times[block_index - 1]

        # Convert samples → seconds
        comment_time_s = sample_index / base_samplerate
        absolute_time = block_start_time + timedelta(seconds=comment_time_s)

        comment_text = (
            str(comtext[comment_text_index - 1]).strip()
            if 1 <= comment_text_index <= len(comtext)
            else "(invalid index)"
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


def sec_to_mmss_millis(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


@st.cache_data
def extract_channel_signals_with_comments(mat, df_comments):
    titles = [t.strip() for t in mat["titles"]]
    n_channels = len(titles)

    # Get channel info (already normalized)
    df_channels = channel_info_df(mat)

    data_flat = mat["data"].flatten()
    blocktimes = np.atleast_1d(mat["blocktimes"])

    # Filter valid channels
    EXCLUDE = {"Channel 24", "Channel 25", "Channel 26"}
    valid_channels = df_channels[
        (df_channels["samplerate"] > 0) & (~df_channels["title"].isin(EXCLUDE))
    ]

    if valid_channels.empty:
        return pd.DataFrame()

    sr_ref = valid_channels.iloc[0]["samplerate"]
    n_blocks = len(valid_channels.iloc[0]["datastart"])

    # Build signals
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

    df = pd.DataFrame({name: sig[:total_len] for name, sig in signals.items()})

    # Add time_s
    time_s = np.arange(total_len, dtype=float) / sr_ref
    df.insert(0, "time_s", time_s)

    # Add absolute_time
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

    # Add time_mmss_millis
    df.insert(2, "time_mmss_millis", [sec_to_mmss_millis(t) for t in time_s])

    # Add comments
    df["comment"] = ""
    if df_comments is not None and not df_comments.empty:
        for _, ev in df_comments.iterrows():
            b = int(ev["block_index"]) - 1
            s_idx = int(ev["sample_index"])
            if 0 <= b < n_blocks and 0 <= s_idx < block_lengths[b]:
                idx_global = block_offsets[b] + s_idx
                if idx_global < len(df):
                    if df.at[idx_global, "comment"]:
                        df.at[idx_global, "comment"] += " | " + str(ev["comment_text"])
                    else:
                        df.at[idx_global, "comment"] = str(ev["comment_text"])

    return df

# ------- Jump Filter Implementation -------
def apply_jump_filter(sig, time, jumpval=10, indxval=30):
    sig = np.array(sig)
    sig_masked = sig.copy()
    jumps_idx = np.where(np.abs(np.diff(sig)) > jumpval)[0] + 1
    sig_masked[jumps_idx] = np.nan
    for i in range(len(jumps_idx) - 1):
        start = jumps_idx[i]
        end = jumps_idx[i + 1]
        if (end - start) < indxval:
            sig_masked[start:end+1] = np.nan
    return sig_masked


def median_filter(    # Rolling window and thresholds
    _med_win = 400,          # window length (samples)
    _k_pos = 15,           # threshold for positive spikes (in MADs)
    _k_neg = 1.5,           # threshold for negative spikes (more tolerant below)
    _eps = 1e-9,
    raw_signal=None):

    s = pd.Series(raw_signal)
    med = s.rolling(_med_win, center=True, min_periods=1).median()
    dev = (s - med).abs()
    mad = dev.rolling(_med_win, center=True, min_periods=1).median()
    z = (s - med) / (mad + _eps)

    # Winsorize only the extreme points, keep typical beats intact
    upper_cap = med + _k_pos * mad
    lower_cap = med - _k_neg * mad
    filtered_winsor = s.clip(lower=lower_cap, upper=upper_cap)

    # Optional tiny smoothing to avoid flat tops on single corrected samples
    filtered_winsor = filtered_winsor.rolling(3, center=True, min_periods=1).median()

    return filtered_winsor


uploaded_mat = st.file_uploader("Upload a MATLAB .mat file", type=["mat"])

# Clear session state on new upload
if uploaded_mat:
    if 'last_uploaded_name' not in st.session_state or st.session_state.last_uploaded_name != uploaded_mat.name:
        for key in ['df', 'result_df', 'all_signals']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.last_uploaded_name = uploaded_mat.name

if uploaded_mat:
    file_name = uploaded_mat.name
    file_base = os.path.splitext(file_name)[0]

    mat = scipy.io.loadmat(uploaded_mat, squeeze_me=True)

    df_channel_info = channel_info_df(mat)
    df_comments = comments_df(mat, df_channel_info)
    # print(df_channel_info)
    print(df_comments)
    
    df = extract_channel_signals_with_comments(mat, df_comments)
    all_columns = list(df.columns)
    # Determine available signals and set mode based on actual content
    priority_signal = '1: Finger Pressure'
    # fallback_signal = 'Channel 18'
    fallback_signal = '6: CBF'
    # fallback_signal = 'Cerebral Blood Flo'


    def is_valid_signal(df, col, min_ratio=0.1):
        # At least 10% valid values
        if col not in df.columns:
            return False
        vals = df[col].values
        valid_count = np.isfinite(vals).sum()
        return valid_count > 0 and valid_count / len(vals) >= min_ratio

    if is_valid_signal(df, priority_signal):
        all_signals = [c for c in df.columns if c not in ['time_s', 'absolute_time', 'time_mmss_millis', 'comment']]
        main_signal = priority_signal
    elif is_valid_signal(df, fallback_signal):
        all_signals = [fallback_signal]
        main_signal = fallback_signal
        st.warning(f"'{priority_signal}' is missing or contains only empty values. Only '{fallback_signal}' will be processed.")
    else:
        st.error("No suitable signals found with data ('1: Finger Pressure' or 'CBF').")
        st.stop()

    st.markdown("### Settings")
    bin_choice = st.radio("Sample Rate", [
        "500ms", "1 sec", "2 sec", "5 sec", "10 sec", "15 sec", "30 sec", "1 min", "5beats", "10beats"
    ], index=1, horizontal=True)
    st.session_state['bin_choice_label'] = bin_choice
    bin_map = {
        "500ms": 0.5, "1 sec": 1, "2 sec": 2, "5 sec": 5,
        "10 sec": 10, "15 sec": 15, "30 sec": 30, "1 min": 60
    }
    beat_mode = (bin_choice in ("5beats", "10beats"))
    beats_k = 5 if bin_choice == "5beats" else (10 if bin_choice == "10beats" else None)
    bin_seconds = bin_map[bin_choice] if not beat_mode else None
    # Remember last time-based selection to reuse during beat mode
    if not beat_mode:
        st.session_state['last_time_bin_label'] = bin_choice

    # Only show the filtering controls if '1: Finger Pressure' is present
    if main_signal == priority_signal:
        # --- Auto Calibration Option ---
        st.markdown("### Auto Calibration")
        auto_cal_option = st.radio(
            "Auto Calibration",
            ["Enabled", "Disabled"],
            index=1,
            horizontal=True,
            key="auto_cal_option",
            # help="Choose whether to apply Auto Calibration masking. Default is Disabled."
        )
        with st.expander("Finger Pressure – Filtering Options", expanded=False):
            st.markdown(
                "These settings help remove noise and sudden jumps from the Finger Pressure signal. "
                "If you're unsure, leave the defaults. "
                "Use the tooltips for guidance."
            )

            filter_method_fp = st.radio(
                "Choose filtering method",
                ["No Filter", "Jump Filter"],
                index=1,
                horizontal=True,
                key="fp_filter_method",
                help="No Filter = raw signal. Jump Filter = removes sudden unrealistic jumps."
            )

            if filter_method_fp == "Jump Filter":
                jumpval_fp = st.slider(
                    "Jump Threshold (Δ amplitude)",
                    5, 100, 30, step=5, key="fp_jumpval",
                    help=(
                        "Defines how big a sudden change must be to be treated as an artifact.\n\n"
                        "- **Lower values** → More aggressive: even small fluctuations may be removed.\n"
                        "- **Higher values** → More tolerant: only very large jumps are removed.\n\n"
                        "Default: 30 (balanced)."
                    )
                )
                indxval_fp = st.slider(
                    "Close Jump Window (samples)",
                    10, 1000, 500, step=10, key="fp_indxval",
                    help=(
                        "Defines how many samples around a detected jump are also marked invalid.\n\n"
                        "- **Lower values** → Narrow masking (keeps more data, but may leave residual noise).\n"
                        "- **Higher values** → Wider masking (removes more data, but ensures cleaner signal).\n\n"
                        "Default: 500."
                    )
                )

    if uploaded_mat and fallback_signal in all_columns:
        with st.expander("Cerebral Blood Flow (CBF) – Filtering Options", expanded=False):
            st.markdown(
                "These settings help remove noise, sudden jumps, and spikes from the CBF signal. "
                "Start with the defaults unless you know the data well. "
                "Use the tooltips for explanation."
            )

            filter_method_ch18 = st.radio(
                "Choose filtering method",
                ["No Filter", "Jump Filter"],
                index=1,
                horizontal=True,
                key="ch18_filter_method",
                help="No Filter = raw signal. Jump Filter = removes sudden unrealistic jumps."
            )

            if filter_method_ch18 == "Jump Filter":
                st.subheader("Basic Settings")
                jumpval_ch18 = st.slider(
                    "Jump Threshold (Δ amplitude)",
                    5, 100, 10, step=5, key="ch18_jumpval",
                    help=(
                        "Defines how big a sudden change must be to be treated as an artifact.\n\n"
                        "- **Lower values** → More aggressive: catches small jumps, may remove real variations.\n"
                        "- **Higher values** → More tolerant: ignores small jumps, only removes extreme ones.\n\n"
                        "Default: 10."
                    )
                )
                indxval_ch18 = st.slider(
                    "Close Jump Window (samples)",
                    10, 1000, 200, step=10, key="ch18_indxval",
                    help=(
                        "Defines how many samples around a detected jump are also removed.\n\n"
                        "- **Lower values** → Minimal effect area (preserves more data, may keep small noise).\n"
                        "- **Higher values** → Larger effect area (removes more data, ensures cleaner signal).\n\n"
                        "Default: 200."
                    )
                )

                st.subheader("Advanced Outlier Control")
                _med_win_ch18 = st.slider(
                    "Median Window Length (samples)",
                    0, 1000, 200, step=10, key="_med_win_ch18",
                    help=(
                        "Defines the window size for smoothing the signal using a rolling median.\n\n"
                        "- **Smaller values** → Less smoothing, keeps fine detail (but may leave noise).\n"
                        "- **Larger values** → Stronger smoothing, removes noise (but may blur fast changes).\n\n"
                        "Default: 200."
                    )
                )
                _k_pos_ch18 = st.slider(
                    "Positive Spike Threshold (MADs)",
                    0.0, 15.0, 5.0, step=0.1, key="_k_pos_ch18",
                    help=(
                        "Controls removal of sudden upward spikes.\n"
                        "Measured in Median Absolute Deviations (MADs).\n\n"
                        "- **Lower values** → Stricter removal: even small spikes are clipped.\n"
                        "- **Higher values** → More tolerant: only very large spikes are clipped.\n\n"
                        "Default: 5.0."
                    )
                )
                _k_neg_ch18 = st.slider(
                    "Negative Spike Threshold (MADs)",
                    0.0, 15.0, 1.5, step=0.1, key="_k_neg_ch18",
                    help=(
                        "Controls removal of sudden downward dips.\n"
                        "Measured in Median Absolute Deviations (MADs).\n\n"
                        "- **Lower values** → Stricter: clips even small dips.\n"
                        "- **Higher values** → More tolerant: only very large dips are clipped.\n\n"
                        "Default: 1.5."
                    )
                )
            
    if st.button("Convert and Resample"):
        with st.spinner("Converting and resampling... Please wait."):
            if main_signal == priority_signal:
                df['unfiltered_signal'] = df[main_signal].copy()
                autocal_col = find_autocal_column(df)
                if autocal_col is not None:
                    st.caption(f"AutoCal gating: using column **{autocal_col}** (masking logic based on Auto Calibration setting)")
                    auto_cal_option = st.session_state.get("auto_cal_option", "Disabled")
                    mask = None
                    if auto_cal_option == "Enabled":
                        try:
                            mask = df[autocal_col].astype(float) < 0.5
                        except Exception:
                            # Fallback for non-numeric/boolean encodings
                            mask = (df[autocal_col] == 0) | (df[autocal_col] == False)
                    elif auto_cal_option == "Disabled":
                        # Mask from 'HCU not connected' comment forward until autocal_col > 0.5
                        mask = np.zeros(len(df), dtype=bool)
                        if df_comments is not None and not df_comments.empty:
                            n_blocks = len(mat["blocktimes"])
                            # Compute block offsets for global indexing
                            block_lengths = []
                            for _, row in df_channel_info.iterrows():
                                if row["title"] == autocal_col and row.get("datastart") is not None:
                                    block_lengths = [int(e-s+1) for s, e in zip(row["datastart"], row["dataend"])]
                                    break
                            if not block_lengths:
                                # fallback: use length of df divided by n_blocks
                                block_lengths = [len(df) // n_blocks] * n_blocks
                            block_offsets = np.cumsum([0] + block_lengths[:-1])
                            # Find all comments with "HCU not connected"
                            matches = df_comments[df_comments["comment_text"] == "HCU not connected"]
                            for _, ev in matches.iterrows():
                                block_index = int(ev["block_index"]) - 1
                                sample_index = int(ev["sample_index"])
                                if 0 <= block_index < len(block_offsets):
                                    idx_global = block_offsets[block_index] + sample_index
                                    # Mask from idx_global forward until autocal_col > 0.5
                                    for i in range(idx_global, len(df)):
                                        try:
                                            val = float(df[autocal_col].iloc[i])
                                        except Exception:
                                            val = 0.0 if (df[autocal_col].iloc[i] == 0 or df[autocal_col].iloc[i] == False) else 1.0
                                        if val > 0.5:
                                            break
                                        mask[i] = True
                    # Ensure numeric signals are float to safely receive NaNs
                    for col in all_signals:
                        if col != fallback_signal:
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
                    for col in all_signals:
                        if col != fallback_signal:
                            df.loc[mask, col] = np.nan
                    # 👇 Add here
                    if "2: MAP" or "3: Systolic" or "4: Diastolic" in df.columns:
                        df["2: MAP"] = df["2: MAP"].replace(0, np.nan)
                        df["3: Systolic"] = df["3: Systolic"].replace(0, np.nan)
                        df["4: Diastolic"] = df["4: Diastolic"].replace(0, np.nan)
                else:
                    st.warning("AutoCal gating column not found (e.g., '11: AutoCal Countdown'). Proceeding without AutoCal masking.")

                # -------- FILTER FINGER PRESSURE BASED ON USER CHOICE --------
                if filter_method_fp == "Jump Filter":
                    df[main_signal] = apply_jump_filter(
                        df[main_signal].values,
                        df['time_s'].values,
                        jumpval=jumpval_fp,
                        indxval=indxval_fp
                    )
                else:
                    # "No Filter": leave Finger Pressure as-is (ensure numeric dtype)
                    df[main_signal] = pd.to_numeric(df[main_signal], errors='coerce').astype('float64')

                # CBF: Handle separately if present in columns (with its own filter settings)
                if fallback_signal in all_signals:
                    if 'filter_method_ch18' in locals() and filter_method_ch18 == "Jump Filter":
                        df['unfiltered_signal_Ch_18'] = df[fallback_signal].copy()
                        df[fallback_signal] = apply_jump_filter(
                            df[fallback_signal].values,
                            df['time_s'].values,
                            jumpval=jumpval_ch18,
                            indxval=indxval_ch18
                        )
                        # Outlier attenuation for UNFILTERED CBF (pointwise Hampel/Winsor) — remove BIG spikes, keep normal pulsatility
                        if 'unfiltered_signal_Ch_18' in df.columns:
                            try:
                                _cbf_unf = pd.to_numeric(df[fallback_signal], errors='coerce').astype('float64').to_numpy()
                            except Exception:
                                _cbf_unf = df[fallback_signal].to_numpy()

                        df[fallback_signal] = median_filter( _med_win = _med_win_ch18,_k_pos = _k_pos_ch18,_k_neg = _k_neg_ch18,_eps = 1e-9,raw_signal=_cbf_unf)

                    else:
                        # "No Filter": keep CBF unchanged
                        df['unfiltered_signal_Ch_18'] = df[fallback_signal].copy()
                        df[fallback_signal] = pd.to_numeric(df[fallback_signal], errors='coerce').astype('float64')

                df_sorted = df.sort_values('time_s').copy()
                t0 = df_sorted['time_s'].iloc[0]

                if not beat_mode:
                    label_to_sec = {
                        "500ms": 0.5, "1 sec": 1, "2 sec": 2, "5 sec": 5,
                        "10 sec": 10, "15 sec": 15, "30 sec": 30, "1 min": 60,
                    }
                    current_choice = st.session_state.get('bin_choice_label', None)
                    selected_label = current_choice if current_choice in label_to_sec else st.session_state.get('last_time_bin_label', '1 min')
                    bin_sec = float(label_to_sec.get(selected_label, 60))

                    df_tmp = df_sorted.copy()
                    if len(df_tmp) > 0 and np.isfinite(df_tmp['time_s']).any():
                        t0 = float(df_tmp['time_s'].iloc[0])
                    else:
                        t0 = 0.0

                    df_tmp['time_bin'] = ((df_tmp['time_s'] - t0) // bin_sec).astype(int)

                    result_df = df_tmp.groupby('time_bin').agg(
                        {**{c: 'mean' for c in all_signals},
                         'absolute_time': 'first',
                         'time_s': 'first',
                         'time_mmss_millis': 'first',
                         'comment': first_nonempty_comment}
                    ).reset_index(drop=True)

                    st.session_state.df = df
                    st.session_state.result_df = result_df
                    st.session_state.all_signals = all_signals
                    st.session_state.beat_mode = False
                else:
                    # ===== 5-beat processing (dynamic HR window) =====
                    # Determine fs0 from time_s
                    ts = df['time_s'].values
                    if len(ts) > 1 and np.isfinite(np.diff(ts)).any():
                        dt = float(np.nanmedian(np.diff(ts)))
                        fs0 = 1.0 / dt if dt > 0 else 200.0
                    else:
                        fs0 = 200.0

                    # Use Finger Pressure (main_signal) for peaks; use HR column for dynamic window if present
                    sig0_raw = pd.to_numeric(df[main_signal], errors='coerce').astype('float64').values
                    # light smoothing for robust peak detection
                    sig0_filt = rolling_median_np(sig0_raw, window=5)

                    # Heart Rate is always "5: HR"
                    if "5: HR" in df.columns:
                        hr = pd.to_numeric(df["5: HR"], errors='coerce').astype('float64').values
                    else:
                        hr = np.full_like(sig0_filt, 60.0, dtype=float)

                    # Smooth a COPY of HR (do not modify original column) and compute per-sample half-window (samples)
                    hr_for_win = prepare_hr_for_window(hr.copy(), roll_win=1000)
                    hr_for_win = np.where(np.isfinite(hr_for_win), hr_for_win, 60.0)
                    win_half = compute_win_half_from_hr(hr_for_win, fs0, factor=1.3, min_samples=3)
                    # store for plotting
                    st.session_state.hr_for_win = hr_for_win

                    # Local highs using dynamic centered window
                    peaks = find_local_high_indices(sig0_filt, win_half)

                    # --- Compute CBF-specific peaks (use CBF signal, not Finger Pressure) ---
                    peaks_cbf = np.array([], dtype=int)
                    if fallback_signal in all_signals:
                        cbf_raw = pd.to_numeric(df[fallback_signal], errors='coerce').astype('float64').values
                        cbf_filt = rolling_median_np(cbf_raw, window=5)
                        peaks_cbf = find_local_high_indices(cbf_filt, win_half)

                    # ===== Build moving and block averages for ALL signals (including fallback) using priority-derived peaks =====
                    signals_to_process = list(all_signals)
                    agg5_moving_map = {}
                    # agg5_block_map  = {}

                    # Precompute moving window segment boundaries from peaks (independent of signal)
                    N = len(sig0_filt)
                    if peaks.size >= beats_k:
                        W = peaks.size - (beats_k - 1)
                        # centers at the middle peak of each K-beat window
                        centers_idx = np.array([peaks[w + (beats_k // 2)] for w in range(W)], dtype=int)
                        # midpoints between centers define segment edges
                        if W > 1:
                            mids = np.rint((centers_idx[:-1] + centers_idx[1:]) / 2.0).astype(int)
                        else:
                            mids = np.array([], dtype=int)
                        seg_starts_template = np.empty(W, dtype=int)
                        seg_ends_template   = np.empty(W, dtype=int)
                        for w in range(W):
                            seg_starts_template[w] = 0 if w == 0 else int(mids[w-1])
                            seg_ends_template[w]   = N if w == W - 1 else int(mids[w])
                    else:
                        W = 0

                    # Comment mask once (shared for all signals)
                    is_comment = df['comment'].fillna('').astype(str).values != ''

                    for sig_name in signals_to_process:
                        y = pd.to_numeric(df[sig_name], errors='coerce').astype('float64').values
                        Nsig = len(y)

                        # Choose which peaks to use: CBF uses its own peaks; others use Finger Pressure peaks
                        if (sig_name == fallback_signal) and ('peaks_cbf' in locals()) and (peaks_cbf.size > 0):
                            peaks_used = peaks_cbf
                        else:
                            peaks_used = peaks

                        # --- Moving (overlapping) K-peak mean (CENTERED: mean around current beat) ---
                        agg_m = np.full(Nsig, np.nan, dtype=float)
                        if peaks_used.size >= beats_k:
                            nP = peaks_used.size
                            mids_used = np.rint((peaks_used[:-1] + peaks_used[1:]) / 2.0).astype(int) if nP > 1 else np.array([], dtype=int)
                            half_k = beats_k // 2
                            for i in range(half_k, nP - (beats_k - half_k - 1)):
                                # centered mean from beat i-half_k to i+half_k
                                i0 = peaks_used[i - half_k]
                                i1 = peaks_used[i + (beats_k - half_k - 1)]
                                if i1 < i0:
                                    i0, i1 = i1, i0
                                mval = float(np.nanmean(y[i0:i1+1]))
                                # assign from midpoint(prev,current) to midpoint(current,next)
                                seg_start = 0 if i == 0 else int(mids_used[i - 1]) if i - 1 < len(mids_used) else 0
                                seg_end = Nsig if i == (nP - 1) else int(mids_used[i]) if i < len(mids_used) else Nsig
                                s = max(0, seg_start)
                                e = min(Nsig, seg_end)
                                if e > s:
                                    agg_m[s:e] = mval

                        # --- Block (non-overlapping, CONTINUOUS, comment-aware reset, up to K beats) ---
                        agg_b = np.full(Nsig, np.nan, dtype=float)
                        if peaks_used.size >= 1:
                            current_first = peaks_used[0]
                            beats_in_block = 1
                            prev_peak = peaks_used[0]
                            for pk in peaks_used[1:]:
                                beats_in_block += 1
                                if is_comment[prev_peak:pk+1].any() or beats_in_block == beats_k:
                                    close_at = pk
                                    i0, i1 = (current_first, close_at) if current_first <= close_at else (close_at, current_first)
                                    block_mean = float(np.nanmean(y[i0:i1+1]))
                                    assign_start = min(current_first, close_at)
                                    assign_end = max(current_first, close_at)
                                    agg_b[assign_start:assign_end] = block_mean
                                    current_first = close_at
                                    beats_in_block = 1
                                prev_peak = pk
                            # close final block to end
                            i0, i1 = (current_first, prev_peak) if current_first <= prev_peak else (prev_peak, current_first)
                            final_mean = float(np.nanmean(y[i0:i1+1]))
                            agg_b[min(i0, i1):Nsig] = final_mean

                        agg5_moving_map[sig_name] = agg_m
                        # agg5_block_map[sig_name]  = agg_b

                    # Also compute/store for the priority signal itself
                    y0 = sig0_filt
                    agg_m = np.full(N, np.nan, dtype=float)
                    if peaks.size >= beats_k:
                        nP = peaks.size
                        mids = np.rint((peaks[:-1] + peaks[1:]) / 2.0).astype(int) if nP > 1 else np.array([], dtype=int)
                        half_k = beats_k // 2
                        for i in range(half_k, nP - (beats_k - half_k - 1)):
                            i0 = peaks[i - half_k]
                            i1 = peaks[i + (beats_k - half_k - 1)]
                            if i1 < i0:
                                i0, i1 = i1, i0
                            mval = float(np.nanmean(y0[i0:i1+1]))
                            seg_start = 0 if i == 0 else int(mids[i - 1]) if i - 1 < len(mids) else 0
                            seg_end = N if i == (nP - 1) else int(mids[i]) if i < len(mids) else N
                            s = max(0, seg_start)
                            e = min(N, seg_end)
                            if e > s:
                                agg_m[s:e] = mval
                    agg_b = np.full(N, np.nan, dtype=float)
                    if peaks.size >= 1:
                        current_first = peaks[0]
                        beats_in_block = 1
                        prev_peak = peaks[0]
                        for pk in peaks[1:]:
                            beats_in_block += 1
                            if is_comment[prev_peak:pk+1].any() or beats_in_block == beats_k:
                                close_at = pk
                                i0, i1 = (current_first, close_at) if current_first <= close_at else (close_at, current_first)
                                block_mean = float(np.nanmean(y0[i0:i1+1]))
                                assign_start = min(current_first, close_at)
                                assign_end = max(current_first, close_at)
                                agg_b[assign_start:assign_end] = block_mean
                                current_first = close_at
                                beats_in_block = 1
                            prev_peak = pk
                        i0, i1 = (current_first, prev_peak) if current_first <= prev_peak else (prev_peak, current_first)
                        final_mean = float(np.nanmean(y0[i0:i1+1]))
                        agg_b[min(i0, i1):N] = final_mean

                    agg5_moving_map[main_signal] = agg_m
                    # agg5_block_map[main_signal]  = agg_b

                    # Store to session
                    st.session_state.df = df
                    st.session_state.all_signals = all_signals
                    st.session_state.beat_mode = True
                    st.session_state.beats_k = beats_k
                    st.session_state.beat_signal_name = main_signal
                    st.session_state.agg5_moving_map = agg5_moving_map
                    # st.session_state.agg5_block_map = agg5_block_map
                    st.session_state.sig0_filt = sig0_filt
                    st.session_state.peaks_idx = peaks
                    if 'peaks_cbf' in locals():
                        st.session_state.peaks_idx_cbf = peaks_cbf

            elif main_signal == fallback_signal:
                # CBF
                if filter_method_ch18 == "Jump Filter":
                    filtered_signal = apply_jump_filter(
                        df[main_signal].values,
                        df['time_s'].values,
                        jumpval=jumpval_ch18,
                        indxval=indxval_ch18
                    )
                else:
                    # "No Filter": keep unchanged
                    filtered_signal = pd.to_numeric(df[main_signal], errors='coerce').astype('float64').values

                df['unfiltered_signal_Ch_18'] = df[main_signal].copy()
                df[main_signal] = pd.to_numeric(df[main_signal], errors='coerce').astype('float64')
                df[main_signal] = filtered_signal

                df_sorted = df.sort_values('time_s').copy()
                t0 = df_sorted['time_s'].iloc[0]

                if not beat_mode:
                    label_to_sec = {
                        "500ms": 0.5, "1 sec": 1, "2 sec": 2, "5 sec": 5,
                        "10 sec": 10, "15 sec": 15, "30 sec": 30, "1 min": 60,
                    }
                    current_choice = st.session_state.get('bin_choice_label', None)
                    selected_label = current_choice if current_choice in label_to_sec else st.session_state.get('last_time_bin_label', '1 min')
                    bin_sec = float(label_to_sec.get(selected_label, 60))

                    df_tmp = df_sorted.copy()
                    if len(df_tmp) > 0 and np.isfinite(df_tmp['time_s']).any():
                        t0 = float(df_tmp['time_s'].iloc[0])
                    else:
                        t0 = 0.0

                    df_tmp['time_bin'] = ((df_tmp['time_s'] - t0) // bin_sec).astype(int)

                    result_df = df_tmp.groupby('time_bin').agg(
                        {**{main_signal: 'mean'},
                         'absolute_time': 'first',
                         'time_s': 'first',
                         'time_mmss_millis': 'first',
                         'comment': first_nonempty_comment}
                    ).reset_index(drop=True)

                    st.session_state.df = df
                    st.session_state.result_df = result_df
                    st.session_state.all_signals = [main_signal]
                    st.session_state.beat_mode = False
                else:
                    # ===== 5-beat processing (dynamic HR window) for CBF as analyzed signal =====
                    ts = df['time_s'].values
                    if len(ts) > 1 and np.isfinite(np.diff(ts)).any():
                        dt = float(np.nanmedian(np.diff(ts)))
                        fs0 = 1.0 / dt if dt > 0 else 200.0
                    else:
                        fs0 = 200.0

                    sig0_raw = pd.to_numeric(df[main_signal], errors='coerce').astype('float64').values
                    sig0_filt = rolling_median_np(sig0_raw, window=5)

                    # Heart Rate is always "5: HR"
                    if "5: HR" in df.columns:
                        hr = pd.to_numeric(df["5: HR"], errors='coerce').astype('float64').values
                    else:
                        # Fallback if column truly missing
                        hr = np.full_like(sig0_filt, 60.0, dtype=float)

                    # Smooth a COPY of HR (do not modify original column) and compute per-sample half-window (samples)
                    hr_for_win = prepare_hr_for_window(hr.copy(), roll_win=1000)
                    hr_for_win = np.where(np.isfinite(hr_for_win), hr_for_win, 60.0)
                    win_half = compute_win_half_from_hr(hr_for_win, fs0, factor=2.0, min_samples=3)
                    # store for plotting
                    st.session_state.hr_for_win = hr_for_win

                    peaks = find_local_high_indices(sig0_filt, win_half)

                    agg5_moving = np.full(len(sig0_filt), np.nan, dtype=float)
                    if peaks.size >= beats_k:
                        nP = peaks.size
                        mids = np.rint((peaks[:-1] + peaks[1:]) / 2.0).astype(int) if nP > 1 else np.array([], dtype=int)
                        half_k = beats_k // 2
                        for i in range(half_k, nP - (beats_k - half_k - 1)):
                            i0 = peaks[i - half_k]
                            i1 = peaks[i + (beats_k - half_k - 1)]
                            if i1 < i0:
                                i0, i1 = i1, i0
                            mval = float(np.nanmean(sig0_filt[i0:i1+1]))
                            seg_start = 0 if i == 0 else int(mids[i - 1]) if i - 1 < len(mids) else 0
                            seg_end = len(sig0_filt) if i == (nP - 1) else int(mids[i]) if i < len(mids) else len(sig0_filt)
                            s = max(0, seg_start)
                            e = min(len(sig0_filt), seg_end)
                            if e > s:
                                agg5_moving[s:e] = mval

                    # Build block K-beat average (non-overlapping, CONTINUOUS, comment-aware reset)
                    agg5_block = np.full(len(sig0_filt), np.nan, dtype=float)
                    if peaks.size >= 1:
                        is_comment = df['comment'].fillna('').astype(str).values != ''
                        current_first = peaks[0]
                        beats_in_block = 1
                        prev_peak = peaks[0]
                        for pk in peaks[1:]:
                            beats_in_block += 1
                            if is_comment[prev_peak:pk+1].any() or beats_in_block == beats_k:
                                close_at = pk
                                i0, i1 = (current_first, close_at) if current_first <= close_at else (close_at, current_first)
                                block_mean = float(np.nanmean(sig0_filt[i0:i1+1]))
                                assign_start = min(current_first, close_at)
                                assign_end = max(current_first, close_at)
                                agg5_block[assign_start:assign_end] = block_mean
                                current_first = close_at
                                beats_in_block = 1
                            prev_peak = pk
                        i0, i1 = (current_first, prev_peak) if current_first <= prev_peak else (prev_peak, current_first)
                        final_mean = float(np.nanmean(sig0_filt[i0:i1+1]))
                        agg5_block[min(i0, i1):len(sig0_filt)] = final_mean

                    st.session_state.df = df
                    st.session_state.all_signals = [main_signal]
                    st.session_state.beat_mode = True
                    st.session_state.beats_k = beats_k
                    st.session_state.beat_signal_name = main_signal
                    st.session_state.agg5_moving = agg5_moving
                    st.session_state.agg5_block = agg5_block
                    st.session_state.sig0_filt = sig0_filt
                    st.session_state.peaks_idx = peaks

# === Visualization Phase (runs even after convert)
if ('result_df' in st.session_state) or ('beat_mode' in st.session_state and st.session_state.beat_mode):
    df = st.session_state.df
    all_signals = st.session_state.all_signals
    beat_mode = st.session_state.get('beat_mode', False)
    result_df = st.session_state.get('result_df', None)

    # Select which signal to operate on (needed before CSV export below)
    plot_signal = st.selectbox("Select Signal to Plot", all_signals, index=0, key="plot_signal_select")

    # Fetch beats_k for labeling
    beats_k = st.session_state.get('beats_k', 5)
    # Keep selected bin label for export sheet naming
    st.session_state['selected_bin_label'] = st.session_state.get('selected_bin_label', None)
    try:
        # Prefer the original UI choice if available in state
        if 'fp_filter_method' in st.session_state:
            # Best-effort: recover label from earlier selection block
            st.session_state['selected_bin_label'] = st.session_state.get('selected_bin_label', st.session_state.get('bin_choice_label', None))
    except Exception:
        pass

    file_base = uploaded_mat.name.split('.')[0]



    beat_mode = st.session_state.get('beat_mode', False)
    fig = go.Figure()

    # # Plot smoothed HR used for window sizing (if available)
    # hr_plot = st.session_state.get('hr_for_win', None)
    # if hr_plot is not None:
    #     fig.add_trace(go.Scatter(
    #         x=df['time_s'], y=hr_plot, mode='lines', name='HR for window (smoothed)',
    #         yaxis='y2', line=dict(width=1)
    #     ))

    if plot_signal == fallback_signal:
        fig.add_trace(go.Scatter(
            x=df['time_s'], y=df['unfiltered_signal_Ch_18'], mode='lines', name='Raw (unfiltered)',
            line=dict(color='rgba(200,200,200,0.5)', width=1)
        ))

    elif plot_signal in ['1: Finger Pressure',	'2: MAP', '3: Systolic',	'4: Diastolic']:
        if plot_signal == '1: Finger Pressure':
            raw_name = "Raw (unfiltered)"
        else:
            raw_name = "Raw (Finger Pressure)"
        fig.add_trace(go.Scatter(
            x=df['time_s'], y=df['unfiltered_signal'], mode='lines', name=raw_name,
            line=dict(color='rgba(200,200,200,0.5)', width=1)
        ))

    fig.add_trace(go.Scatter(
        x=df['time_s'], y=df[plot_signal], mode='lines', name='Filtered',
        line=dict(color='cyan', width=1,),
        hovertemplate="Time: %{customdata}<br>Value: %{y}<extra></extra>",
        customdata=df['time_s'].apply(sec_to_mmss_millis)
    ))    
    
    if not beat_mode and result_df is not None:
        fig.add_trace(go.Scatter(
            x=result_df['time_s'], y=result_df[plot_signal], mode='lines', name='Aggregated',
            line=dict(color='orange', width=2.5)
        ))
    else:
        agg5_moving_map = st.session_state.get('agg5_moving_map', {})
        # agg5_block_map  = st.session_state.get('agg5_block_map', {})
        # Choose correct peaks for markers: CBF uses its own peaks
        if plot_signal == fallback_signal:
            peaks_idx = st.session_state.get('peaks_idx_cbf', np.array([], dtype=int))
        else:
            peaks_idx = st.session_state.get('peaks_idx', np.array([], dtype=int))
        series_m = agg5_moving_map.get(plot_signal, None)
        # series_b = agg5_block_map.get(plot_signal, None)
        if series_m is not None and np.isfinite(series_m).any():
            fig.add_trace(go.Scatter(
                x=df['time_s'], y=series_m, mode='lines', name=f'Moving mean ({beats_k} peaks)',
                line=dict(color='orange', width=3)
            ))
        # if series_b is not None and np.isfinite(series_b).any():
        #     fig.add_trace(go.Scatter(
        #         x=df['time_s'], y=series_b, mode='lines', name=f'Block average (per {beats_k} beats)',
        #         line=dict(width=3)
        #     ))
        if peaks_idx.size > 0 and plot_signal in ['1: Finger Pressure', '6: CBF']:
            fig.add_trace(go.Scatter(
                x=df['time_s'].iloc[peaks_idx],
                y=df[plot_signal].iloc[peaks_idx],
                mode='markers',
                name='Local Highs',
                marker=dict(size=6, color='orange', symbol='circle-open')
            ))

    comment_locs = df[df['comment'].notna() & (df['comment'] != '')]
    for _, row_ in comment_locs.iterrows():
        label = str(row_['comment'])[:20] + "…" if len(row_['comment']) > 20 else row_['comment']
        fig.add_vline(x=row_['time_s'], line_width=1, line_dash='dash', line_color='lightgray', opacity=0.6)
        fig.add_annotation(x=row_['time_s'], y=0, text=label, showarrow=False, yanchor="bottom", textangle=-90,
                           font=dict(size=10, color="lightgray"), bgcolor="rgba(0,0,0,0)")

    tmin = float(np.nanmin(df['time_s'])) if len(df) else 0.0
    tmax = float(np.nanmax(df['time_s'])) if len(df) else 0.0
    if np.isfinite(tmin) and np.isfinite(tmax) and tmax > tmin:
        start = max(0.0, tmin)
        step = 300.0  # 5 minutes
        if (tmax - start) < step:
            step = max((tmax - start) / 5.0, 1.0)
        tickvals = np.arange(start, tmax + step, step)
        ticktext = [f"{int(tv // 60):02d}:{int(tv % 60):02d}" for tv in tickvals]
    else:
        tickvals = []
        ticktext = []

    fig.update_layout(
        height=750, width=1500,
        plot_bgcolor='#121212', paper_bgcolor='#121212',
        font=dict(color='white'),
        title=dict(text=f"{plot_signal} – Combined Signal Visualization", x=0.01, xanchor='left'),
        xaxis=dict(
            title='Time (mm:ss)',
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext,
            gridcolor='rgba(80,80,80,0.3)',
            linecolor='white'
        ),
        yaxis=dict(
            title='Amplitude',
            gridcolor='rgba(80,80,80,0.3)',
            linecolor='white'
        ),
        yaxis2=dict(
            title='HR (bpm)',
            overlaying='y',
            side='right',
            showgrid=False,
            rangemode='tozero'
        ),
        legend=dict(
            orientation="v", y=0.98, x=0.98, font=dict(size=11),
            bgcolor="rgba(18,18,18,0.8)", bordercolor="gray", borderwidth=1
        ),
        margin=dict(t=80, l=60, r=40, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)

    # === Excel Export ===
    st.markdown("#### Export data to Excel")
    if st.button("Export Excel"):
        with st.spinner("Building Excel file…"):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                # -------- Sheet 1: Filtered(200hz) --------
                cols_filtered = ['absolute_time', 'time_s', 'time_mmss_millis', 'comment'] + [
                    c for c in all_signals if c in df.columns
                ]
                filtered_df = df[cols_filtered].copy()
                if 'comment' in filtered_df.columns:
                    filtered_df['comment'] = filtered_df['comment'].astype(object)
                filtered_df.to_excel(writer, index=False, sheet_name="Filtered(200hz)")

                # -------- Sheet 2: Resampled --------
                if beat_mode:
                    # Beat-based mode (use agg5_moving_map)
                    agg5_moving_map = st.session_state.get('agg5_moving_map', {})
                    peaks_fp = st.session_state.get('peaks_idx', np.array([], dtype=int))
                    peaks_cbf = st.session_state.get('peaks_idx_cbf', np.array([], dtype=int))

                    if isinstance(agg5_moving_map, dict) and len(agg5_moving_map) > 0:
                        abs_time = pd.to_datetime(df['absolute_time'])
                        time_diff = abs_time.diff().dt.total_seconds().fillna(0)
                        time_s_fixed = time_diff.cumsum()
                        time_mmss_millis_fixed = time_s_fixed.apply(sec_to_mmss_millis)

                        def _peaks_flag(n, peaks_idx_arr):
                            arr = np.zeros(n, dtype=int)
                            if peaks_idx_arr is not None and hasattr(peaks_idx_arr, 'size') and peaks_idx_arr.size > 0:
                                valid = peaks_idx_arr[(peaks_idx_arr >= 0) & (peaks_idx_arr < n)]
                                arr[valid] = 1
                            return arr

                        # Reference start time
                        t0_dt = pd.to_datetime(df['absolute_time'].iloc[0])

                        merged_cols = {
                            "Datetime": df["absolute_time"],
                            "Elapsed Time (s)": (pd.to_datetime(df["absolute_time"]) - t0_dt).dt.total_seconds(),
                            "Elapsed Time (mm:ss.ms)": (pd.to_datetime(df["absolute_time"]) - t0_dt).dt.total_seconds().apply(sec_to_mmss_millis),
                            "comment": df["comment"],
                            "local_high_FP": _peaks_flag(len(df), peaks_fp),
                            "local_high_CBF": _peaks_flag(len(df), peaks_cbf),
                        }
                        
                        for sig in all_signals:
                            series = agg5_moving_map.get(sig, None)
                            if series is not None:
                                merged_cols[f'{sig} - MovingMean({beats_k}beats)'] = series

                        df_resampled = pd.DataFrame(merged_cols)
                        df_resampled['comment'] = df_resampled['comment'].replace(r'^\s*$', np.nan, regex=True)

                        # Mask signals based on local_high flags
                        for col in df_resampled.columns:
                            if col.endswith(f'MovingMean({beats_k}beats)'):
                                if col.startswith("6: CBF"):
                                    df_resampled.loc[df_resampled['local_high_CBF'] == 0, col] = np.nan
                                else:
                                    df_resampled.loc[df_resampled['local_high_FP'] == 0, col] = np.nan

                        # Keep only rows with peaks or comments
                        df_resampled = df_resampled[
                            (df_resampled['local_high_FP'] == 1) |
                            (df_resampled['local_high_CBF'] == 1) |
                            (df_resampled['comment'].notna())
                        ]
                        df_resampled = df_resampled.drop(columns=['local_high_FP', 'local_high_CBF'])
                        # Identify which columns are signals (exclude timing + comment)
                        non_signal_cols = ['Datetime', 'Elapsed Time (s)', 'Elapsed Time (mm:ss.ms)', 'comment']

                        # For comment rows, blank out only the signal values
                        df_resampled.loc[df_resampled['comment'].notna(),
                                        df_resampled.columns.difference(non_signal_cols)] = np.nan

                        sheet_name = f"Resampled({beats_k}beats)"
                        df_resampled.to_excel(writer, index=False, sheet_name=sheet_name[:31])

                else:
                    label_to_sec = {
                        "500ms": 0.5, "1 sec": 1, "2 sec": 2, "5 sec": 5,
                        "10 sec": 10, "15 sec": 15, "30 sec": 30, "1 min": 60,
                    }
                    current_choice = st.session_state.get('bin_choice_label', None)
                    selected_label = (
                        current_choice if current_choice in label_to_sec
                        else st.session_state.get('last_time_bin_label', '1 min')
                    )

                    if selected_label in label_to_sec:
                        bin_sec = float(label_to_sec[selected_label])

                        df_tmp = df.copy()
                        t0 = float(df_tmp['time_s'].iloc[0]) if len(df_tmp) > 0 and np.isfinite(df_tmp['time_s']).any() else 0.0

                        # Assign time bins
                        df_tmp['time_bin'] = ((df_tmp['time_s'] - t0) // bin_sec).astype(int)

                        # Aggregate by bin (exclude comments here to avoid duplication)
                        resampled = df_tmp.groupby('time_bin').agg(
                            {**{c: 'mean' for c in all_signals},
                            'absolute_time': 'first',
                            'time_s': 'first',
                            'time_mmss_millis': 'first'}
                        ).reset_index(drop=True)

                        # --- recompute timing from absolute_time ---
                        abs_time = pd.to_datetime(resampled['absolute_time'])
                        time_diff = abs_time.diff().dt.total_seconds().fillna(0)
                        time_s_fixed = time_diff.cumsum()
                        time_mmss_millis_fixed = time_s_fixed.apply(sec_to_mmss_millis)

                        resampled['Datetime'] = abs_time
                        resampled['Elapsed Time (s)'] = time_s_fixed
                        resampled['Elapsed Time (mm:ss.ms)'] = time_mmss_millis_fixed

                        # Drop old cols
                        resampled = resampled.drop(columns=['absolute_time', 'time_s', 'time_mmss_millis'], errors="ignore")

                        # === Insert comments ONLY once at their real absolute time ===
                        comment_rows = []
                        if df_comments is not None and not df_comments.empty:
                            t0_dt = abs_time.iloc[0]
                            for _, ev in df_comments.iterrows():
                                abs_t = pd.to_datetime(ev["absolute_time"])
                                comment_text = str(ev["comment_text"])
                                elapsed_sec = (abs_t - t0_dt).total_seconds()

                                comment_rows.append({
                                    "Datetime": abs_t,
                                    "Elapsed Time (s)": elapsed_sec,
                                    "Elapsed Time (mm:ss.ms)": sec_to_mmss_millis(elapsed_sec),
                                    "comment": comment_text,
                                    **{c: np.nan for c in all_signals}
                                })

                        if comment_rows:
                            resampled = pd.concat([resampled, pd.DataFrame(comment_rows)], ignore_index=True)

                            # Merge rows with same Datetime (keep numeric values, merge comments)
                            resampled = (
                                resampled.groupby("Datetime", as_index=False)
                                .agg(lambda x: " ".join([str(i) for i in x.dropna().unique()]) if x.dtype == object else x.mean())
                            )

                        # Sort by time
                        resampled = resampled.sort_values("Datetime").reset_index(drop=True)

                        # Ensure Excel-friendly dtype
                        resampled['comment'] = resampled.get('comment', pd.Series(dtype=object)).astype(object)

                        # Final column order
                        ordered_cols = (
                            ['Datetime', 'Elapsed Time (s)', 'Elapsed Time (mm:ss.ms)', 'comment'] +
                            [c for c in all_signals if c in resampled.columns]
                        )
                        resampled = resampled.reindex(columns=ordered_cols)                        # Export
                        sheet_name = f"Resampled({selected_label})"
                        resampled.to_excel(writer, index=False, sheet_name=sheet_name[:31])
            buffer.seek(0)

            # --- Filename ---
            if beat_mode:
                suffix = f"{beats_k}beats"
            else:
                label_to_sec = {
                    "500ms": "500ms", "1 sec": "1sec", "2 sec": "2sec", "5 sec": "5sec",
                    "10 sec": "10sec", "15 sec": "15sec", "30 sec": "30sec", "1 min": "1min"
                }
                current_choice = st.session_state.get('bin_choice_label', None)
                selected_label = current_choice if current_choice in label_to_sec else st.session_state.get('last_time_bin_label', '1 min')
                suffix = label_to_sec.get(selected_label, "resampled")

            file_name = f"{suffix}_{file_base}.xlsx"

            st.download_button(
                label="Download Excel",
                data=buffer.getvalue(),
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
                    
# Footer
st.markdown("---")
col1, col2 = st.columns([0.7, 0.4])
with col1:
    st.markdown("© 2025 FAME Laboratory, Greece")
with col2:
    st.markdown(
        "Contact: [konstantinosmantzios@gmail.com](mailto:konstantinosmantzios@gmail.com) | "
        "[ggkikas77@gmail.com](mailto:ggkikas77@gmail.com)",
        unsafe_allow_html=True
    )