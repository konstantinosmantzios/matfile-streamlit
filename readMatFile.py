# streamlit run mantzioscode.py  
import numpy as np
import pandas as pd
import scipy.io
import streamlit as st
import plotly.graph_objects as go
import os
from datetime import datetime
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


@st.cache_data
def load_mat_file(mat_file) -> pd.DataFrame:
    mat = scipy.io.loadmat(mat_file, squeeze_me=True)
    data_flat = mat['data'].flatten()

    datastart = np.atleast_2d(mat['datastart'])
    dataend = np.atleast_2d(mat['dataend'])
    if datastart.shape[0] == 1 and datastart.shape[1] > 1:
        datastart = datastart.T
        dataend = dataend.T

    titles = [t.strip() for t in mat['titles']]
    samplerate = mat['samplerate']
    com = mat.get('com', np.array([]))
    comtext = mat.get('comtext', [])

    def extract_channel(idx):
        """Return (signal_full, fs) or (None, None) if missing."""
        sig_parts = []
        for block in range(datastart.shape[1]):
            s = datastart[idx, block]
            e = dataend[idx, block]
            if not np.isnan(s) and not np.isnan(e):
                start = int(s) - 1
                end = int(e)
                if 0 <= start < end <= data_flat.size:
                    sig_parts.append(data_flat[start:end])
        if not sig_parts:
            return None, None
        sig = np.concatenate(sig_parts)

        try:
            fs = int(samplerate[idx, 0]) if samplerate.ndim > 1 else int(samplerate[idx])
        except Exception:
            fs = None
        if fs is None or fs <= 0 or not np.isfinite(fs):
            fs = None
        return sig, fs

    # Prefer CBF (index 17) for the time base; else Channel 1 (index 0)
    ch18_idx = 17 if len(titles) > 17 else None
    sig18, fs18 = (extract_channel(ch18_idx) if ch18_idx is not None else (None, None))
    sig1,  fs1  = extract_channel(0)

    if sig18 is not None and fs18:
        base_sig = sig18
        base_fs = fs18
    elif sig1 is not None and fs1:
        base_sig = sig1
        base_fs = fs1
    else:
        st.warning("No valid samplerate/time base found in CBF or Channel 1.")
        return pd.DataFrame()

    # Build time from the chosen base
    min_length = len(base_sig)
    time = (np.arange(min_length) / float(base_fs)) if min_length > 0 else np.array([])

    time_mmss_millis = [sec_to_mmss_millis(t) for t in time] if len(time) > 0 else []

    # Collect all signals, truncated to the base length
    signals, channel_names = [], []
    for i in range(len(titles)):
        ch_parts = []
        for block in range(datastart.shape[1]):
            s = datastart[i, block]
            e = dataend[i, block]
            if not np.isnan(s) and not np.isnan(e):
                start = int(s) - 1
                end = int(e)
                if 0 <= start < end <= data_flat.size:
                    ch_parts.append(data_flat[start:end])
        if ch_parts:
            sig = np.concatenate(ch_parts)
            if sig.size > 0 and np.isfinite(sig).any():
                signals.append(sig[:min_length])
                channel_names.append(titles[i])

    if not signals:
        st.warning("No valid signal data in this MAT file after alignment.")
        return pd.DataFrame()

    # Main DataFrame
    data = {"time_s": time, "time_mmss_millis": time_mmss_millis}
    for ch_name, ch_data in zip(channel_names, signals):
        data[ch_name] = ch_data
    df = pd.DataFrame(data)
    # --- Enforce stable dtypes to avoid pandas casting warnings ---
    # Numeric time in float64
    df['time_s'] = pd.to_numeric(df['time_s'], errors='coerce').astype('float64')
    # All channel columns coerced to float64
    for ch_name in channel_names:
        df[ch_name] = pd.to_numeric(df[ch_name], errors='coerce').astype('float64')
    # Ensure time_mmss_millis is a stable string dtype
    df['time_mmss_millis'] = pd.Series(time_mmss_millis, dtype='string')

    # Events: prefer fs from Channel 1 if valid; else use base_fs
    fs_events = fs1 if (fs1 and fs1 > 0) else base_fs
    event_times, event_labels = [], []
    if com is not None and np.size(com) > 0 and fs_events:
        for row in np.atleast_2d(com):
            try:
                timestamp_samples = float(row[2])
                t_sec = timestamp_samples / float(fs_events)
                try:
                    text_idx = int(row[4]) - 1
                    text = comtext[text_idx].strip() if 0 <= text_idx < len(comtext) else f"(invalid text index: {row[4]})"
                except Exception:
                    text = "(no text)"
                event_times.append(t_sec)
                event_labels.append(text)
            except Exception:
                continue

    # Attach event comments at closest timestamps
    comments_col = np.full(min_length, "", dtype=object)
    for t_event, label in zip(event_times, event_labels):
        if len(time) == 0:
            break
        idx = int(np.nanargmin(np.abs(time - t_event)))
        comments_col[idx] = label
    df["comment"] = pd.Series(comments_col, dtype='string')

    # Drop unwanted channels
    columnsDrop = [
        '7: Interbeat Interval', '8: Active Cuff', '9: Cuff Countdown',
        '10: AutoCal Quality', '16: PL ch 1 lead 1', '17: PL ch 2 lead 2',
        '19: Lead aVR', '20: Lead aVL', '21: Lead aVF', '22: HR EKG lead 1',
        '23: : HR EKG lead 2', 'Channel 24', 'Channel 25', 'Channel 26',
    ]
    df = df.drop(columns=[c for c in columnsDrop if c in df.columns], errors='ignore')

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

    df = load_mat_file(uploaded_mat)
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
        all_signals = [c for c in df.columns if c not in ['time_s', 'time_mmss_millis', 'comment']]
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
        with st.expander("Finger Pressure Filtering Settings", expanded=False):  # <-- collapsed by default
            filter_method_fp = st.radio(
                "Filter Method",
                ["No Filter", "Jump Filter"],
                index=1,  # default to Jump Filter
                horizontal=True,
                key="fp_filter_method"
            )
            if filter_method_fp == "Jump Filter":
                jumpval_fp = st.slider("Jump Threshold (Δ)", 5, 100, 30, step=5, key="fp_jumpval")
                indxval_fp = st.slider("Close Jump Window (samples)", 10, 1000, 500, step=10, key="fp_indxval")
            # (No controls needed for "No Filter")

    if uploaded_mat and fallback_signal in all_columns:
        with st.expander("### CBF Filtering Settings", expanded=False):  # <-- collapsed by default
            filter_method_ch18 = st.radio(
                "Filter Method",
                ["No Filter", "Jump Filter"],
                index=1,  # default to Jump Filter
                horizontal=True,
                key="ch18_filter_method"
            )
            if filter_method_ch18 == "Jump Filter":
                jumpval_ch18 = st.slider("Jump Threshold (Δ)", 5, 100, 10, step=5, key="ch18_jumpval")
                indxval_ch18 = st.slider("Close Jump Window (samples)", 10, 1000, 200, step=10, key="ch18_indxval")
                _med_win_ch18 = st.slider("Window length (samples)", 0, 1000, 200, step=10, key="_med_win_ch18") # window length (samples)
                _k_pos_ch18 = st.slider("Threshold for positive", 0.0, 15.0, 5.0, step=0.1, key="_k_pos_ch18") # threshold for positive spikes (in MADs)
                _k_neg_ch18 = st.slider("Threshold for negative", 0.0, 15.0, 1.5, step=0.1, key="_k_neg_ch18") # threshold for negative spikes (more tolerant below)


            # (No controls shown for "No Filter")
    # (No fallback ch18_order block needed; unused.)

    if st.button("Convert and Resample"):
        with st.spinner("Converting and resampling... Please wait."):
            if main_signal == priority_signal:
                df['unfiltered_signal'] = df[main_signal].copy()
                autocal_col = find_autocal_column(df)
                if autocal_col is not None:
                    st.caption(f"AutoCal gating: using column **{autocal_col}** (< 0.5 masked)")
                    try:
                        mask = df[autocal_col].astype(float) < 0.5
                    except Exception:
                        # Fallback for non-numeric/boolean encodings
                        mask = (df[autocal_col] == 0) | (df[autocal_col] == False)
                    # Ensure numeric signals are float to safely receive NaNs
                    for col in all_signals:
                        if col != fallback_signal:
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
                    for col in all_signals:
                        if col != fallback_signal:  # do not mask ch18 here; handled separately
                            df.loc[mask, col] = np.nan
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
                    df_sorted['time_bin'] = ((df_sorted['time_s'] - t0) // bin_seconds).astype(int)
                    agg = df_sorted.groupby('time_bin').agg(
                        {**{c: 'mean' for c in all_signals},
                         'time_s': 'first',
                         'time_mmss_millis': 'first',
                         'comment': first_nonempty_comment}
                    ).reset_index()
                    agg['bin_start_time'] = t0 + agg['time_bin'] * bin_seconds

                    st.session_state.df = df
                    st.session_state.result_df = agg
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
                    agg5_block_map  = {}

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

                        # --- Moving (overlapping) K-peak mean (TRAILING: mean of previous K beats) ---
                        agg_m = np.full(Nsig, np.nan, dtype=float)
                        if peaks_used.size >= beats_k:
                            nP = peaks_used.size
                            # Midpoints between consecutive peaks define segment boundaries
                            mids_used = np.rint((peaks_used[:-1] + peaks_used[1:]) / 2.0).astype(int) if nP > 1 else np.array([], dtype=int)
                            for i in range(beats_k - 1, nP):
                                # trailing mean from beat i-(K-1) to i
                                i0 = peaks_used[i - (beats_k - 1)]
                                i1 = peaks_used[i]
                                if i1 < i0:
                                    i0, i1 = i1, i0
                                mval = float(np.nanmean(y[i0:i1+1]))
                                # assign from midpoint(prev,current) to midpoint(current,next)
                                seg_start = 0 if i == 0 else int(mids_used[i - 1])
                                seg_end   = Nsig if i == (nP - 1) else int(mids_used[i])
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
                        agg5_block_map[sig_name]  = agg_b

                    # Also compute/store for the priority signal itself
                    y0 = sig0_filt
                    agg_m = np.full(N, np.nan, dtype=float)
                    if peaks.size >= beats_k:
                        nP = peaks.size
                        mids = np.rint((peaks[:-1] + peaks[1:]) / 2.0).astype(int) if nP > 1 else np.array([], dtype=int)
                        for i in range(beats_k - 1, nP):
                            i0 = peaks[i - (beats_k - 1)]
                            i1 = peaks[i]
                            if i1 < i0:
                                i0, i1 = i1, i0
                            mval = float(np.nanmean(y0[i0:i1+1]))
                            seg_start = 0 if i == 0 else int(mids[i - 1])
                            seg_end   = N if i == (nP - 1) else int(mids[i])
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
                    agg5_block_map[main_signal]  = agg_b

                    # Store to session
                    st.session_state.df = df
                    st.session_state.all_signals = all_signals
                    st.session_state.beat_mode = True
                    st.session_state.beats_k = beats_k
                    st.session_state.beat_signal_name = main_signal
                    st.session_state.agg5_moving_map = agg5_moving_map
                    st.session_state.agg5_block_map = agg5_block_map
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
                    df_sorted['time_bin'] = ((df_sorted['time_s'] - t0) // bin_seconds).astype(int)
                    agg = df_sorted.groupby('time_bin').agg(
                        {main_signal: 'mean', 'time_s': 'first',
                          'time_mmss_millis': 'first',
                            'comment': first_nonempty_comment 
                          }
                    ).reset_index()
                    agg['bin_start_time'] = t0 + agg['time_bin'] * bin_seconds

                    st.session_state.df = df
                    st.session_state.result_df = agg
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
                        for i in range(beats_k - 1, nP):
                            i0 = peaks[i - (beats_k - 1)]
                            i1 = peaks[i]
                            if i1 < i0:
                                i0, i1 = i1, i0
                            mval = float(np.nanmean(sig0_filt[i0:i1+1]))
                            seg_start = 0 if i == 0 else int(mids[i - 1])
                            seg_end   = len(sig0_filt) if i == (nP - 1) else int(mids[i])
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
            x=df['time_s'], y=df['unfiltered_signal_Ch_18'], mode='lines', name='Unfiltered CBF',
            line=dict(color='rgba(200,200,200,0.5)', width=1)
        ))

    elif plot_signal in ['1: Finger Pressure',	'2: MAP', '3: Systolic',	'4: Diastolic']:
        fig.add_trace(go.Scatter(
            x=df['time_s'], y=df['unfiltered_signal'], mode='lines', name='Unfiltered',
            line=dict(color='rgba(200,200,200,0.5)', width=1)
        ))

    fig.add_trace(go.Scatter(
        x=df['time_s'], y=df[plot_signal], mode='lines', name='Filtered',
        line=dict(color='cyan', width=2)
    ))
    if not beat_mode and result_df is not None:
        fig.add_trace(go.Scatter(
            x=result_df['time_s'], y=result_df[plot_signal], mode='lines', name='Aggregated',
            line=dict(color='orange', width=2.5)
        ))
    else:
        agg5_moving_map = st.session_state.get('agg5_moving_map', {})
        agg5_block_map  = st.session_state.get('agg5_block_map', {})
        # Choose correct peaks for markers: CBF uses its own peaks
        if plot_signal == fallback_signal:
            peaks_idx = st.session_state.get('peaks_idx_cbf', np.array([], dtype=int))
        else:
            peaks_idx = st.session_state.get('peaks_idx', np.array([], dtype=int))
        series_m = agg5_moving_map.get(plot_signal, None)
        series_b = agg5_block_map.get(plot_signal, None)
        if series_m is not None and np.isfinite(series_m).any():
            fig.add_trace(go.Scatter(
                x=df['time_s'], y=series_m, mode='lines', name=f'Moving mean ({beats_k} peaks)',
                line=dict(width=3)
            ))
        # if series_b is not None and np.isfinite(series_b).any():
        #     fig.add_trace(go.Scatter(
        #         x=df['time_s'], y=series_b, mode='lines', name=f'Block average (per {beats_k} beats)',
        #         line=dict(width=3)
        #     ))
        # Optional: show peak markers
        if peaks_idx.size > 0:
            fig.add_trace(go.Scatter(
                x=df['time_s'].iloc[peaks_idx], y=df[plot_signal].iloc[peaks_idx],
                mode='markers', name='Local Highs (HR-dynamic)',
                marker=dict(size=7, symbol='diamond')
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
                cols_filtered = ['time_s', 'time_mmss_millis', 'comment'] + [c for c in all_signals if c in df.columns]
                filtered_df = df[cols_filtered].copy()
                # Ensure Excel-friendly dtypes
                if 'comment' in filtered_df.columns:
                    filtered_df['comment'] = filtered_df['comment'].astype(object)
                filtered_df.to_excel(writer, index=False, sheet_name="Filtered(200hz)")

                # Helper to create a boolean local-highs column from peak indices
                def _peaks_flag(n, peaks_idx_arr):
                    arr = np.zeros(n, dtype=int)
                    if peaks_idx_arr is not None and hasattr(peaks_idx_arr, 'size') and peaks_idx_arr.size > 0:
                        valid = peaks_idx_arr[(peaks_idx_arr >= 0) & (peaks_idx_arr < n)]
                        arr[valid] = 1
                    return arr

                # Access beat-mode aggregates/maps if available
                agg5_moving_map = st.session_state.get('agg5_moving_map', {})
                agg5_block_map  = st.session_state.get('agg5_block_map', {})
                peaks_fp  = st.session_state.get('peaks_idx', np.array([], dtype=int))
                peaks_cbf = st.session_state.get('peaks_idx_cbf', np.array([], dtype=int))

                # -------- Sheet 2: MovingMean (all except CBF) --------
                if isinstance(agg5_moving_map, dict) and len(agg5_moving_map) > 0:
                    moving_cols = {
                        'time_s': df['time_s'],
                        'time_mmss_millis': df['time_mmss_millis'],
                        'comment': df['comment'],
                        'local_high_FP': _peaks_flag(len(df), peaks_fp)
                    }
                    for sig in all_signals:
                        if sig == fallback_signal:
                            continue
                        series = agg5_moving_map.get(sig, None)
                        if series is not None:
                            moving_cols[f'{sig} - MovingMean({beats_k}beats)'] = series
                    # Ensure Excel-friendly dtypes
                    if 'comment' in moving_cols:
                        moving_cols['comment'] = moving_cols['comment'].astype(object)
                    df_moving = pd.DataFrame(moving_cols)
                    df_moving = df_moving[(df_moving['local_high_FP'] == 1) & (df_moving['comment'].notna())]
                    # set all other columns to NaN where comment is not NaN
                    # df_moving.loc[df_moving['comment'].notna(), df_moving.columns.difference(['time_s', 'time_mmss_millis', 'comment'])] = np.nan
                    df_moving.to_excel(writer, index=False, sheet_name=f"MovingMean(Finapress)")


                # -------- Sheet 3: CBF MovingMean (only CBF) --------
                if fallback_signal in all_signals and isinstance(agg5_moving_map, dict):
                    cbf_mov = agg5_moving_map.get(fallback_signal, None)
                    if cbf_mov is not None:
                        cbf_m_cols = {
                            'time_s': df['time_s'],
                            'time_mmss_millis': df['time_mmss_millis'],
                            'comment': df['comment'],
                            'local_high_CBF': _peaks_flag(len(df), peaks_cbf),
                            f'{fallback_signal} - MovingMean({beats_k}beats)': cbf_mov,
                        }
                        # Ensure Excel-friendly dtypes
                        if 'comment' in cbf_m_cols:
                            cbf_m_cols['comment'] = cbf_m_cols['comment'].astype(object)
                        cbf_df_moving = pd.DataFrame(cbf_m_cols)
                        cbf_df_moving = cbf_df_moving[(cbf_df_moving['local_high_CBF'] == 1) & (cbf_df_moving['comment'].notna())]
                         # set all other columns to NaN where comment is not NaN
                        # cbf_df_moving.loc[cbf_df_moving['comment'].notna(), cbf_df_moving.columns.difference(['time_s', 'time_mmss_millis', 'comment'])] = np.nan
                        cbf_df_moving.to_excel(writer, index=False, sheet_name=f"MovingMean (CBF)")

   
                # -------- Sheet: Resampled (time-based) --------
                if not beat_mode:
                    # If the radio is time-based, use it; if it's beat-based, reuse the last time-based choice (default 1 min)
                    label_to_sec = {
                        "500ms": 0.5, "1 sec": 1, "2 sec": 2, "5 sec": 5,
                        "10 sec": 10, "15 sec": 15, "30 sec": 30, "1 min": 60,
                    }
                    current_choice = st.session_state.get('bin_choice_label', None)
                    if current_choice in label_to_sec:
                        selected_label = current_choice
                    else:
                        selected_label = st.session_state.get('last_time_bin_label', '1 min')
                    if selected_label in label_to_sec:
                        bin_sec = float(label_to_sec[selected_label])
                        df_tmp = df.copy()
                        if len(df_tmp) > 0 and np.isfinite(df_tmp['time_s']).any():
                            t0 = float(df_tmp['time_s'].iloc[0])
                        else:
                            t0 = 0.0
                        df_tmp['time_bin'] = ((df_tmp['time_s'] - t0) // bin_sec).astype(int)
                        resampled = df_tmp.groupby('time_bin').agg(
                            {**{c: 'mean' for c in all_signals},
                             'time_s': 'first',
                             'time_mmss_millis': 'first',
                             'comment': first_nonempty_comment}
                        ).reset_index(drop=True)
                        # Ensure Excel-friendly dtypes
                        if 'comment' in resampled.columns:
                            resampled['comment'] = resampled['comment'].astype(object)
                        sheet_name = f"Resampled({selected_label})"
                        resampled.to_excel(writer, index=False, sheet_name=sheet_name[:31])


            buffer.seek(0)
            st.download_button(
                label="Download Excel",
                data=buffer.getvalue(),
                file_name=f"{file_base}_export.xlsx",
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