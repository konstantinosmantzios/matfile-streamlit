import numpy as np
import pandas as pd
import scipy.io
import streamlit as st
import plotly.graph_objects as go
from export import generate_excel_report, extract_stats_for_signal
import io
import os
from datetime import datetime, timedelta

# ---------------------------------------------------------
# Page Configurations
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="MAT File Converter & Resampling Tool")

from processing import (
    first_nonempty_comment,
    rolling_median_np,
    prepare_hr_for_window,
    compute_win_half_from_hr,
    find_local_high_indices,
    channel_info_df,
    comments_df,
    extract_channel_signals_with_comments,
    apply_savgol_filter,
    apply_butter_lowpass,
    apply_hampel_filter,
    find_autocal_column
)

# ---------------------------------------------------------
# Cached Data Loading & Processing Functions
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_mat_file(file_bytes: bytes):
    """Load and parse a MAT file once per upload. Results cached by file content hash."""
    mat = scipy.io.loadmat(io.BytesIO(file_bytes), squeeze_me=True)
    df_ci = channel_info_df(mat)
    df_com = comments_df(mat, df_ci)
    df_extracted = extract_channel_signals_with_comments(mat, df_com)
    n_blks = int(len(np.atleast_1d(mat["blocktimes"])))
    return df_ci, df_com, df_extracted, n_blks


@st.cache_data(show_spinner=False)
def compute_processed_signals(
    df: pd.DataFrame,
    all_signals: tuple,
    priority_signal: str,
    fallback_signal: str,
    main_signal: str,
    auto_cal_option: str,
    use_channel_masking: bool,
    autocal_col,
    df_comments: pd.DataFrame,
    df_channel_info: pd.DataFrame,
    n_blocks: int,
    filter_method_fp: str,
    fp_savgol_win: int, fp_savgol_poly: int,
    fp_butter_cutoff: float, fp_butter_order: int,
    fp_hampel_win: int, fp_hampel_sig: float,
    filter_method_cbf: str,
    cbf_savgol_win: int, cbf_savgol_poly: int,
    cbf_butter_cutoff: float, cbf_butter_order: int,
    cbf_hampel_win: int, cbf_hampel_sig: float,
    beat_mode: bool,
    beats_k: int,
    bin_sec: float,
):
    """Apply filters and resampling. Cached: only re-runs when filter/resample params change."""
    all_signals_list = list(all_signals)
    df_raw = df.sort_values('time_s').copy()
    df_sorted = df.sort_values('time_s').copy()

    # --- APPLY AUTOCAL MASK ---
    mask = np.zeros(len(df_sorted), dtype=bool)
    if use_channel_masking and autocal_col is not None:
        try:
            temp_col = pd.to_numeric(df_sorted[autocal_col], errors='coerce')
            # When AutoCal is ON, we ALWAYS mask when it is near 0.
            mask = temp_col < 0.5
        except Exception:
            mask = (df_sorted[autocal_col] == 0) | (df_sorted[autocal_col] == False)
    else:
        # When AutoCal is OFF, rely on HCU comments and mask until the next spike (temp_col >= 0.5)
        if df_comments is not None and not df_comments.empty and autocal_col is not None:
            block_lengths = []
            for _, row in df_channel_info.iterrows():
                if row["title"] == autocal_col and row.get("datastart") is not None:
                    block_lengths = [int(e - s + 1) for s, e in zip(row["datastart"], row["dataend"])]
                    break
            if not block_lengths:
                block_lengths = [len(df_sorted) // n_blocks] * n_blocks
            block_offsets = np.cumsum([0] + block_lengths[:-1])

            matches = df_comments[df_comments["comment_text"] == "HCU not connected"]
            temp_col = pd.to_numeric(df_sorted[autocal_col], errors='coerce').fillna(0)
            is_spike = (temp_col >= 0.5).values

            for _, ev in matches.iterrows():
                block_index = int(ev["block_index"]) - 1
                sample_index = int(ev["sample_index"])
                if 0 <= block_index < len(block_offsets):
                    idx_global = block_offsets[block_index] + sample_index
                    future_spikes = np.where(is_spike[idx_global:])[0]
                    if len(future_spikes) > 0:
                        end_idx = idx_global + future_spikes[0]
                        mask[idx_global:end_idx] = True
                    else:
                        fs_fb = float(df_channel_info[df_channel_info["title"] == all_signals_list[0]]["samplerate"].iloc[0]) if not df_channel_info.empty else 1000.0
                        end_idx = min(len(df_sorted), idx_global + int(15 * fs_fb))
                        mask[idx_global:end_idx] = True

    # --- APPLY NUMERIC CONVERSION & AUTOCAL MASK TO SIGNALS ---
    for col in all_signals_list:
        if col != fallback_signal and col != autocal_col:
            df_sorted[col] = pd.to_numeric(df_sorted[col], errors='coerce').astype('float64')
            # Only apply the autocal masking to the Finger Pressure signal
            if col == priority_signal:
                df_sorted.loc[mask, col] = np.nan

    for bp_col in ["2: MAP", "3: Systolic", "4: Diastolic"]:
        if bp_col in df_sorted.columns:
            df_sorted[bp_col] = df_sorted[bp_col].replace(0, np.nan)

    # --- APPLY FINGER PRESSURE FILTER ---
    if main_signal == priority_signal:
        df_sorted[main_signal] = pd.to_numeric(df_sorted[main_signal], errors='coerce').astype('float64')
        fs = float(df_channel_info[df_channel_info["title"] == main_signal]["samplerate"].iloc[0]) if not df_channel_info[df_channel_info["title"] == main_signal].empty else 1000.0
        if filter_method_fp == "Savitzky-Golay":
            df_sorted[main_signal] = apply_savgol_filter(df_sorted[main_signal].values, window_length=fp_savgol_win, polyorder=fp_savgol_poly)
        elif filter_method_fp == "Butterworth Low-Pass":
            df_sorted[main_signal] = apply_butter_lowpass(df_sorted[main_signal].values, cutoff_freq=fp_butter_cutoff, fs=fs, order=fp_butter_order)
        elif filter_method_fp == "Hampel (Outlier Removal)":
            df_sorted[main_signal] = apply_hampel_filter(df_sorted[main_signal].values, window_size=fp_hampel_win, n_sigmas=fp_hampel_sig)

    # --- APPLY CBF FILTER ---
    if fallback_signal in all_signals_list:
        df_sorted[fallback_signal] = pd.to_numeric(df_sorted[fallback_signal], errors='coerce').astype('float64')
        fs_cbf = float(df_channel_info[df_channel_info["title"] == fallback_signal]["samplerate"].iloc[0]) if not df_channel_info[df_channel_info["title"] == fallback_signal].empty else 1000.0
        if filter_method_cbf == "Savitzky-Golay":
            df_sorted[fallback_signal] = apply_savgol_filter(df_sorted[fallback_signal].values, window_length=cbf_savgol_win, polyorder=cbf_savgol_poly)
        elif filter_method_cbf == "Butterworth Low-Pass":
            df_sorted[fallback_signal] = apply_butter_lowpass(df_sorted[fallback_signal].values, cutoff_freq=cbf_butter_cutoff, fs=fs_cbf, order=cbf_butter_order)
        elif filter_method_cbf == "Hampel (Outlier Removal)":
            df_sorted[fallback_signal] = apply_hampel_filter(df_sorted[fallback_signal].values, window_size=cbf_hampel_win, n_sigmas=cbf_hampel_sig)

    # Ensure all signals are float64 before resampling
    for col in all_signals_list:
        df_sorted[col] = pd.to_numeric(df_sorted[col], errors='coerce').astype('float64')

    # --- RESAMPLING ---
    result_df = None
    agg5_moving_map = None
    peaks = np.array([], dtype=int)
    peaks_cbf = np.array([], dtype=int)

    if not beat_mode:
        # Time-based resampling
        df_tmp = df_sorted.copy()
        t0 = float(df_tmp['time_s'].iloc[0]) if len(df_tmp) > 0 and np.isfinite(df_tmp['time_s']).any() else 0.0
        df_tmp['time_bin'] = ((df_tmp['time_s'] - t0) // bin_sec).astype(int)
        result_df = df_tmp.groupby('time_bin').agg(
            {**{c: 'mean' for c in all_signals_list},
             'absolute_time': 'first',
             'time_s': 'first',
             'time_mmss_millis': 'first',
             'comment': first_nonempty_comment}
        ).reset_index(drop=True)
    else:
        # Dynamic beat-based resampling
        ts = df_sorted['time_s'].values
        fs0 = 1.0 / float(np.nanmedian(np.diff(ts))) if len(ts) > 1 else 200.0

        sig0_raw = pd.to_numeric(df_sorted[main_signal], errors='coerce').astype('float64').values
        sig0_filt = rolling_median_np(sig0_raw, window=5)

        hr = pd.to_numeric(df_sorted["5: HR"], errors='coerce').astype('float64').values if "5: HR" in df_sorted.columns else np.full_like(sig0_filt, 60.0)
        hr_for_win = prepare_hr_for_window(hr.copy(), roll_win=1000)
        hr_for_win = np.where(np.isfinite(hr_for_win), hr_for_win, 60.0)
        win_half = compute_win_half_from_hr(hr_for_win, fs0, factor=1.3, min_samples=3)

        agg5_moving_map = {sig_name: np.full(len(df_sorted), np.nan, dtype=float) for sig_name in all_signals_list}

        if "block_index" not in df_sorted.columns:
            df_sorted["block_index"] = 0
        block_ids = df_sorted["block_index"].values
        unique_blocks = np.unique(block_ids)

        for b_id in unique_blocks:
            b_mask = (block_ids == b_id)
            b_indices = np.where(b_mask)[0]
            if len(b_indices) == 0:
                continue

            b_offset = b_indices[0]
            sig0_b = sig0_filt[b_mask]
            win_half_b = win_half[b_mask]

            p_b = find_local_high_indices(sig0_b, win_half_b)
            if len(p_b) > 0:
                peaks = np.concatenate([peaks, p_b + b_offset])

            if fallback_signal in all_signals_list:
                sig_cbf_b = rolling_median_np(pd.to_numeric(df_sorted[fallback_signal].values[b_mask], errors='coerce').astype('float64'), window=5)
                p_cbf_b = find_local_high_indices(sig_cbf_b, win_half_b)
                if len(p_cbf_b) > 0:
                    peaks_cbf = np.concatenate([peaks_cbf, p_cbf_b + b_offset])
            else:
                p_cbf_b = np.array([], dtype=int)

            for sig_name in all_signals_list:
                y_b = pd.to_numeric(df_sorted[sig_name].values[b_mask], errors='coerce').astype('float64')
                peaks_used = p_cbf_b if (sig_name == fallback_signal and p_cbf_b.size > 0) else p_b

                agg_m_b = np.full(len(y_b), np.nan, dtype=float)
                if peaks_used.size >= beats_k:
                    nP = peaks_used.size
                    mids_used = np.rint((peaks_used[:-1] + peaks_used[1:]) / 2.0).astype(int) if nP > 1 else np.array([], dtype=int)
                    half_k = beats_k // 2
                    for i in range(half_k, nP - (beats_k - half_k - 1)):
                        i0, i1 = peaks_used[i - half_k], peaks_used[i + (beats_k - half_k - 1)]
                        seg_vals = y_b[i0:i1 + 1]
                        mval = np.nan if np.isnan(seg_vals).all() else float(np.nanmean(seg_vals))
                        seg_start = 0 if i == 0 else int(mids_used[i - 1])
                        seg_end = len(y_b) if i == (nP - 1) else int(mids_used[i])
                        agg_m_b[seg_start:seg_end] = mval

                # Set the last sample of the block to NaN to break Plotly lines across huge gaps
                if b_id != unique_blocks[-1] and len(agg_m_b) > 0:
                    agg_m_b[-1] = np.nan

                agg5_moving_map[sig_name][b_mask] = agg_m_b

    return df_raw, df_sorted, result_df, agg5_moving_map, peaks, peaks_cbf


# ---------------------------------------------------------
# Streamlit Interface Layout
# ---------------------------------------------------------

st.title("MAT Resampler & Stats Analyzer")
st.write("Convert, filter, and extract physiological standing response statistics.")

with st.sidebar:
    uploaded_mat = st.file_uploader("Upload a MATLAB .mat file to begin", type=["mat"])

if uploaded_mat is not None:
    if st.session_state.get("last_uploaded_name") != uploaded_mat.name:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.last_uploaded_name = uploaded_mat.name
        st.rerun()

if uploaded_mat:
    _file_bytes = uploaded_mat.read()
    with st.spinner("Loading MAT file..."):
        df_channel_info, df_comments, df, n_blocks = load_mat_file(_file_bytes)
    all_columns = list(df.columns)

    st.session_state.df_comments = df_comments

    priority_signal = '1: Finger Pressure'
    fallback_signal = '6: CBF'
    
    def is_valid_signal(df, col, min_ratio=0.1):
        if col not in df.columns:
            return False
        vals = df[col].values
        valid_count = np.isfinite(vals).sum()
        return valid_count > 0 and valid_count / len(vals) >= min_ratio

    desired_signals = ['finger pressure', 'map', 'systolic', 'diastolic', 'cbf']
    if is_valid_signal(df, priority_signal):
        all_signals = [c for c in df.columns if c not in ['time_s', 'absolute_time', 'time_mmss_millis', 'comment'] and any(d in c.lower() for d in desired_signals)]
        main_signal = priority_signal
    elif is_valid_signal(df, fallback_signal):
        all_signals = [c for c in df.columns if c not in ['time_s', 'absolute_time', 'time_mmss_millis', 'comment'] and any(d in c.lower() for d in desired_signals)]
        main_signal = fallback_signal
        st.sidebar.warning(f"'{priority_signal}' is missing/empty. Only '{fallback_signal}' will be processed.")
    else:
        st.error("No suitable signals found with data ('1: Finger Pressure' or '6: CBF').")
        st.stop()

    # Sidebar settings
    st.sidebar.markdown("### Conversion Settings")
    resample_mode = st.sidebar.radio("Resampling Mode", ["Time-based", "Beat-based"], index=1, key="resample_mode")
    
    if resample_mode == "Time-based":
        time_options = ["500ms", "1 sec", "2 sec", "5 sec", "10 sec", "15 sec", "30 sec", "1 min"]
        bin_choice = st.sidebar.select_slider("Resampling Rate", options=time_options, value="1 sec", key="bin_choice")
        beat_mode = False
        beats_k = None
    else:
        beat_options = [1, 2, 3, 5, 10, 15, 20, 30]
        beats_k = st.sidebar.select_slider("Resampling Rate (Beats)", options=beat_options, value=5, key="beats_k")
        bin_choice = f"{beats_k}beats"
        beat_mode = True

    st.session_state['bin_choice_label'] = bin_choice
    
    bin_map = {
        "500ms": 0.5, "1 sec": 1, "2 sec": 2, "5 sec": 5,
        "10 sec": 10, "15 sec": 15, "30 sec": 30, "1 min": 60
    }
    
    st.sidebar.markdown("### Filters (Phase 2)")
    
    # Debugging Options
    debug_mode = st.sidebar.checkbox("Enable Debugging", value=False, key="debug_mode")
    
    # AutoCal Filter
    if debug_mode:
        auto_cal_option = st.sidebar.radio(
            "AutoCal Noise Removal",
            ["Auto-Detect", "Force Enabled (Channel)", "Force Disabled (Comments)"],
            index=0,
            key="auto_cal_option",
            help=(
                "**Auto-Detect**: Automatically filters calibration artifacts.\n\n"
                "**Force Enabled**: When the researcher used the setting.\n\n"
                "**Force Disabled**: When the setting was not used."
            )
        )
    else:
        auto_cal_option = "Auto-Detect"
    
    # Finger Pressure Filter Sandbox
    st.sidebar.markdown(f"**{priority_signal} Filter Sandbox**")
    filter_method_fp = st.sidebar.selectbox(
        "Select Filter Algorithm (FP)",
        ["None", "Savitzky-Golay", "Butterworth Low-Pass", "Hampel (Outlier Removal)"],
        key="filter_method_fp"
    )
    
    fp_savgol_win, fp_savgol_poly = 51, 3
    fp_butter_cutoff, fp_butter_order = 5.0, 4
    fp_hampel_win, fp_hampel_sig = 5, 3.0
    
    if filter_method_fp == "Savitzky-Golay":
        fp_savgol_win = st.sidebar.slider("Window Length (Odd) [FP]", 5, 201, 51, step=2, key="fp_savgol_win", help="Larger window increases smoothing but reduces peak amplitude.")
        fp_savgol_poly = st.sidebar.slider("Polynomial Order [FP]", 1, 10, 5, key="fp_savgol_poly", help="Higher order preserves narrow peaks better.")
    elif filter_method_fp == "Butterworth Low-Pass":
        fp_butter_cutoff = st.sidebar.slider("Cutoff Frequency (Hz) [FP]", 0.5, 20.0, 5.0, step=0.5, key="fp_butter_cutoff", help="Frequencies above this value are attenuated. Lower = smoother.")
        fp_butter_order = st.sidebar.slider("Filter Order [FP]", 1, 10, 4, key="fp_butter_order", help="Higher order creates a steeper frequency cutoff.")
    elif filter_method_fp == "Hampel (Outlier Removal)":
        fp_hampel_win = st.sidebar.slider("Rolling Window Size [FP]", 3, 50, 5, key="fp_hampel_win", help="Number of neighbors on each side to compute median.")
        fp_hampel_sig = st.sidebar.slider("Sigma Threshold [FP]", 1.0, 10.0, 3.0, step=0.5, key="fp_hampel_sig", help="Number of standard deviations to classify an outlier.")

    # CBF Filter Sandbox
    st.sidebar.markdown(f"**{fallback_signal} Filter Sandbox**")
    filter_method_cbf = st.sidebar.selectbox(
        "Select Filter Algorithm (CBF)",
        ["None", "Savitzky-Golay", "Butterworth Low-Pass", "Hampel (Outlier Removal)"],
        key="filter_method_cbf"
    )
    
    cbf_savgol_win, cbf_savgol_poly = 51, 3
    cbf_butter_cutoff, cbf_butter_order = 5.0, 4
    cbf_hampel_win, cbf_hampel_sig = 5, 3.0
    
    if filter_method_cbf == "Savitzky-Golay":
        cbf_savgol_win = st.sidebar.slider("Window Length (Odd) [CBF]", 5, 201, 51, step=2, key="cbf_savgol_win", help="Larger window increases smoothing but reduces peak amplitude.")
        cbf_savgol_poly = st.sidebar.slider("Polynomial Order [CBF]", 1, 10, 5, key="cbf_savgol_poly", help="Higher order preserves narrow peaks better.")
    elif filter_method_cbf == "Butterworth Low-Pass":
        cbf_butter_cutoff = st.sidebar.slider("Cutoff Frequency (Hz) [CBF]", 0.5, 20.0, 5.0, step=0.5, key="cbf_butter_cutoff", help="Frequencies above this value are attenuated. Lower = smoother.")
        cbf_butter_order = st.sidebar.slider("Filter Order [CBF]", 1, 10, 4, key="cbf_butter_order", help="Higher order creates a steeper frequency cutoff.")
    elif filter_method_cbf == "Hampel (Outlier Removal)":
        cbf_hampel_win = st.sidebar.slider("Rolling Window Size [CBF]", 3, 50, 5, key="cbf_hampel_win", help="Number of neighbors on each side to compute median.")
        cbf_hampel_sig = st.sidebar.slider("Sigma Threshold [CBF]", 1.0, 10.0, 3.0, step=0.5, key="cbf_hampel_sig", help="Number of standard deviations to classify an outlier.")
    
    # ----- AutoCal detection (outside cache so sidebar messages always display) -----
    _df_for_detection = df.sort_values('time_s')
    autocal_col = find_autocal_column(_df_for_detection)
    use_channel_masking = False
    if auto_cal_option == "Force Enabled (Channel)":
        use_channel_masking = True
        st.sidebar.info("AutoCal mask manually enabled.")
    elif auto_cal_option == "Auto-Detect" and autocal_col is not None:
        try:
            _temp_detect = pd.to_numeric(_df_for_detection[autocal_col], errors='coerce')
            _perc_below = (_temp_detect < 0.5).mean()
            if _perc_below > 0.95:
                use_channel_masking = False
                st.sidebar.info(f"Auto-Detect: AutoCal was OFF ({_perc_below*100:.1f}% resting). Relying on comments only.")
            else:
                use_channel_masking = True
                st.sidebar.success(f"Auto-Detect: AutoCal was ON ({_perc_below*100:.1f}% active noise). Using precision signal masking.")
        except Exception:
            use_channel_masking = False

    # Compute bin_sec for time-based mode (unused in beat mode)
    bin_sec = float(bin_map.get(bin_choice, 1)) if not beat_mode else 0.0

    # ----- Cached processing: filters + resampling -----
    with st.spinner("Processing & resampling signal vectors..."):
        df_raw, df_filtered, result_df, agg5_moving_map, peaks, peaks_cbf = compute_processed_signals(
            df=df,
            all_signals=tuple(all_signals),
            priority_signal=priority_signal,
            fallback_signal=fallback_signal,
            main_signal=main_signal,
            auto_cal_option=auto_cal_option,
            use_channel_masking=use_channel_masking,
            autocal_col=autocal_col,
            df_comments=df_comments,
            df_channel_info=df_channel_info,
            n_blocks=n_blocks,
            filter_method_fp=filter_method_fp,
            fp_savgol_win=fp_savgol_win, fp_savgol_poly=fp_savgol_poly,
            fp_butter_cutoff=fp_butter_cutoff, fp_butter_order=fp_butter_order,
            fp_hampel_win=fp_hampel_win, fp_hampel_sig=fp_hampel_sig,
            filter_method_cbf=filter_method_cbf,
            cbf_savgol_win=cbf_savgol_win, cbf_savgol_poly=cbf_savgol_poly,
            cbf_butter_cutoff=cbf_butter_cutoff, cbf_butter_order=cbf_butter_order,
            cbf_hampel_win=cbf_hampel_win, cbf_hampel_sig=cbf_hampel_sig,
            beat_mode=beat_mode,
            beats_k=beats_k if beats_k is not None else 1,
            bin_sec=bin_sec,
        )

    # Populate session_state from cached results
    st.session_state.df_raw = df_raw
    st.session_state.df = df_filtered
    st.session_state.all_signals = all_signals
    if beat_mode:
        st.session_state.beat_mode = True
        st.session_state.agg5_moving_map = agg5_moving_map
        st.session_state.peaks_idx = peaks
        st.session_state.peaks_idx_cbf = peaks_cbf
    else:
        st.session_state.beat_mode = False
        st.session_state.result_df = result_df

    # Display Visualization
    st.subheader("Data Visualization")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        viz_mode = st.radio("Visualization Mode", ["Filtering Preview", "Supine to Standing Analysis"], index=1, horizontal=True, key="viz_mode")
    with col2:
        show_comments = st.checkbox("Show Comments", value=True, key="show_comments")
        
    if viz_mode == "Supine to Standing Analysis":
        with st.expander("Analysis Settings", expanded=True):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                baseline_window = st.number_input("Baseline (s)", min_value=10, max_value=600, value=60, step=10, help="Duration of the baseline period prior to the standing transition.")
            with col_b:
                end_marker_window = st.number_input("End Marker (s)", min_value=1, max_value=300, value=10, step=1, help="Time after the standing comment to place the End marker.")
            with col_c:
                st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
                use_baseline_area = st.checkbox("Area from Baseline", value=False, key="use_baseline_area", help="Calculate shaded areas relative to the Baseline Mean instead of the marker values.")
        plot_height = st.slider("Main Plot Height (px)", min_value=300, max_value=2000, value=800, step=50, key="plot_height", help="Adjust the height of the signal plot area.")
    else:
        baseline_window = 60
        end_marker_window = 10
        use_baseline_area = False
        plot_height = 600
        
    plot_signal = st.selectbox("Select Signal Waveform to Plot", all_signals, index=0, key="plot_signal")
    
    if viz_mode == "Filtering Preview":
        raw_vis = True
        filt_vis = True
        peak_vis = True
    else: # "Supine to Standing Analysis"
        raw_vis = 'legendonly'
        filt_vis = 'legendonly'
        peak_vis = 'legendonly'
    
    # Prepare x and y for raw and resampled data
    raw_x = st.session_state.df_raw['time_s'].values
    raw_t = st.session_state.df_raw['absolute_time'].values
    raw_y = st.session_state.df_raw[plot_signal].values
    filt_y = st.session_state.df[plot_signal].values
    
    resampled_label = st.session_state.get('bin_choice_label', 'Resampled Data')
    
    if st.session_state.beat_mode:
        resampled_y = st.session_state.agg5_moving_map.get(plot_signal, np.array([]))
        resampled_x = raw_x
        resampled_t = raw_t
    else:
        resampled_x = st.session_state.result_df['time_s'].values
        resampled_t = st.session_state.result_df['absolute_time'].values
        resampled_y = st.session_state.result_df[plot_signal].values
        
    # # Downsample for Plotly visualization to avoid out-of-memory errors on Streamlit Cloud
    # max_plot_pts = 20000
    # step_raw = len(raw_t) // max_plot_pts if len(raw_t) > max_plot_pts else 1
    # step_res = len(resampled_t) // max_plot_pts if len(resampled_t) > max_plot_pts else 1
    
    fig = go.Figure()
    
    # Plot Raw Trace (Background)
    fig.add_trace(go.Scattergl(
        x=raw_t, y=raw_y, mode='lines', name='Raw Data (Unfiltered)',
        # x=raw_t[::step_raw], y=raw_y[::step_raw], mode='lines', name='Raw Data (Unfiltered)',

        line=dict(color='rgba(156,163,175,0.4)', width=1),
        visible=raw_vis
    ))
    
    # Plot Filtered Trace (Foreground)
    fig.add_trace(go.Scattergl(
        x=raw_t, y=filt_y, mode='lines', name='Filtered Data',
        # x=raw_t[::step_raw], y=filt_y[::step_raw], mode='lines', name='Filtered Data',

        line=dict(color='#38bdf8', width=1),
        visible=filt_vis
    ))
    
    # Plot Resampled Trace
    fig.add_trace(go.Scattergl(
        x=resampled_t, y=resampled_y, mode='lines', name=f'Resampled ({resampled_label})',  
        # x=resampled_t[::step_res], y=resampled_y[::step_res], mode='lines', name=f'Resampled ({resampled_label})',

        line=dict(color='#f97316', width=2)
    ))
    
    # Plot Peaks if in beat mode (only for FP and CBF)
    if st.session_state.beat_mode:
        if plot_signal in [priority_signal, fallback_signal]:
            peaks_to_plot = st.session_state.peaks_idx_cbf if (plot_signal == fallback_signal and st.session_state.peaks_idx_cbf.size > 0) else st.session_state.peaks_idx
            peaks_to_plot = peaks_to_plot[(peaks_to_plot >= 0) & (peaks_to_plot < len(raw_x))]
            if peaks_to_plot.size > 0:
                fig.add_trace(go.Scattergl(
                    x=raw_t[peaks_to_plot], 
                    y=filt_y[peaks_to_plot],
                    mode='markers', 
                    name='Detected Peaks',
                    marker=dict(color='#ef4444', size=5, symbol='x'),
                    visible=peak_vis
                ))
            
    if show_comments:
        # Add comment annotations from original DF
        comments_data = st.session_state.df[st.session_state.df['comment'] != ""]
        if not comments_data.empty:
            comment_list = comments_data[['time_s', 'absolute_time', 'comment']].to_dict('records')

            # Full transition analysis only in Supine to Standing mode
            if viz_mode == "Supine to Standing Analysis":
                i = 0
                while i < len(comment_list):
                    c1 = comment_list[i]
                    if "transition" in str(c1['comment']).lower():
                        # Valid test: next comment must be a 'stand' / 'standing'
                        if i + 1 < len(comment_list) and str(comment_list[i+1]['comment']).lower().strip(" .") in ["stand", "standing"]:
                            c_stand = comment_list[i+1]

                            # Yellow background for the transition period (transition → standing)
                            fig.add_vrect(
                                x0=c1['absolute_time'], x1=c_stand['absolute_time'],
                                fillcolor="rgba(250, 204, 21, 0.1)",
                                opacity=1.0, layer="below", line_width=0,
                                annotation_text="Transition",
                                annotation_position="top left",
                                annotation_font_size=12,
                                annotation_font_color="rgba(250, 204, 21, 0.8)"
                            )

                            # Compute baseline before transition
                            t_start = c1['time_s']
                            t_base_start = max(0, t_start - baseline_window)
                            mask_base = (resampled_x >= t_base_start) & (resampled_x <= t_start)
                            t_base_start_idx = np.abs(resampled_x - t_base_start).argmin()
                            y_window = resampled_y[mask_base]
                            baseline_mean = None
                            if y_window.size > 0:
                                valid_y = y_window[np.isfinite(y_window)]
                                if valid_y.size > 0:
                                    baseline_mean = np.mean(valid_y)

                            # Draw baseline segment
                            if baseline_mean is not None:
                                fig.add_trace(go.Scatter(
                                    x=[pd.to_datetime(resampled_t[t_base_start_idx]), pd.to_datetime(c1['absolute_time'])],
                                    y=[baseline_mean, baseline_mean],
                                    mode='lines+text', name='Baseline',
                                    line=dict(color='#3b82f6', width=3),
                                    textposition="top right",
                                    textfont=dict(color='#3b82f6', size=14),
                                    showlegend=False, hoverinfo='skip'
                                ))

                            # Draw orange marker at the transition comment
                            trans_idx = np.abs(resampled_x - t_start).argmin()
                            if trans_idx < len(resampled_y) and np.isfinite(resampled_y[trans_idx]):
                                fig.add_trace(go.Scatter(
                                    x=[pd.to_datetime(resampled_t[trans_idx])], y=[resampled_y[trans_idx]],
                                    mode='markers', name='Transition Start',
                                    marker=dict(color='#f97316', size=13, symbol='square'),
                                    showlegend=False, hoverinfo='skip'
                                ))

                            # Draw green marker at the standing comment
                            t_stand = c_stand['time_s']
                            start_idx = np.abs(resampled_x - t_stand).argmin()
                            if start_idx < len(resampled_y) and np.isfinite(resampled_y[start_idx]):
                                fig.add_trace(go.Scatter(
                                    x=[pd.to_datetime(resampled_t[start_idx])], y=[resampled_y[start_idx]],
                                    mode='markers', name='Start',
                                    marker=dict(color='#22c55e', size=13, symbol='circle'),
                                    showlegend=False, hoverinfo='skip'
                                ))

                            # Draw End marker
                            t_end_marker = t_stand + end_marker_window
                            end_idx = np.abs(resampled_x - t_end_marker).argmin()
                            if end_idx < len(resampled_y) and np.isfinite(resampled_y[end_idx]):
                                fig.add_trace(go.Scatter(
                                    x=[pd.to_datetime(resampled_t[end_idx])], y=[resampled_y[end_idx]],
                                    mode='markers+text', name='End',
                                    marker=dict(color='#ef4444', size=13, symbol='x'),
                                    text=["End"], textposition="top center",
                                    textfont=dict(color='#ef4444', size=12),
                                    showlegend=False, hoverinfo='skip'
                                ))

                            area_above_or, area_below_or = np.nan, np.nan
                            min_val_or, pct_drop_or = np.nan, np.nan
                            if trans_idx < len(resampled_y) and np.isfinite(resampled_y[trans_idx]):
                                area_mask = (resampled_x >= t_start) & (resampled_x <= t_end_marker)
                                x_area = resampled_x[area_mask]
                                t_area = resampled_t[area_mask]
                                y_area = resampled_y[area_mask]
                                if len(x_area) > 0:
                                    upper_y = baseline_mean if (use_baseline_area and baseline_mean is not None) else resampled_y[trans_idx]
                                    valid_mask = np.isfinite(y_area)
                                    x_val_or = x_area[valid_mask]
                                    t_val_or = t_area[valid_mask]
                                    y_val_or = y_area[valid_mask]
                                    if len(x_val_or) > 1:
                                        area_above_or = np.trapezoid(np.maximum(y_val_or - upper_y, 0), x_val_or)
                                        area_below_or = np.trapezoid(np.maximum(upper_y - y_val_or, 0), x_val_or)
                                        min_val_or = float(np.min(y_val_or))
                                        if upper_y != 0:
                                            pct_drop_or = (upper_y - min_val_or) / abs(upper_y) * 100
                                        min_idx_or = np.argmin(y_val_or)
                                        fig.add_trace(go.Scatter(
                                            x=[pd.to_datetime(t_val_or[min_idx_or])], y=[min_val_or],
                                            mode='markers', name='Min (Or)',
                                            marker=dict(color='#ef4444', size=10, symbol='diamond'),
                                            showlegend=False, hoverinfo='skip'
                                        ))
                                    fig.add_trace(go.Scatter(
                                        x=t_area, y=np.full(len(t_area), upper_y),
                                        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=t_area, y=y_area, mode='lines', line=dict(width=0),
                                        fill='tonexty', fillcolor='rgba(249, 115, 22, 0.15)',
                                        showlegend=False, hoverinfo='skip'
                                    ))

                            area_above_gr, area_below_gr = np.nan, np.nan
                            min_val_gr, pct_drop_gr = np.nan, np.nan
                            if start_idx < len(resampled_y) and np.isfinite(resampled_y[start_idx]):
                                area_mask_green = (resampled_x >= t_stand) & (resampled_x <= t_end_marker)
                                x_area_green = resampled_x[area_mask_green]
                                t_area_green = resampled_t[area_mask_green]
                                y_area_green = resampled_y[area_mask_green]
                                if len(x_area_green) > 0:
                                    upper_y_green = baseline_mean if (use_baseline_area and baseline_mean is not None) else resampled_y[start_idx]
                                    valid_mask_gr = np.isfinite(y_area_green)
                                    x_val_gr = x_area_green[valid_mask_gr]
                                    t_val_gr = t_area_green[valid_mask_gr]
                                    y_val_gr = y_area_green[valid_mask_gr]
                                    if len(x_val_gr) > 1:
                                        area_above_gr = np.trapezoid(np.maximum(y_val_gr - upper_y_green, 0), x_val_gr)
                                        area_below_gr = np.trapezoid(np.maximum(upper_y_green - y_val_gr, 0), x_val_gr)
                                        min_val_gr = float(np.min(y_val_gr))
                                        if upper_y_green != 0:
                                            pct_drop_gr = (upper_y_green - min_val_gr) / abs(upper_y_green) * 100
                                        min_idx_gr = np.argmin(y_val_gr)
                                        fig.add_trace(go.Scatter(
                                            x=[pd.to_datetime(t_val_gr[min_idx_gr])], y=[min_val_gr],
                                            mode='markers', name='Min (Gr)',
                                            marker=dict(color='#ef4444', size=10, symbol='diamond'),
                                            showlegend=False, hoverinfo='skip'
                                        ))
                                    fig.add_trace(go.Scatter(
                                        x=t_area_green, y=np.full(len(t_area_green), upper_y_green),
                                        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=t_area_green, y=y_area_green, mode='lines', line=dict(width=0),
                                        fill='tonexty', fillcolor='rgba(34, 197, 94, 0.25)',
                                        showlegend=False, hoverinfo='skip'
                                    ))

                            # Embedded summary table annotation
                            trans_val = resampled_y[trans_idx] if (trans_idx < len(resampled_y) and np.isfinite(resampled_y[trans_idx])) else np.nan
                            stand_val = resampled_y[start_idx] if (start_idx < len(resampled_y) and np.isfinite(resampled_y[start_idx])) else np.nan
                            end_val   = resampled_y[end_idx]   if (end_idx   < len(resampled_y) and np.isfinite(resampled_y[end_idx]))   else np.nan
                            base_val  = baseline_mean if baseline_mean is not None else np.nan
                            transition_duration = t_stand - t_start

                            def _fmt(v, decimals=2):
                                return f"{v:.{decimals}f}" if np.isfinite(v) else "—"

                            def _row(label, val, val_color="white"):
                                return (
                                    f"<span style='color:#9ca3af;'>{label}:</span>  "
                                    f"<span style='color:{val_color};'><b>{val}</b></span><br>"
                                )

                            table_html = (
                                "<b>📊 Transition Summary</b><br>"
                                "─────────────────────<br>"
                                + _row("Baseline", _fmt(base_val), "#60a5fa")
                                + _row("Transition Time", f"{transition_duration:.1f} s", "#facc15")
                                + "<br>"
                                + "<span style='color:#f97316;'><b>🟠 Transition to End</b></span><br>"
                                + _row("  Value at Trans. Start", _fmt(trans_val), "#f97316")
                                + _row("  Min / Drop", f"{_fmt(min_val_or)}  ({_fmt(pct_drop_or)}%)", "#ef4444")
                                + _row("  Area ↑ / ↓", f"{_fmt(area_above_or)} / {_fmt(area_below_or)}")
                                + "<br>"
                                + "<span style='color:#22c55e;'><b>🟢 Stand to End</b></span><br>"
                                + _row("  Value at Stand Start", _fmt(stand_val), "#22c55e")
                                + _row("  Min / Drop", f"{_fmt(min_val_gr)}  ({_fmt(pct_drop_gr)}%)", "#ef4444")
                                + _row("  Area ↑ / ↓", f"{_fmt(area_above_gr)} / {_fmt(area_below_gr)}")
                                + "<br>"
                                + "<span style='color:#ef4444;'><b>🔴 End Marker</b></span><br>"
                                + _row("  Value", _fmt(end_val), "#ef4444")
                            )
                            fig.add_annotation(
                                x=c1['absolute_time'], y=1.0, yref="paper",
                                text=table_html, showarrow=False, align="left",
                                bordercolor="rgba(255,255,255,0.15)", borderwidth=1, borderpad=10,
                                bgcolor="rgba(15, 15, 20, 0.92)",
                                xanchor="left", yanchor="top",
                                font=dict(size=12, color="white")
                            )

                    i += 1

            # Then, add the vertical lines and text annotations
            for _, row in comments_data.iterrows():
                comment_text = str(row['comment'])
                is_stand = comment_text.lower().strip(" .") in ["stand", "standing"]
                is_hcu = "hcu not connected" in comment_text.lower()
                
                if is_stand:
                    formatted_text = f"<b>{comment_text}</b>"
                    text_color = "rgba(255,255,255,1.0)"
                    line_color = "rgba(255,255,255,0.6)"
                elif is_hcu:
                    formatted_text = f"<b>{comment_text}</b>"
                    text_color = "#ef4444"  # Red
                    line_color = "#ef4444"
                else:
                    formatted_text = f"{comment_text}"
                    text_color = "rgba(180,180,180,1.0)"
                    line_color = "rgba(255,255,255,0.3)"

                fig.add_vline(x=row['absolute_time'], line_width=1, line_dash="dash", line_color=line_color)
                fig.add_annotation(
                    x=row['absolute_time'],
                    y=0.02,
                    yref="paper",
                    text=formatted_text,
                    showarrow=False,
                    textangle=-90,
                    xanchor="center",
                    yanchor="bottom",
                    font=dict(size=10, color=text_color)
                )
    
    # Plot autocal signal if debugging is on
    if debug_mode:
        autocal_col = find_autocal_column(st.session_state.df_raw)
        if autocal_col is not None:
            fig.add_trace(go.Scatter(
                x=st.session_state.df_raw['time_s'], 
                y=pd.to_numeric(st.session_state.df_raw[autocal_col], errors='coerce'), 
                mode='lines', 
                name='AutoCal Channel (Debugging)',
                line=dict(color='#ef4444', width=1, dash='dot'),
                yaxis='y2'
            ))
    
    # Compute domains and total height
    total_height = plot_height
    table_height = 360  # Fixed height in px for the table annotations
    if viz_mode == "Supine to Standing Analysis":
        total_height += table_height
        table_frac = table_height / total_height
        main_domain = [0, max(0.1, 1 - table_frac - 0.02)]
    else:
        main_domain = [0, 1]

    fig.update_layout(
        height=total_height,
        margin=dict(t=40, b=80),  # Keep top margin tight since title is moved
        xaxis=dict(title='Time'),
        yaxis=dict(title='Value', domain=main_domain),
        yaxis2=dict(
            title='AutoCal Level',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center")
    )
    
    st.subheader(f"📊 {plot_signal} Vector Tracking Timeline")
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📚 Metrics Definitions", expanded=False):
        definitions_data = {
            "Metric": [
                "Test Number",
                "Transition Time",
                "Transition Duration (s)",
                "Baseline Mean",
                "Value at Transition Start",
                "Min (Transition to End)",
                "Drop % (Transition to End)",
                "Area Above (Transition to End)",
                "Area Below (Transition to End)",
                "Value at Stand Start",
                "Min (Stand to End)",
                "Drop % (Stand to End)",
                "Area Above (Stand to End)",
                "Area Below (Stand to End)",
                "End Value"
            ],
            "Definition": [
                "Sequential number of the Supine-to-Standing test (1, 2, 3...) for the given signal.",
                "The absolute time the transition started (at the 'transition' comment).",
                "Time elapsed from 'transition' comment to 'stand' comment.",
                "Mean value of the signal during the user-defined baseline period.",
                "Signal value exactly at the 'transition' comment.",
                "Minimum signal value from the 'transition' comment until the 'End Marker'.",
                "Percentage drop from the Baseline (or Value at Transition Start) down to Min (Transition to End).",
                "Area of the signal above the Baseline (or Transition Value) from 'transition' to 'End Marker'.",
                "Area of the signal below the Baseline (or Transition Value) from 'transition' to 'End Marker'.",
                "Signal value exactly at the 'stand' comment.",
                "Minimum signal value from the 'stand' comment until the 'End Marker'.",
                "Percentage drop from the Baseline (or Stand Value) down to Min (Stand to End).",
                "Area of the signal above the Baseline (or Stand Value) from 'stand' to 'End Marker'.",
                "Area of the signal below the Baseline (or Stand Value) from 'stand' to 'End Marker'.",
                "Signal value exactly at the 'End Marker' time."
            ]
        }
        st.table(pd.DataFrame(definitions_data))
    
    # --- EXCEL EXPORT ---
    if viz_mode == "Supine to Standing Analysis":
        st.markdown("### Export Analysis")
        st.write("Generate an Excel report containing all filtered data, resampled data, metadata, and transition statistics.")
        
        if st.button("Generate Excel Report", type="primary"):
            with st.spinner("Generating Excel..."):
                # 1. Metadata
                metadata_dict = {
                    "File Name": uploaded_mat.name if uploaded_mat else "Unknown",
                    "Resampling Mode": "Beat-based" if st.session_state.beat_mode else "Time-based",
                    "Interval": resampled_label,
                    "Finger Pressure Filter": filter_method_fp
                }
                
                if filter_method_fp == "Savitzky-Golay":
                    metadata_dict["FP SavGol Window"] = fp_savgol_win
                    metadata_dict["FP SavGol Poly"] = fp_savgol_poly
                elif filter_method_fp == "Butterworth Low-Pass":
                    metadata_dict["FP Butter Cutoff (Hz)"] = fp_butter_cutoff
                    metadata_dict["FP Butter Order"] = fp_butter_order
                elif filter_method_fp == "Hampel (Outlier Removal)":
                    metadata_dict["FP Hampel Window"] = fp_hampel_win
                    metadata_dict["FP Hampel Sigma"] = fp_hampel_sig
                    
                metadata_dict["CBF Filter"] = filter_method_cbf
                
                if filter_method_cbf == "Savitzky-Golay":
                    metadata_dict["CBF SavGol Window"] = cbf_savgol_win
                    metadata_dict["CBF SavGol Poly"] = cbf_savgol_poly
                elif filter_method_cbf == "Butterworth Low-Pass":
                    metadata_dict["CBF Butter Cutoff (Hz)"] = cbf_butter_cutoff
                    metadata_dict["CBF Butter Order"] = cbf_butter_order
                elif filter_method_cbf == "Hampel (Outlier Removal)":
                    metadata_dict["CBF Hampel Window"] = cbf_hampel_win
                    metadata_dict["CBF Hampel Sigma"] = cbf_hampel_sig
                    
                metadata_dict.update({
                    "AutoCal Setting": auto_cal_option,
                    "Baseline Window (s)": baseline_window,
                    "End Marker Window (s)": end_marker_window,
                    "Use Baseline Area": use_baseline_area
                })
                
                # 2. Resampled Data Prep
                if st.session_state.beat_mode:
                    # Construct Beat DataFrame
                    beat_dfs = []
                    df_resampled = pd.DataFrame()
                    for sig_col in all_signals:
                        if sig_col in st.session_state.agg5_moving_map:
                            peaks_used = st.session_state.peaks_idx_cbf if (sig_col == fallback_signal and st.session_state.peaks_idx_cbf.size > 0) else st.session_state.peaks_idx
                            peaks_used = peaks_used[(peaks_used >= 0) & (peaks_used < len(st.session_state.df_raw))]
                            if len(peaks_used) > 0:
                                beat_vals = st.session_state.agg5_moving_map[sig_col][peaks_used]
                                beat_time_s = st.session_state.df_raw['time_s'].values[peaks_used]
                                beat_abs = st.session_state.df_raw['absolute_time'].values[peaks_used]
                                
                                temp_df = pd.DataFrame({
                                    'time_s': beat_time_s,
                                    'absolute_time': beat_abs,
                                    sig_col: beat_vals
                                })
                                beat_dfs.append(temp_df)
                    
                    if beat_dfs:
                        # Merge them all on time_s
                        df_resampled = beat_dfs[0]
                        for i in range(1, len(beat_dfs)):
                            df_resampled = pd.merge(df_resampled, beat_dfs[i], on=['time_s', 'absolute_time'], how='outer')
                        df_resampled = df_resampled.sort_values('time_s').reset_index(drop=True)
                else:
                    df_resampled = st.session_state.result_df.copy()
                    
                # 3. Stats Data Prep
                stats_list = []
                comments_data = st.session_state.df[st.session_state.df['comment'] != ""]
                comment_list = comments_data[['time_s', 'absolute_time', 'comment']].to_dict('records')
                
                for target_sig in [priority_signal, fallback_signal]:
                    if target_sig in all_signals:
                        if st.session_state.beat_mode:
                            targ_res_y = st.session_state.agg5_moving_map.get(target_sig, np.array([]))
                            targ_res_x = st.session_state.df_raw['time_s'].values
                        else:
                            targ_res_y = st.session_state.result_df[target_sig].values
                            targ_res_x = st.session_state.result_df['time_s'].values
                            
                        stats_list.extend(extract_stats_for_signal(
                            target_sig, targ_res_x, targ_res_y, comment_list, 
                            baseline_window, end_marker_window, use_baseline_area
                        ))
                        
                # Generate File
                excel_data = generate_excel_report(
                    st.session_state.df, 
                    df_resampled, 
                    metadata_dict, 
                    stats_list, 
                    resampled_label,
                    comment_list=comment_list,
                    baseline_window=baseline_window,
                    end_marker_window=end_marker_window,
                    use_baseline_area=use_baseline_area,
                    fp_signal=priority_signal,
                    cbf_signal=fallback_signal,
                    raw_df=st.session_state.df_raw,
                    agg5_map=st.session_state.agg5_moving_map if st.session_state.beat_mode else None,
                    result_df=st.session_state.result_df if not st.session_state.beat_mode else None,
                    is_beat_mode=st.session_state.beat_mode
                )
                
                # Construct filename
                now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                safe_resampled_label = str(resampled_label).replace(' ', '_').replace(':', '')
                base_name = uploaded_mat.name.split('.')[0] if uploaded_mat else "analysis"
                export_filename = f"{base_name}_{safe_resampled_label}_{now_str}.xlsx"
                
                st.download_button(
                    label="📥 Download Excel Report",
                    data=excel_data,
                    file_name=export_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# ---------------------------------------------------------
# Footer Credits
# ---------------------------------------------------------
st.markdown("---")
col_f1, col_f2 = st.columns([0.7, 0.3])
with col_f1:
    st.markdown("<p style='color: #64748b; font-size: 0.85rem;'>© 2026 FAME Laboratory, Greece.</p>", unsafe_allow_html=True)
with col_f2:
    st.markdown(
    "<p style='color: #64748b; font-size: 0.85rem; text-align: right;'>"
    "Contact: <a href='mailto:konstantinosmantzios@gmail.com' style='color: #38bdf8;'>K. Mantzios</a> | "
    "<a href='mailto:ggkikas77@gmail.com' style='color: #38bdf8;'>G. Gkikas</a>"
    "</p>",
    unsafe_allow_html=True
    )
