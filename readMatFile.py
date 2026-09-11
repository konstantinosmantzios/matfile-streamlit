import numpy as np
import gc
import pandas as pd
import scipy.io
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from export import generate_excel_report, extract_stats_for_signal
import os
from datetime import datetime, timedelta
import psutil

def log_memory(step_name):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Memory at {step_name}: {mem_info.rss / 1024 / 1024:.2f} MB")

def log_interaction(event, details=""):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] INTERACTION: {event} | {details}")

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
    find_autocal_column,
    estimate_dynamic_hr_array
)

# ---------------------------------------------------------
# Streamlit Interface Layout
# ---------------------------------------------------------

st.title("MAT Resampler & Stats Analyzer")
st.write("Convert, filter, and extract physiological standing response statistics.")
log_interaction("App Render Started", f"Session keys: {list(st.session_state.keys())}")
log_memory("App Start / Render")

with st.sidebar:
    uploaded_mat = st.file_uploader("Upload a MATLAB .mat file to begin", type=["mat"])

if uploaded_mat is not None:
    if st.session_state.get("last_uploaded_name") != uploaded_mat.name:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.last_uploaded_name = uploaded_mat.name
        st.rerun()

if uploaded_mat:
    if "df_raw" not in st.session_state:
        log_interaction("File Selected", uploaded_mat.name)
        log_memory("Before loadmat")
        mat = scipy.io.loadmat(uploaded_mat, squeeze_me=True)
        log_memory("After loadmat")
        df_channel_info = channel_info_df(mat)
        df_comments = comments_df(mat, df_channel_info)
        df = extract_channel_signals_with_comments(mat, df_comments)
        
        # Force garbage collection of the massive MATLAB dictionary
        del mat
        gc.collect()
        log_memory("After extract_channel_signals_with_comments")
        
        st.session_state.df_raw = df.sort_values('time_s').copy()
        st.session_state.df_comments = df_comments
    else:
        df = st.session_state.df_raw
        df_comments = st.session_state.df_comments

    all_columns = list(df.columns)

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
    
    apply_savgol_global = st.sidebar.checkbox("Apply Savitzky-Golay Filter", value=False, key="apply_savgol_global")
    if apply_savgol_global:
        savgol_win = st.sidebar.number_input("Filter Window Length", min_value=5, max_value=201, value=51, step=2)
        savgol_poly = st.sidebar.number_input("Filter Poly Order", min_value=1, max_value=5, value=3, step=1)
    else:
        savgol_win = 51
        savgol_poly = 3
        
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
    
    current_resample_mode = resample_mode
    current_bin_choice = bin_choice
    
    needs_resample = False
    if "result_df" not in st.session_state:
        needs_resample = True
    elif st.session_state.get("last_resample_mode") != current_resample_mode:
        needs_resample = True
    elif st.session_state.get("last_bin_choice") != current_bin_choice:
        needs_resample = True

    elif st.session_state.get("last_apply_savgol") != apply_savgol_global:
        needs_resample = True
    elif st.session_state.get("last_savgol_win") != savgol_win:
        needs_resample = True
    elif st.session_state.get("last_savgol_poly") != savgol_poly:
        needs_resample = True

    if needs_resample:
        st.session_state.last_resample_mode = current_resample_mode
        st.session_state.last_bin_choice = current_bin_choice
        st.session_state.last_apply_savgol = apply_savgol_global
        st.session_state.last_savgol_win = savgol_win
        st.session_state.last_savgol_poly = savgol_poly
        
        with st.spinner("Processing & resampling signal vectors..."):
            # We already have df_raw stored
            df_sorted = st.session_state.df_raw.copy()
                
            # Make sure numeric types are correct for all remaining signals before resampling
            for col in all_signals:
                df_sorted[col] = pd.to_numeric(df_sorted[col], errors='coerce').astype('float64')
                if apply_savgol_global:
                    valid_mask = ~np.isnan(df_sorted[col].values)
                    if valid_mask.any():
                        df_sorted.loc[valid_mask, col] = apply_savgol_filter(df_sorted.loc[valid_mask, col].values, window_length=savgol_win, polyorder=savgol_poly)

            if not beat_mode:
                # Time-based resampling
                selected_label = st.session_state.get('bin_choice_label', '1 sec')
                bin_sec = float(bin_map.get(selected_label, 1))
                
                df_tmp = df_sorted.copy()
                t0 = float(df_tmp['time_s'].iloc[0]) if len(df_tmp) > 0 and np.isfinite(df_tmp['time_s']).any() else 0.0
                df_tmp['time_bin'] = ((df_tmp['time_s'] - t0) // bin_sec).astype(int)
                
                result_df = df_tmp.groupby('time_bin').agg(
                    {**{c: 'mean' for c in all_signals},
                     'absolute_time': 'first',
                     'time_s': 'first',
                     'time_mmss_millis': 'first',
                     'comment': first_nonempty_comment}
                ).reset_index(drop=True)
                
                st.session_state.df = df_sorted
                st.session_state.result_df = result_df
                st.session_state.all_signals = all_signals
                st.session_state.beat_mode = False
            else:
                # Dynamic beat-based resampling
                ts = df_sorted['time_s'].values
                fs0 = 1.0 / float(np.nanmedian(np.diff(ts))) if len(ts) > 1 else 200.0
                
                sig0_raw = pd.to_numeric(df_sorted[main_signal], errors='coerce').astype('float64').values
                sig0_filt = rolling_median_np(sig0_raw, window=5)
                
                if "5: HR" in df_sorted.columns and not df_sorted["5: HR"].isna().all():
                    hr = pd.to_numeric(df_sorted["5: HR"], errors='coerce').astype('float64').values
                    est_hr_arr = estimate_dynamic_hr_array(sig0_filt, fs0)
                    hr = np.where(np.isnan(hr) | (hr <= 0), est_hr_arr, hr)
                else:
                    hr = estimate_dynamic_hr_array(sig0_filt, fs0)
                    
                hr_for_win = prepare_hr_for_window(hr.copy(), roll_win=1000)
                hr_for_win = np.where(np.isfinite(hr_for_win), hr_for_win, 60.0)
                # Factor = 2.0 means win_half is exactly 0.5 * R-R interval.
                # This guarantees that the exclusion window will NEVER overlap with the adjacent peak
                # even if the heart rate changes abruptly.
                win_half = compute_win_half_from_hr(hr_for_win, fs0, factor=2.0, min_samples=3)
                
                peaks = np.array([], dtype=int)
                peaks_cbf = np.array([], dtype=int)
                # Instead of mapping to 200Hz, we store discrete beat values
                beat_data_map = {sig_name: {'value': [], 'idx': []} for sig_name in all_signals}
                
                if "block_index" not in df_sorted.columns:
                    df_sorted["block_index"] = 0
                block_ids = df_sorted["block_index"].values
                unique_blocks = np.unique(block_ids)
                
                for b_id in unique_blocks:
                    b_mask = (block_ids == b_id)
                    b_indices = np.where(b_mask)[0]
                    if len(b_indices) == 0: continue
                    
                    b_offset = b_indices[0]
                    sig0_b = sig0_filt[b_mask]
                    win_half_b = win_half[b_mask]
                    
                    p_b = find_local_high_indices(sig0_b, win_half_b)
                    if len(p_b) > 0:
                        peaks = np.concatenate([peaks, p_b + b_offset])
                        
                    if fallback_signal in all_signals:
                        sig_cbf_b = rolling_median_np(pd.to_numeric(df_sorted[fallback_signal].values[b_mask], errors='coerce').astype('float64'), window=5)
                        p_cbf_b = find_local_high_indices(sig_cbf_b, win_half_b)
                        if len(p_cbf_b) > 0:
                            peaks_cbf = np.concatenate([peaks_cbf, p_cbf_b + b_offset])
                    else:
                        p_cbf_b = np.array([], dtype=int)
                        
                    for sig_name in all_signals:
                        y_b = pd.to_numeric(df_sorted[sig_name].values[b_mask], errors='coerce').astype('float64')
                        peaks_used = p_cbf_b if (sig_name == fallback_signal and p_cbf_b.size > 0) else p_b
                        
                        if peaks_used.size >= beats_k:
                            nP = peaks_used.size
                            half_k = beats_k // 2
                            for i in range(half_k, nP - (beats_k - half_k - 1)):
                                i0, i1 = peaks_used[i - half_k], peaks_used[i + (beats_k - half_k - 1)]
                                seg_vals = y_b[i0:i1+1]
                                mval = np.nan if np.isnan(seg_vals).all() else float(np.nanmean(seg_vals))
                                beat_data_map[sig_name]['value'].append(mval)
                                beat_data_map[sig_name]['idx'].append(peaks_used[i] + b_offset)

                # Construct Beat DataFrame (result_df) uniformly
                beat_dfs = []
                for sig_col in all_signals:
                    idx_arr = np.array(beat_data_map[sig_col]['idx'], dtype=int)
                    if len(idx_arr) > 0:
                        temp_df = pd.DataFrame({
                            'time_s': df_sorted['time_s'].values[idx_arr],
                            'absolute_time': df_sorted['absolute_time'].values[idx_arr],
                            sig_col: np.array(beat_data_map[sig_col]['value'], dtype=float)
                        })
                        beat_dfs.append(temp_df)
                        
                df_resampled = pd.DataFrame()
                if beat_dfs:
                    df_resampled = beat_dfs[0]
                    for i in range(1, len(beat_dfs)):
                        df_resampled = pd.merge(df_resampled, beat_dfs[i], on=['time_s', 'absolute_time'], how='outer')
                    df_resampled = df_resampled.sort_values('time_s').reset_index(drop=True)

                st.session_state.df = df_sorted
                st.session_state.all_signals = all_signals
                st.session_state.beat_mode = True
                st.session_state.result_df = df_resampled
                st.session_state.peaks_idx = peaks
                st.session_state.peaks_idx_cbf = peaks_cbf

    if True:
        # Display Visualization
        log_memory("Before Data Visualization Plotting")
        st.subheader("Data Visualization")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            viz_mode = st.radio("Visualization Mode", ["Free View", "Supine to Standing Analysis"], index=1, horizontal=True, key="viz_mode")
        with col2:
            show_comments = st.checkbox("Show Comments", value=True, key="show_comments")
            show_tables = st.checkbox("Show Summary Tables", value=True, key="show_tables")
            
        with st.expander("Analysis Settings", expanded=True):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                baseline_window = st.number_input("Baseline (s)", min_value=10, max_value=600, value=30, step=10, help="Duration of the baseline period prior to the standing transition.")
            with col_b:
                end_marker_window = st.number_input("End Marker (s)", min_value=1, max_value=300, value=10, step=1, help="Time after the standing comment to place the End marker.")
            with col_c:
                st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
                use_baseline_area = st.checkbox("Area from Baseline", value=False, key="use_baseline_area", help="Calculate shaded areas relative to the Baseline Mean instead of the marker values.")
        plot_height = st.slider("Main Plot Height (px)", min_value=300, max_value=2000, value=800, step=50, key="plot_height", help="Adjust the height of the signal plot area.")
            
        plot_signal = st.selectbox("Select Signal Waveform to Plot", all_signals, index=0, key="plot_signal")
        log_interaction("Signal Selected for Plot", plot_signal)
        
        # Prepare x and y for raw and resampled data
        raw_x = st.session_state.df_raw['time_s'].values
        raw_t = st.session_state.df_raw['absolute_time'].values
        
        resampled_label = st.session_state.get('bin_choice_label', 'Resampled Data')
        
        resampled_x = st.session_state.result_df['time_s'].values
        resampled_t = st.session_state.result_df['absolute_time'].values
        if plot_signal in st.session_state.result_df.columns:
            resampled_y = st.session_state.result_df[plot_signal].values
            
            # Remove NaNs caused by outer joins across signals in beat mode
            valid_mask = ~np.isnan(resampled_y)
            resampled_x = resampled_x[valid_mask]
            resampled_t = resampled_t[valid_mask]
            resampled_y = resampled_y[valid_mask]
        else:
            resampled_y = np.array([])
            
        # --- Slice Data for Plotly ---
        t_min = 0
        max_time = np.max(resampled_x) if len(resampled_x) > 0 else 0
        t_max = max_time
        plot_res_t = resampled_t
        plot_res_y = resampled_y
        segment_sec = 0
            
        base_fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        fig = FigureResampler(base_fig, default_n_shown_samples=150_000)
        
        # Plot Raw Trace in Free View
        if viz_mode == "Free View":
            raw_y = st.session_state.df_raw[plot_signal].values
            
            # Always plot Raw Signal at the back
            fig.add_trace(go.Scattergl(
                x=raw_t, y=raw_y, mode='lines', name='Raw Signal (200Hz)',
                line=dict(color='rgba(150, 150, 150, 0.5)', width=1),
                hoverinfo='skip'
            ))

            # If filter is globally on, st.session_state.df has the filtered data
            if st.session_state.get("last_apply_savgol", False):
                filtered_y = st.session_state.df[plot_signal].values
                fig.add_trace(go.Scattergl(
                    x=raw_t, y=filtered_y, mode='lines', name='Savitzky-Golay Filtered',
                    line=dict(color='rgba(50, 150, 250, 0.8)', width=1.5),
                    hoverinfo='skip'
                ))

        # Plot Resampled Trace
        fig.add_trace(go.Scattergl(
            x=plot_res_t, y=plot_res_y, mode='lines', name=f'Resampled ({resampled_label})',  
            line=dict(color='#f97316', width=2)
        ))
        
        # Plot Peaks if in beat mode (only for FP and CBF and Free View)
        if st.session_state.beat_mode and viz_mode != "Supine to Standing Analysis":
            if plot_signal in [priority_signal, fallback_signal]:
                peaks_to_plot = st.session_state.peaks_idx_cbf if (plot_signal == fallback_signal and st.session_state.peaks_idx_cbf.size > 0) else st.session_state.peaks_idx
                peaks_to_plot = peaks_to_plot[(peaks_to_plot >= 0) & (peaks_to_plot < len(raw_x))]
                if peaks_to_plot.size > 0:
                    y_vals = st.session_state.df[plot_signal].values
                    # Slice peaks
                    if segment_sec > 0:
                        peak_times = raw_x[peaks_to_plot]
                        peak_mask = (peak_times >= t_min) & (peak_times <= t_max)
                        peaks_to_plot = peaks_to_plot[peak_mask]
                    
                    if peaks_to_plot.size > 0:
                        fig.add_trace(go.Scattergl(
                            x=raw_t[peaks_to_plot], 
                            y=y_vals[peaks_to_plot],
                            mode='markers', 
                            name='Detected Peaks',
                            marker=dict(color='#ef4444', size=5, symbol='x')
                        ))
                        
        # Plot Peak-to-Peak Interval on the second subplot
        priority_signal = "1: Finger Pressure"
        fallback_signal = "12: CBF"
        
        peaks_for_interval = np.array([])
        if plot_signal in [priority_signal, fallback_signal]:
            peaks_for_interval = st.session_state.peaks_idx_cbf if (plot_signal == fallback_signal and st.session_state.peaks_idx_cbf.size > 0) else st.session_state.peaks_idx
        else:
            # Fallback to Finger Pressure peaks if we are plotting something else (like ECG)
            peaks_for_interval = st.session_state.peaks_idx
            
        peaks_for_interval = peaks_for_interval[(peaks_for_interval >= 0) & (peaks_for_interval < len(raw_x))]
        
        if len(peaks_for_interval) > 1:
            if segment_sec > 0:
                peak_times = raw_x[peaks_for_interval]
                peak_mask = (peak_times >= t_min) & (peak_times <= t_max)
                peaks_for_interval = peaks_for_interval[peak_mask]
                
            if len(peaks_for_interval) > 1:
                rr_interval_s = np.diff(raw_x[peaks_for_interval])
                rr_interval_ms = rr_interval_s * 1000.0
                interval_t = raw_t[peaks_for_interval[1:]]
                
                fig.add_trace(go.Scattergl(
                    x=interval_t, 
                    y=rr_interval_ms, 
                    mode='lines+markers', 
                    name='P-P Interval (ms)',
                    line=dict(color='#8b5cf6', width=1),
                    marker=dict(size=3)
                ), row=2, col=1)
                
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
                                
                                t_start = c1['time_s']
                                t_stand = c_stand['time_s']
                                t_end_marker = t_stand + end_marker_window
                                t_base_start = max(0, t_start - baseline_window)
                                
                                # Skip if completely out of view
                                if segment_sec > 0 and (t_end_marker < t_min or t_base_start > t_max):
                                    i += 1
                                    continue

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
                                if show_tables:
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
                    # Filter comments by time
                    if segment_sec > 0 and (row['time_s'] < t_min or row['time_s'] > t_max):
                        continue
                        
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
        debug_mode = st.session_state.get('debug_mode', False)
        if debug_mode:
            autocal_col = find_autocal_column(st.session_state.df_raw)
            if autocal_col is not None:
                if segment_sec > 0:
                    ac_mask = (st.session_state.df_raw['time_s'] >= t_min) & (st.session_state.df_raw['time_s'] <= t_max)
                    ac_x = st.session_state.df_raw['time_s'][ac_mask]
                    ac_y = pd.to_numeric(st.session_state.df_raw[autocal_col][ac_mask], errors='coerce')
                else:
                    ac_x = st.session_state.df_raw['time_s']
                    ac_y = pd.to_numeric(st.session_state.df_raw[autocal_col], errors='coerce')
                    
                fig.add_trace(go.Scatter(
                    x=ac_x, 
                    y=ac_y, 
                    mode='lines', 
                    name='AutoCal Channel (Debugging)',
                    line=dict(color='#ef4444', width=1, dash='dot'),
                    yaxis='y2'
                ))
        
        # Compute domains and total height
        total_height = plot_height + 250 # Increase height for the second subplot
        
        table_height = 360  # Fixed height in px for the table annotations
        if show_tables and viz_mode == "Supine to Standing Analysis":
            total_height += table_height
            table_frac = table_height / total_height
            top_bound = max(0.1, 1 - table_frac - 0.02)
        else:
            top_bound = 1.0

        # Define domains to create space at the top if tables are shown
        domain_y1 = [0.3, top_bound]
        domain_y3 = [0.0, 0.25]

        fig.update_layout(
            height=total_height,
            margin=dict(t=40, b=80),  # Keep top margin tight since title is moved
            xaxis=dict(title='Time'),
            xaxis2=dict(title='Time'),
            yaxis=dict(title='Value', domain=domain_y1),
            yaxis2=dict(
                title='AutoCal Level',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            yaxis3=dict(
                title='P-P Interval (ms)',
                domain=domain_y3,
                showgrid=True,
                gridcolor="rgba(255,255,255,0.1)"
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
        if True:
            st.markdown("### Export Analysis")
            st.write("Generate an Excel report containing resampled data, metadata, and transition statistics.")
            
            if st.button("Generate Excel Report", type="primary"):
                log_interaction("Generate Excel", "Button clicked")
                with st.spinner("Generating Excel..."):
                    # 1. Metadata
                    metadata_dict = {
                        "File Name": uploaded_mat.name if uploaded_mat else "Unknown",
                        "Resampling Mode": "Beat-based" if st.session_state.beat_mode else "Time-based",
                        "Interval": resampled_label
                    }
                    
                    metadata_dict.update({
                        "Baseline Window (s)": baseline_window,
                        "End Marker Window (s)": end_marker_window,
                        "Use Baseline Area": use_baseline_area
                    })
                    
                    # 2. Resampled Data Prep
                    df_resampled = st.session_state.result_df.copy()
                        
                    # 3. Stats Data Prep
                    stats_list = []
                    comments_data = st.session_state.df[st.session_state.df['comment'] != ""]
                    comment_list = comments_data[['time_s', 'absolute_time', 'comment']].to_dict('records')
                    
                    for target_sig in [priority_signal, fallback_signal]:
                        if target_sig in all_signals:
                            targ_res_y = st.session_state.result_df.get(target_sig, pd.Series()).values
                            targ_res_x = st.session_state.result_df['time_s'].values
                            
                            valid_mask = ~np.isnan(targ_res_y)
                            targ_res_y = targ_res_y[valid_mask]
                            targ_res_x = targ_res_x[valid_mask]
                                
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
                        result_df=st.session_state.result_df,
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
                    log_interaction("Download Excel", "Button rendered")

import gc
gc.collect()

import sys
def get_size(obj):
    if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
        return obj.memory_usage(deep=True).sum()
    elif isinstance(obj, np.ndarray):
        return obj.nbytes
    else:
        return sys.getsizeof(obj)

sizes = []
for k, v in st.session_state.items():
    sizes.append((k, get_size(v)))
sizes.sort(key=lambda x: x[1], reverse=True)
log_interaction("Top 5 Session State Memory", ", ".join([f"{k}: {v/1024/1024:.2f} MB" for k, v in sizes[:5]]))

log_memory("App End")
log_interaction("App Render Finished")
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
