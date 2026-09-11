import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from processing import (
    rolling_median_np, prepare_hr_for_window, compute_win_half_from_hr,
    find_local_high_indices, apply_savgol_filter, apply_butter_lowpass,
    apply_hampel_filter, find_autocal_column, first_nonempty_comment,
    estimate_dynamic_hr_array
)

def process_session_data(df_raw, df_channel_info, df_comments, settings):
    df_sorted = df_raw.sort_values('time_s').copy()
    all_signals = df_sorted.columns[df_sorted.columns.str.contains(':', regex=False)].tolist()
    
    if not all_signals:
        return df_sorted, pd.DataFrame(), [], [], {}
        
    priority_signal = next((s for s in all_signals if s.startswith("1:") or "Finger Pressure" in s), all_signals[0])
    fallback_signal = next((s for s in all_signals if s.startswith("6:") or "CBF" in s), None)
    main_signal = priority_signal
    
    # --- AutoCal Masking ---
    autocal_col = find_autocal_column(df_sorted)
    mask = np.zeros(len(df_sorted), dtype=bool)
    
    auto_cal_option = settings.get("autoCalOption", "Auto-Detect")
    use_channel_masking = False
    
    if auto_cal_option == "Force Enabled (Channel)":
        use_channel_masking = True
    elif auto_cal_option == "Auto-Detect":
        try:
            temp_col = pd.to_numeric(df_sorted[autocal_col], errors='coerce')
            perc_below_05 = (temp_col < 0.5).mean()
            if perc_below_05 <= 0.95:
                use_channel_masking = True
        except:
            pass
            
    if use_channel_masking and autocal_col is not None:
        try:
            temp_col = pd.to_numeric(df_sorted[autocal_col], errors='coerce')
            mask = temp_col < 0.5
        except:
            mask = (df_sorted[autocal_col] == 0) | (df_sorted[autocal_col] == False)
    else:
        # HCU comment masking
        if df_comments is not None and not df_comments.empty and autocal_col is not None:
            matches = df_comments[df_comments["comment_text"] == "HCU not connected"]
            if not matches.empty:
                n_blocks = int(df_sorted["block_index"].max()) + 1 if "block_index" in df_sorted.columns else 1
                block_lengths = [0] * n_blocks
                for _, row in df_channel_info.iterrows():
                    if row["title"] == autocal_col and row.get("datastart") is not None:
                        block_lengths = [int(e-s+1) for s, e in zip(row["datastart"], row["dataend"])]
                        break
                if not block_lengths or block_lengths[0] == 0:
                    block_lengths = [len(df_sorted) // n_blocks] * n_blocks
                    
                block_offsets = np.cumsum([0] + block_lengths[:-1])
                temp_col = pd.to_numeric(df_sorted[autocal_col], errors='coerce').fillna(0)
                is_spike = (temp_col >= 0.5).values
                
                for _, ev in matches.iterrows():
                    block_index = int(ev["block_index"]) - 1
                    sample_index = int(ev["sample_index"])
                    if 0 <= block_index < len(block_offsets):
                        idx_global = block_offsets[block_index] + sample_index
                        future_spikes = np.where(is_spike[idx_global:])[0]
                        if len(future_spikes) > 0:
                            mask[idx_global:idx_global + future_spikes[0]] = True
                        else:
                            fs = float(df_channel_info[df_channel_info["title"] == all_signals[0]]["samplerate"].iloc[0]) if not df_channel_info.empty else 1000.0
                            end_idx = min(len(df_sorted), idx_global + int(15 * fs))
                            mask[idx_global:end_idx] = True

    # --- Convert to numeric and apply AutoCal mask ---
    for col in all_signals:
        if col != fallback_signal and col != autocal_col:
            df_sorted[col] = pd.to_numeric(df_sorted[col], errors='coerce').astype('float64')
            if col == priority_signal:
                df_sorted.loc[mask, col] = np.nan
                
    for bp_col in ["2: MAP", "3: Systolic", "4: Diastolic"]:
        if bp_col in df_sorted.columns:
            df_sorted[bp_col] = df_sorted[bp_col].replace(0, np.nan)

    # =========================================================================
    # STEP 1: Apply user-configured filters BEFORE peak detection
    # =========================================================================
    # Determine sampling rate for the main signal
    fs_main = float(
        df_channel_info[df_channel_info["title"] == main_signal]["samplerate"].iloc[0]
    ) if not df_channel_info[df_channel_info["title"] == main_signal].empty else 200.0

    # Filter Finger Pressure
    fm_fp = settings.get("fpFilter", "None")
    if fm_fp == "Savitzky-Golay":
        df_sorted[main_signal] = apply_savgol_filter(
            df_sorted[main_signal].values,
            settings.get("fpSavgolWin", 51),
            settings.get("fpSavgolPoly", 5)
        )
    elif fm_fp == "Butterworth Low-Pass":
        df_sorted[main_signal] = apply_butter_lowpass(
            df_sorted[main_signal].values,
            settings.get("fpButterCutoff", 5.0),
            fs_main,
            settings.get("fpButterOrder", 4)
        )
    elif fm_fp == "Hampel":
        df_sorted[main_signal] = apply_hampel_filter(
            df_sorted[main_signal].values,
            settings.get("fpHampelWin", 5),
            settings.get("fpHampelSig", 3.0)
        )

    # Filter CBF
    if fallback_signal and fallback_signal in all_signals:
        fs_cbf = float(
            df_channel_info[df_channel_info["title"] == fallback_signal]["samplerate"].iloc[0]
        ) if not df_channel_info[df_channel_info["title"] == fallback_signal].empty else 200.0

        fm_cbf = settings.get("cbfFilter", "None")
        if fm_cbf == "Savitzky-Golay":
            df_sorted[fallback_signal] = apply_savgol_filter(
                df_sorted[fallback_signal].values,
                settings.get("cbfSavgolWin", 51),
                settings.get("cbfSavgolPoly", 5)
            )
        elif fm_cbf == "Butterworth Low-Pass":
            df_sorted[fallback_signal] = apply_butter_lowpass(
                df_sorted[fallback_signal].values,
                settings.get("cbfButterCutoff", 5.0),
                fs_cbf,
                settings.get("cbfButterOrder", 4)
            )
        elif fm_cbf == "Hampel":
            df_sorted[fallback_signal] = apply_hampel_filter(
                df_sorted[fallback_signal].values,
                settings.get("cbfHampelWin", 5),
                settings.get("cbfHampelSig", 3.0)
            )
    else:
        fs_cbf = fs_main

    # Convert to float32 for memory efficiency (after filtering is done in float64)
    for col in all_signals:
        df_sorted[col] = pd.to_numeric(df_sorted[col], errors='coerce').astype('float32')

    # =========================================================================
    # STEP 2: Resampling (time-based or beat-based with peak detection)
    # =========================================================================
    resample_mode = settings.get("resampleMode", "Time-based")
    beat_mode = (resample_mode == "Beat-based")
    
    # Identify segments based on large time gaps (> 1 second)
    df_sorted['segment_id'] = (df_sorted['absolute_time'].diff() > pd.Timedelta(seconds=1)).cumsum()
    
    result_df = pd.DataFrame()
    agg5_moving_map = {}
    peaks = np.array([], dtype=int)
    peaks_cbf = np.array([], dtype=int)
    
    if not beat_mode:
        bin_sec = float(settings.get("resampleRateTime", 1))
        
        df_tmp = df_sorted.copy()
        t0 = float(df_tmp['time_s'].iloc[0]) if len(df_tmp) > 0 and np.isfinite(df_tmp['time_s']).any() else 0.0
        df_tmp['time_bin'] = ((df_tmp['time_s'] - t0) // bin_sec).astype(int)
        
        agg_funcs = {c: 'mean' for c in all_signals}
        agg_updates = {'absolute_time': 'mean', 'time_s': 'mean', 'comment': first_nonempty_comment}
        if 'time_mmss_millis' in df_tmp.columns:
            agg_updates['time_mmss_millis'] = 'first'
        agg_funcs.update(agg_updates)
        
        result_df = df_tmp.groupby(['segment_id', 'time_bin']).agg(agg_funcs).reset_index(drop=True)
    else:
        # ---- Beat-based resampling with peak detection on FILTERED data ----
        beats_k = settings.get("resampleRateBeat", 5)
        ts = df_sorted['time_s'].values
        fs0 = 1.0 / float(np.nanmedian(np.diff(ts))) if len(ts) > 1 else 200.0
        
        sig0_filtered = pd.to_numeric(df_sorted[main_signal], errors='coerce').astype('float64').values
        
        if "5: HR" in df_sorted.columns and not df_sorted["5: HR"].isna().all():
            hr = pd.to_numeric(df_sorted["5: HR"], errors='coerce').astype('float64').values
            est_hr_arr = estimate_dynamic_hr_array(sig0_filtered, fs0)
            hr = np.where(np.isnan(hr) | (hr <= 0), est_hr_arr, hr)
        else:
            hr = estimate_dynamic_hr_array(sig0_filtered, fs0)

        hr_for_win = prepare_hr_for_window(hr.copy(), roll_win=1000)
        hr_for_win = np.where(np.isfinite(hr_for_win), hr_for_win, 60.0)
        win_half = compute_win_half_from_hr(hr_for_win, fs0, factor=2.0, min_samples=3)
        
        block_ids = df_sorted["segment_id"].values
        unique_blocks = np.unique(block_ids)
        
        beat_rows = []
        
        for b_id in unique_blocks:
            b_mask = (block_ids == b_id)
            b_indices = np.where(b_mask)[0]
            if len(b_indices) == 0: continue
            
            b_offset = b_indices[0]
            sig0_b = sig0_filtered[b_mask]
            win_half_b = win_half[b_mask]
            
            p_b = find_local_high_indices(sig0_b, win_half_b, fs=fs0)
            if len(p_b) > 0:
                peaks = np.concatenate([peaks, p_b + b_offset])
                
            p_cbf_b = np.array([], dtype=int)
            if fallback_signal and fallback_signal in all_signals:
                sig_cbf_b = pd.to_numeric(df_sorted[fallback_signal].values[b_mask], errors='coerce').astype('float64')
                p_cbf_b = find_local_high_indices(sig_cbf_b, win_half_b, fs=fs_cbf)
                if len(p_cbf_b) > 0:
                    peaks_cbf = np.concatenate([peaks_cbf, p_cbf_b + b_offset])
                    
            nP = p_b.size
            if nP < beats_k + 1:
                continue
                
            half_k = beats_k // 2
            if beats_k % 2 == 0:
                start_i = half_k
                end_i = nP - half_k
            else:
                start_i = half_k
                end_i = nP - half_k - 1
                
            for i in range(start_i, end_i):
                row = {}
                if beats_k % 2 == 0:
                    i0 = p_b[i - half_k]
                    i1 = p_b[i + half_k]
                    center_idx = p_b[i]
                else:
                    i0 = p_b[i - half_k]
                    i1 = p_b[i + half_k + 1]
                    center_idx = (p_b[i] + p_b[i+1]) // 2
                
                row['absolute_time'] = float(df_sorted['absolute_time'].values[center_idx + b_offset])
                row['time_s'] = float(ts[center_idx + b_offset])
                
                c_val = df_sorted['comment'].values[center_idx + b_offset]
                row['comment'] = c_val if isinstance(c_val, str) and c_val.strip() else ""
                
                for sig_name in all_signals:
                    y_b = pd.to_numeric(df_sorted[sig_name].values[b_mask], errors='coerce').astype('float64')
                    
                    if sig_name == fallback_signal and p_cbf_b.size > 0:
                        idx_cbf_start = np.argmin(np.abs(p_cbf_b - i0))
                        idx_cbf_end = np.argmin(np.abs(p_cbf_b - i1))
                        cbf_i0 = p_cbf_b[min(idx_cbf_start, idx_cbf_end)]
                        cbf_i1 = p_cbf_b[max(idx_cbf_start, idx_cbf_end)]
                        seg_vals = y_b[cbf_i0 : cbf_i1 + 1]
                    else:
                        seg_vals = y_b[i0 : i1 + 1]
                        
                    mval = np.nan if np.isnan(seg_vals).all() else float(np.nanmean(seg_vals))
                    row[sig_name] = mval
                    
                beat_rows.append(row)
                
        result_df = pd.DataFrame(beat_rows)
                
    return df_sorted, result_df, peaks, peaks_cbf, agg5_moving_map
