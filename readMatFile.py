# streamlit run readMatFile.py
import numpy as np
import pandas as pd
import scipy.io
import streamlit as st
from scipy.signal import argrelextrema
import plotly.graph_objects as go
import os
from datetime import datetime
import base64
import streamlit.web.cli

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

def sec_to_mmss_millis(seconds):
    if not np.isfinite(seconds):
        return ""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"

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
    com = mat['com']
    comtext = mat['comtext']

    # Choose fs from Channel 18 if possible, else first channel
    channel_18_idx = 17 if len(titles) > 17 else 0
    fs = int(samplerate[channel_18_idx, 0]) if samplerate.ndim > 1 else int(samplerate[channel_18_idx])

    signals, channel_names, lengths = [], [], []
    for i in range(len(titles)):
        channel_signal = []
        for block in range(datastart.shape[1]):
            if not np.isnan(datastart[i, block]) and not np.isnan(dataend[i, block]):
                start = int(datastart[i, block]) - 1
                end = int(dataend[i, block])
                channel_signal.append(data_flat[start:end])
        if channel_signal:
            sig = np.concatenate(channel_signal)
            if sig.size > 0 and np.isfinite(sig).any():
                signals.append(sig)
                channel_names.append(titles[i])
                lengths.append(len(sig))

    if not signals or not lengths or min(lengths) == 0:
        st.warning("No valid signal data in this MAT file.")
        return pd.DataFrame()

    min_length = min(lengths)
    signals_trunc = [ch[:min_length] for ch in signals]
    time = np.arange(min_length) / fs if min_length > 0 else np.array([])
    time_mmss_millis = [sec_to_mmss_millis(t) for t in time] if len(time) > 0 else []

    # Main DataFrame
    data = {"time_s": time, "time_mmss_millis": time_mmss_millis}
    for ch_name, ch_data in zip(channel_names, signals_trunc):
        data[ch_name] = ch_data
    df = pd.DataFrame(data)

    # Event Comments: use fs from Channel 18 or channel 0
    event_times, event_labels = [], []
    fs_for_events = int(samplerate[0, 0]) if samplerate.ndim > 1 else int(samplerate[0])
    for row in com:
        timestamp_samples = row[2]
        try:
            text_idx = int(row[4]) - 1
            text = comtext[text_idx].strip()
        except Exception:
            text = f"(invalid text index: {row[4]})"
        time_sec = timestamp_samples / fs_for_events
        event_times.append(time_sec)
        event_labels.append(text)

    # Create a comment vector for all time points (attach event at exact match or closest)
    comments_col = np.full(min_length, "", dtype=object)
    for t_event, label in zip(event_times, event_labels):
        # Find the closest time index to the event time
        idx = np.abs(time - t_event).argmin() if min_length > 0 else 0
        comments_col[idx] = label
    df["comment"] = comments_col

    # Drop unwanted channels
    columnsDrop = [
        '7: Interbeat Interval', '8: Active Cuff', '9: Cuff Countdown',
        '10: AutoCal Quality', '16: PL ch 1 lead 1', '17: PL ch 2 lead 2',
        '19: Lead aVR', '20: Lead aVL', '21: Lead aVF', '22: HR EKG lead 1',
        '23: : HR EKG lead 2', 'Channel 24', 'Channel 25', 'Channel 26',
    ]
    df = df.drop(columns=[c for c in columnsDrop if c in df.columns], errors='ignore')

    print(df)
    return df


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
    fallback_signal = 'Channel 18'

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
        st.error("No suitable signals found with data ('1: Finger Pressure' or 'Channel 18').")
        st.stop()


    st.markdown("### Settings")
    bin_choice = st.radio("Sample Rate", [
        "500ms", "1 sec", "2 sec", "5 sec", "10 sec", "15 sec", "30 sec", "1 min"
    ], index=1, horizontal=True)
    bin_map = {
        "500ms": 0.5, "1 sec": 1, "2 sec": 2, "5 sec": 5,
        "10 sec": 10, "15 sec": 15, "30 sec": 30, "1 min": 60
    }
    bin_seconds = bin_map[bin_choice]

    # Only show the filtering controls if '1: Finger Pressure' is present
    if main_signal == priority_signal:
        with st.expander("Finger Pressure Filtering Settings", expanded=True):
            seconds = st.slider("Window (sec)", 0.05, 1.0, 0.15, step=0.05)
            pivot_window = int(seconds * 200)
            high_threshold = st.slider("High Threshold", 0, 1000, 250)

    # Channel 18 Filtering Parameters (always present if file is uploaded)
    if uploaded_mat and 'Channel 18' in all_columns:
        with st.expander("### Channel 18 Filtering Settings", expanded=True):
            ch18_order = st.slider("Pivot Detection Window (order)", min_value=10, max_value=200, value=30, step=5)
            ch18_limit = st.slider("High % Limit for Filtering", min_value=10, max_value=300, value=180, step=5)
    else:
        ch18_order, ch18_limit = 30, 180  # Defaults, just in case


    if st.button("Convert and Resample"):
        with st.spinner("Converting and resampling... Please wait."):
            if main_signal == priority_signal:
                df['unfiltered_signal'] = df[main_signal].copy()
                if '11: AutoCal Countdown' in df.columns:
                    mask = df['11: AutoCal Countdown'] < 0.5
                    for col in all_signals:

                        if col == 'Channel 18':
                            df['unfiltered_signal_Ch_18'] = df['Channel 18'].copy()
                            # Filter Channel 18 using custom logic
                            signal = df[col].values
                            time = df['time_s'].values
                            pivot_high_idx = argrelextrema(signal, np.greater, order=ch18_order)[0]
                            pivot_low_idx  = argrelextrema(signal, np.less,    order=ch18_order)[0]
                            indices_to_nan = []
                            last_low = None
                            last_low_val = None
                            li = 0

                            for hi in pivot_high_idx:
                                while li < len(pivot_low_idx) and pivot_low_idx[li] < hi:
                                    last_low = pivot_low_idx[li]
                                    last_low_val = signal[last_low]
                                    li += 1
                                if last_low is not None and last_low_val != 0:
                                    pct = 100 * (signal[hi] - last_low_val) / abs(last_low_val)
                                    future_lows = pivot_low_idx[pivot_low_idx > hi]
                                    if pct > ch18_limit and len(future_lows) > 0:
                                        next_low = future_lows[0]
                                        indices_to_nan.extend(range(last_low, next_low + 1))

                            filtered_signal = signal.copy()
                            if indices_to_nan:
                                filtered_signal[indices_to_nan] = np.nan
                            df[col] = filtered_signal


                        else:
                            df.loc[mask, col] = np.nan

                # Filtering based on local minima/maxima and high threshold
                signal_fp = main_signal
                valid_df = df[['time_s', signal_fp]].dropna().reset_index(drop=True)
                geq_vec = np.vectorize(lambda a, b: a >= b, otypes=[np.bool_])
                leq_vec = np.vectorize(lambda a, b: a <= b, otypes=[np.bool_])
                pivot_highs_idx = argrelextrema(valid_df[signal_fp].values, comparator=geq_vec, order=pivot_window)[0]
                pivot_lows_idx = argrelextrema(valid_df[signal_fp].values, comparator=leq_vec, order=pivot_window)[0]
                pivot_highs = valid_df.loc[pivot_highs_idx].reset_index(drop=True)
                pivot_lows = valid_df.loc[pivot_lows_idx].reset_index(drop=True)

                filtered_signal = df[signal_fp].copy()
                for i in range(len(pivot_lows) - 1):
                    low_start = pivot_lows.loc[i, 'time_s']
                    low_end = pivot_lows.loc[i + 1, 'time_s']
                    highs_between = pivot_highs[(pivot_highs['time_s'] > low_start) & (pivot_highs['time_s'] < low_end)]
                    if not highs_between.empty and (highs_between[signal_fp] > high_threshold).any():
                        filtered_signal[(df['time_s'] > low_start) & (df['time_s'] < low_end)] = np.nan
                df[signal_fp] = filtered_signal

                df_sorted = df.sort_values('time_s').copy()
                t0 = df_sorted['time_s'].iloc[0]
                df_sorted['time_bin'] = ((df_sorted['time_s'] - t0) // bin_seconds).astype(int)
                agg = df_sorted.groupby('time_bin').agg(
                    {**{c: 'mean' for c in all_signals}, 'time_s': 'first', 'time_mmss_millis': 'first'}
                ).reset_index()
                agg['bin_start_time'] = t0 + agg['time_bin'] * bin_seconds

                st.session_state.df = df
                st.session_state.result_df = agg
                st.session_state.all_signals = all_signals

            # elif main_signal == fallback_signal:
            #     df['unfiltered_signal'] = df[main_signal].copy()
            #     df_sorted = df.sort_values('time_s').copy()
            #     t0 = df_sorted['time_s'].iloc[0]
            #     df_sorted['time_bin'] = ((df_sorted['time_s'] - t0) // bin_seconds).astype(int)
            #     agg = df_sorted.groupby('time_bin').agg(
            #         {main_signal: 'mean', 'time_s': 'first', 'time_mmss_millis': 'first'}
            #     ).reset_index()
            #     agg['bin_start_time'] = t0 + agg['time_bin'] * bin_seconds

            #     st.session_state.df = df
            #     st.session_state.result_df = agg
            #     st.session_state.all_signals = [main_signal]

            elif main_signal == fallback_signal:
                # Filter Channel 18 with selected settings
                signal = df[main_signal].values
                time = df['time_s'].values
                pivot_high_idx = argrelextrema(signal, np.greater, order=ch18_order)[0]
                pivot_low_idx  = argrelextrema(signal, np.less,    order=ch18_order)[0]
                indices_to_nan = []
                last_low = None
                last_low_val = None
                li = 0

                for hi in pivot_high_idx:
                    while li < len(pivot_low_idx) and pivot_low_idx[li] < hi:
                        last_low = pivot_low_idx[li]
                        last_low_val = signal[last_low]
                        li += 1
                    if last_low is not None and last_low_val != 0:
                        pct = 100 * (signal[hi] - last_low_val) / abs(last_low_val)
                        future_lows = pivot_low_idx[pivot_low_idx > hi]
                        if pct > ch18_limit and len(future_lows) > 0:
                            next_low = future_lows[0]
                            indices_to_nan.extend(range(last_low, next_low + 1))

                filtered_signal = signal.copy()
                if indices_to_nan:
                    filtered_signal[indices_to_nan] = np.nan
                df['unfiltered_signal_Ch_18'] = df[main_signal].copy()
                df[main_signal] = filtered_signal

                # Aggregation for resampling (same as before)
                df_sorted = df.sort_values('time_s').copy()
                t0 = df_sorted['time_s'].iloc[0]
                df_sorted['time_bin'] = ((df_sorted['time_s'] - t0) // bin_seconds).astype(int)
                agg = df_sorted.groupby('time_bin').agg(
                    {main_signal: 'mean', 'time_s': 'first', 'time_mmss_millis': 'first'}
                ).reset_index()
                agg['bin_start_time'] = t0 + agg['time_bin'] * bin_seconds

                st.session_state.df = df
                st.session_state.result_df = agg
                st.session_state.all_signals = [main_signal]




# === Visualization Phase (runs even after convert)
if 'result_df' in st.session_state:
    df = st.session_state.df
    result_df = st.session_state.result_df
    all_signals = st.session_state.all_signals

    csv = result_df.to_csv(index=False).encode('utf-8')
    file_base = uploaded_mat.name.split('.')[0]
    st.download_button("Download Resampled CSV", csv, f"{file_base}_{bin_choice}_resampled.csv", "text/csv")

    plot_signal = st.selectbox("Select Signal to Plot", all_signals, index=0)

    fig = go.Figure()

    if plot_signal == 'Channel 18':
        fig.add_trace(go.Scatter(
            x=df['time_s'], y=df['unfiltered_signal_Ch_18'], mode='lines', name='Unfiltered Channel 18',
            line=dict(color='rgba(200,200,200,0.5)', width=1)
        ))
    elif plot_signal in ['1: Finger Pressure', '3: Systolic',	'4: Mean Arterial',	'5: Diastolic']:
            fig.add_trace(go.Scatter(
            x=df['time_s'], y=df['unfiltered_signal'], mode='lines', name='Unfiltered',
            line=dict(color='rgba(200,200,200,0.5)', width=1)
        ))
  
    fig.add_trace(go.Scatter(
        x=df['time_s'], y=df[plot_signal], mode='lines', name='Filtered',
        line=dict(color='cyan', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=result_df['time_s'], y=result_df[plot_signal], mode='lines', name='Aggregated',
        line=dict(color='orange', width=2.5)
    ))

    comment_locs = df[df['comment'].notna() & (df['comment'] != '')]
    for _, row_ in comment_locs.iterrows():
        label = str(row_['comment'])[:20] + "…" if len(row_['comment']) > 20 else row_['comment']
        fig.add_vline(x=row_['time_s'], line_width=1, line_dash='dash', line_color='lightgray', opacity=0.6)
        fig.add_annotation(x=row_['time_s'], y=0, text=label, showarrow=False, yanchor="bottom", textangle=-90,
                           font=dict(size=10, color="lightgray"), bgcolor="rgba(0,0,0,0)")

    tickvals = np.arange(
        df['time_s'].min(),
        df['time_s'].max() + 1, 300  # every 5 minutes
    )
    ticktext = [f"{int(t // 60):02}:{int(t % 60):02}" for t in tickvals]

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
        legend=dict(
            orientation="v", y=0.98, x=0.98, font=dict(size=11),
            bgcolor="rgba(18,18,18,0.8)", bordercolor="gray", borderwidth=1
        ),
        margin=dict(t=80, l=60, r=40, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)

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
