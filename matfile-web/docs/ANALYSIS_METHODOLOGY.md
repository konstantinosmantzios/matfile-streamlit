# Supine-to-Standing Hemodynamic Analysis Methodology

This document outlines the methodology used by the Matfile Web App to preprocess signals, identify key postural milestones, compute hemodynamic statistics during active standing tests, and export report data.

---

## 1. Preprocessing & Resampling

1. **AutoCal Masking**: The Finapres/hemodynamic device's periodic internal calibration cycles (AutoCal / Physiocal) produce artificial stepped plateaus and spikes. These intervals are automatically detected (via the AutoCal channel or comment annotations) and masked with `NaN` to prevent contaminating downstream analysis.
2. **Signal Filtering (Optional)**: Configurable digital filters can be applied to the raw high-frequency waveforms prior to peak detection and resampling:
   - **Savitzky-Golay**: Preserves waveform peak morphology while smoothing high-frequency noise.
   - **Butterworth Low-Pass**: Zero-phase, forward-backward low-pass filter (default 5 Hz cutoff, 4th order).
   - **Hampel Identifier**: Windowed median and Median Absolute Deviation (MAD) filter that replaces transient spikes and artifacts.
3. **Centered Resampling (Zero-Phase)**: Raw continuous physiological signals recorded at high acquisition rates (e.g., 200 Hz – 1000 Hz) are downsampled to extract the underlying mean trend while eliminating cardiac pulsatility. **All resampling in the application is strictly centered**:
   - **What is "Centered" Resampling?**
     In standard causal (trailing) sliding windows, the value calculated across an interval is assigned to the *end* (right edge) of that interval. While computationally simple, trailing windows introduce an artificial **phase lag (time delay)** equal to half the window width, shifting physiological drops and inflection points forward in time. Conversely, leading windows shift features backward. 
     **Centered resampling** assigns the average value of a window to the **exact temporal midpoint (center)** of that window. This zero-phase alignment ensures that physiological events (such as the sudden blood pressure nadir upon standing or the recovery inflection point) remain perfectly synchronized with actual event markers and concurrent physiological channels.
   
   - **Time-Based Centered Resampling**:
     - The continuous recording is divided into uniform time intervals of duration $\Delta t$ (configured by the *Time Window*, e.g., 1.0 s).
     - Signal samples within each bin are averaged (arithmetic mean), and the resulting value is stamped at the **mean timestamp** of the samples in that bin.
     - *Example*: For a 1.0-second window spanning $t = 10.0\text{ s}$ to $t = 11.0\text{ s}$ recorded at 200 Hz (200 raw samples), all values in $[10.0, 11.0]\text{ s}$ are averaged. The resulting resampled data point is placed at the center timestamp $t = 10.5\text{ s}$. If trailing averaging were used, it would appear at $t = 11.0\text{ s}$, creating a 0.5-second artificial latency.
   
   - **Beat-Based Centered Resampling**:
     - Systolic peaks (heartbeats) are detected dynamically across the filtered pressure or CBF waveform.
     - A moving window of $K$ consecutive beats (configured by the *Beat Window*, e.g., 5 beats) is averaged.
     - The window is symmetrically centered around the index of the target heartbeat:
       - For an odd beat count $K$ (e.g., $K = 5$, where $\text{half\_k} = 2$), the window spans 2 beats prior to 2 beats following the center beat (spanning peaks $i - 2$ to $i + 3$).
       - The mean value of all raw samples across this multi-beat interval is computed and assigned directly to the timestamp of the **central heartbeat peak $p_i$**.
     - *Example*: Consider a 5-beat average ($K = 5$) evaluated around beat #10 located at $t = 12.0\text{ s}$. The averaging window spans across the preceding 2 beats (from beat #8 at $t = 10.2\text{ s}$) through the succeeding 2 beats (to beat #13 at $t = 13.8\text{ s}$). The resulting average pressure is stamped at the exact timestamp of beat #10 ($t = 12.0\text{ s}$). This centers the beat-averaged trend over the cardiac cycle without lagging behind postural transitions.

---

## 2. Event Markers

Temporal boundaries are determined from timestamped comment markers embedded within the recording:
- **Baseline End Comment ("Baseline Ends At")**: The reference comment denoting the end of the steady-state supine rest period (typically `Transition` or `Standing`).
- **Transition**: The precise timestamp when the subject initiates active postural change from supine to standing.
- **Standing**: The precise timestamp when the subject has fully achieved the erect standing position.

*Note: To achieve sub-sample accuracy, marker timestamps are interpolated against the resampled time vector.*

---

## 3. Statistical Metrics & Exported Report

### 3.1 Baseline Mean
- **Interval**: Spans from `(Baseline Ends At - Baseline Window)` to `Baseline Ends At`.
- **Computation**: The arithmetic mean of the resampled signal across this steady-state window (typically the 30 seconds preceding postural transition).

### 3.2 Analysis Regions

Three complementary analytical methods evaluate the orthostatic hemodynamic response:

#### Method 1: Transition-to-End (Orange Region)
- **Interval**: From `Transition` to `Transition + End Window`.
- **Purpose**: Evaluates the total orthostatic response encompassing both the active physical exertion of standing and initial upright stabilization.

#### Method 2: Standing-to-End (Green Region)
- **Interval**: From `Standing` to `Standing + End Window`.
- **Purpose**: Focuses exclusively on post-standing stabilization and reflex compensation, eliminating the initial mechanical artifacts of standing up.

#### Method 3: Baseline Recovery (Blue Region)
- **Start**: The exact interpolated time when the signal crosses **down** below the Supine Baseline Mean ($y_{prev} \ge \text{Baseline} > y_{curr}$) following the Transition. Transient dips lasting $< 0.5\text{ s}$ are filtered out as artifactual noise.
- **End**: The exact interpolated time when the signal crosses **up** back to or above the Supine Baseline Mean ($y_{prev} < \text{Baseline} \le y_{curr}$).
- **Validity & Override**: If the signal fails to return to baseline within 30 seconds following the Standing marker, recovery is considered incomplete (stats remain blank). Users can manually adjust or specify the end recovery point using interactive marker drag-and-drop.

---

### 3.3 Exported Excel Workbook Structure

When exporting data, the application generates a styled Microsoft Excel workbook (`Analysis_<FileName>_<LocalTimestamp>.xlsx`) containing three sheets:

1. **`Statistics`**: Consolidated summary table containing all computed hemodynamic metrics for every test, signal, and analysis method.
2. **`Resampled_Data`**: The continuous resampled time series (with columns for `time_s`, `comment_text`, `absolute_time`, and each resampled signal), allowing researchers to inspect or plot the exact curves used for analysis.
3. **`Metadata`**: Complete audit log of all active settings, including resample mode, baseline windows, filter configurations, and the count of manual marker overrides applied.

---

### 3.4 Detailed Explanation of the `Statistics` Sheet Columns

Each row in the **`Statistics`** sheet represents an analyzed region for a given standing test and physiological signal. The columns are structured as follows:

| Column Header | Data Type / Units | Description |
| :--- | :--- | :--- |
| **`Test ID`** | Integer (1, 2, ...) | Sequential identifier for each active standing test detected in the file (paired `Transition` and `Standing` comments). |
| **`Signal`** | Text | The physiological channel analyzed in this row: <br>• `FP (Resampled)`: Continuous Finger Blood Pressure.<br>• `FP (MAP 5s)`: 5-second Gaussian-filtered MAP curve.<br>• `CBF`: Cerebral Blood Flow velocity. |
| **`Method`** | Text | The analytical window applied: <br>• `Method 1: Transition-to-End`<br>• `Method 2: Standing-to-End`<br>• `Method 3: Baseline Recovery` |
| **`Transition Duration (s)`** | Seconds ($\text{s}$) | Physical duration of the postural change, calculated as: <br>$$t_{\text{standing}} - t_{\text{transition}}$$ |
| **`Duration (s)`** | Seconds ($\text{s}$) | The elapsed time span of the active analysis region: <br>• **Method 1**: Duration from `Transition` to `Transition + End Window`.<br>• **Method 2**: Duration from `Standing` to `Standing + End Window`.<br>• **Method 3**: Net recovery time ($t_{\text{rec\_end}} - t_{\text{rec\_start}}$). Blank if recovery did not complete within the 30-second post-standing limit. |
| **`Drop (%)`** | Percentage ($\%$) | Maximum relative percentage decrease from the reference level to the lowest point (nadir): <br>$$\text{Drop (\%)} = \frac{\text{Reference Value} - \text{Minimum Value}}{\lvert \text{Reference Value} \rvert} \times 100$$ |
| **`Min`** | Signal Units ($\text{mmHg}$ or $\text{cm/s}$) | The absolute minimum value (nadir) reached by the signal within the analysis interval. |
| **`Area Above Curve (AUC)`** | $\text{mmHg}\cdot\text{s}$ or $(\text{cm/s})\cdot\text{s}$ | The cumulative area between the reference level and the signal curve when the signal falls below reference, computed via the numerical **trapezoidal rule**: <br>$$\text{AUC} = \int_{t_{\text{start}}}^{t_{\text{end}}} \max(0, y_{\text{ref}} - y(t)) \, dt$$ <br>Quantifies the total hemodynamic deficit (depth $\times$ duration of hypotension). |
| **`Start / Baseline Value`** | Signal Units ($\text{mmHg}$ or $\text{cm/s}$) | The reference value used as the baseline for the region: <br>• **Method 1**: Interpolated signal value at `Transition` (or Baseline Mean if configured).<br>• **Method 2**: Interpolated signal value at `Standing` (or Baseline Mean if configured).<br>• **Method 3**: The Supine Baseline Mean. |
| **`End Value`** | Signal Units ($\text{mmHg}$ or $\text{cm/s}$) | The signal value at the termination of the analysis window: <br>• **Method 1 & 2**: Interpolated signal value at the end of the configured End Window.<br>• **Method 3**: Interpolated signal value at the point of upward baseline crossing ($t_{\text{rec\_end}}$). |
| **`Started In`** | Category (`"Transition"` / `"Standing"`) | *Applicable to Method 3 only*: Categorizes whether the blood pressure began its initial drop below baseline during the physical transition motion (`t_rec_start < t_standing`) or after the subject had already achieved full upright standing (`t_rec_start >= t_standing`). (Blank for Methods 1 & 2). |
