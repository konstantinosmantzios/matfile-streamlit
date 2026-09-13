# Supine-to-Standing Hemodynamic Analysis Methodology

This document outlines the methodology used by the Matfile Web App to compute hemodynamic statistics during active standing tests.

---

## 1. Preprocessing & Resampling

1. **AutoCal Masking**: The device's internal calibration periods (AutoCal) are automatically masked to prevent artifact contamination.
2. **Signal Filtering (Optional)**: Configurable filters (Savitzky-Golay, Butterworth, Hampel) can be applied to remove noise and outliers.
3. **Resampling**: The raw high-frequency signal is downsampled to extract the underlying blood pressure trend.
   - **Time-Based Resampling**: A sliding window average is computed at a fixed time interval (e.g., 1 Hz).
   - **Beat-Based Resampling**: Systolic peaks are detected, and the signal is averaged between consecutive heartbeats (beat-to-beat).

---

## 2. Event Markers

Temporal boundaries are defined by timestamped comments within the recording:
- **Baseline End Comment ("Ends At")**: The comment marking the end of the supine rest period (typically `Transition` or `Standing`).
- **Transition**: The moment the subject begins to stand.
- **Standing**: The moment the subject has fully achieved the standing position.

*Note: To ensure precision, exact comment timestamps are interpolated against the resampled signal for all statistical computations.*

---

## 3. Statistical Metrics

### 3.1 Baseline Mean
- **Interval**: From `(Ends At - Baseline Window)` to `Ends At`.
- **Computation**: The arithmetic mean of the resampled signal during the configured baseline window (e.g., the 30 seconds prior to Transition).

### 3.2 Analysis Regions

Three primary regions are analyzed to evaluate the orthostatic response. The duration of Method 1 and Method 2 is controlled by the **End Window** setting.

#### Method 1: Transition-to-End (Orange Region)
- **Interval**: From `Transition` to `Transition + End Window`.
- **Purpose**: Captures the full hemodynamic response from the onset of the postural change.

#### Method 2: Standing-to-End (Green Region)
- **Interval**: From `Standing` to `Standing + End Window`.
- **Purpose**: Captures the stabilization response excluding the mechanical transition period.

#### Method 3: Baseline Recovery (Blue Region)
- **Start**: The exact moment the pressure crosses down the Baseline Mean after the Transition ($y_{prev} \ge \text{Baseline} > y_{curr}$). Points where the curve is already below baseline without crossing down are ignored. Dips lasting < 0.5s are ignored as noise.
- **End**: The exact moment the pressure crosses up the Baseline Mean ($y_{prev} < \text{Baseline} \le y_{curr}$).
- **Validity**: If the pressure does not recover within 30 seconds after standing, the recovery is considered incomplete and duration/area stats are omitted.

### 3.3 Computed Statistics

For each of the three analysis regions, the following metrics are computed:

- **Duration (s)**: The total time span of the region.
- **Minimum Value (mmHg)**: The lowest pressure point (nadir) within the region.
- **Percentage Drop (%)**: The maximum percentage decrease relative to the reference value (Baseline Mean or the signal value at the start of the region).
- **Area Integral (mmHg·s)**: The cumulative area between the signal and the reference line, computed using the trapezoidal rule. It represents the total pressure deficit/excess over the time window.
