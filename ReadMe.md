---
title: LabChart MAT File Analyzer
emoji: 🫀
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.36.0"
app_file: readMatFile.py
pinned: false
license: mit
---

# LabChart Mat-File Analysis Tool

A Streamlit-based web application developed for the FAME Laboratory, Greece, to analyze continuous physiological data (like Finger Pressure, CBF, MAP, Heart Rate) extracted from LabChart `.mat` files.

This tool focuses on extracting and analyzing Supine-to-Standing cardiovascular transitions.

## Features
- **Upload & Read:** Seamlessly upload `.mat` files (supporting both old-style and HDF5 v7.3 MATLAB formats) and extract signal channels dynamically.
- **Data Filtering:** Clean raw physiological signals using interactive filters:
    - Savitzky-Golay (Smoothing)
    - Butterworth Low-Pass (Frequency filtering)
    - Hampel Filter (Spike/Outlier removal)
- **Interactive Visualization:** Explore the data with rich Plotly charts. You can zoom, pan, and hover to inspect values across the whole timeline.
- **Supine to Standing Analysis:**
    - The algorithm automatically parses LabChart comments to find `transition` and `stand` markers.
    - It extracts customizable "Baseline" segments before the transition.
    - It highlights the physiological drop (Orange Phase) and the recovery/standing phase (Green Phase).
- **Beat-Based & Time-Based Resampling:** Per-beat averages using peak-to-peak detection.
- **Excel Report Generation:** Multi-sheet `.xlsx` export with stats, filtered data, and plots.

## Usage

1. Upload your `.mat` file from the sidebar.
2. Configure filters and resampling parameters.
3. Explore the interactive charts and export your report.

## Contact / Authors
© 2026 FAME Laboratory, Greece.
Contact: K. Mantzios | G. Gkikas
