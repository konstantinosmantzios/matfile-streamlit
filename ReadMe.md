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
- **Beat-Based & Time-Based Resampling:** Optionally resample the data using the natural peak intervals of the primary signal (e.g., peak-to-peak beat detection) to compute per-beat averages, instead of raw 1000Hz continuous data.
- **Excel Report Generation:** Export a highly-formatted, multi-sheet `.xlsx` report that includes:
    - The raw (filtered) data.
    - The resampled/beat-based data.
    - Extracted Transition Statistics (Drop %, Area Under/Over Curve, Min values).
    - Headless plots capturing the exact visual snapshot of each transition test.

## Installation

1. Clone or download this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Open a terminal and navigate to the folder containing `app.py`.
2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. A browser window will open automatically (typically at `http://localhost:8501`).
4. Upload your `.mat` file from the sidebar and configure your parameters.

## Directory Structure
- `app.py`: The main Streamlit web application.
- `export.py`: Handles the automated backend Excel and Plot generation logic.
- `requirements.txt`: Python package dependencies.
- `readMatFile.py`: Helper script / older scripts (if applicable).

## Contact / Authors
© 2026 FAME Laboratory, Greece.
Contact: K. Mantzios | G. Gkikas
