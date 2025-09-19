# Standard Operating Procedure (SOP)  
### Streamlit MAT File Conversion & Filtering Tool

&nbsp;  
&nbsp;  

## 📌 Purpose of the Tool
This tool converts physiological data stored in MATLAB `.mat` files into an easy-to-use format (Excel).  
It also allows **filtering** and **resampling** of the signals so they can be cleaned, averaged, and better visualized.  

The tool is designed for **students and researchers with little or no programming background**.  
Everything happens through the **Streamlit web interface**:  
- Upload a file  
- Adjust a few settings  
- Download clean Excel files  

&nbsp;  


## ⚙️ How the Tool Works
1. Upload a `.mat` file.  
2. The tool automatically detects key signals:  
   - **Finger Pressure**  
   - **Cerebral Blood Flow (CBF)**  
   - and other derived signals  
3. Calibration phases in **Finapress** are automatically removed when possible.  
4. Choose **filtering methods** for Finger Pressure and/or CBF to remove artifacts.  
5. Resample the data:  
   - **Time-based** (e.g., every 1 sec, 5 sec, 1 min)  
   - **Beat-based** (e.g., 5 or 10 heartbeats)  
6. Export the results into **Excel** with:  
   - Timestamps  
   - Elapsed time  
   - Comments  
   - Selected signals  

&nbsp;  

## 🔎 Filtering Methods Explained
Filtering is **optional**. If unsure, start with defaults.

### 1. No Filter
- Leaves the signal exactly as recorded.  
- Use this if you trust the raw data.

---

### 2. Jump Filter  
Removes sudden, unrealistic jumps in the signal caused by **sensor noise or movement**.  

**a. Jump Threshold (size of change)**  
- Defines how big a sudden change must be to count as an artifact.  
- If the signal changes more than this threshold between two points, the point is flagged as invalid.  

- Lower values → **more strict** (even small fluctuations removed).  
- Higher values → **more tolerant** (only very large spikes removed).  

**b. Close Jump Window (gap between jumps)**  
- Looks at how close together multiple jumps are.  
- If two jumps happen within this many samples, the **entire section between them is deleted**.  

- Lower values → **narrow effect** (only the exact jumps removed).  
- Higher values → **wider effect** (whole section between close jumps removed).  

**Examples (at 200 Hz sampling):**  
- 100 samples ≈ **0.5 sec** → removes very short noisy bursts.  
- 500 samples ≈ **2.5 sec** → balanced (default).  
- 1000 samples ≈ **5 sec** → aggressive, removes long sections if multiple jumps occur close together.  

---

### 3. Median Filter (Advanced for CBF)  
Suppresses small **spikes (upward)** or **dips (downward)** in the CBF signal while preserving the overall shape.  

**Parameters:**  

- **Median Window Length (smoothing strength)**  
  - Defines how wide the averaging window is for smoothing.  
  - Lower values → lighter smoothing (**keeps detail but leaves noise**).  
  - Higher values → stronger smoothing (**removes noise but may blur sharp changes**).  

- **Positive Spike Threshold (upward spikes)**  
  - Defines strictness for suppressing sudden **upward spikes**.  
  - Lower values → stricter (removes even small upward jumps).  
  - Higher values → more tolerant (keeps detail, but large spikes may remain).  

- **Negative Spike Threshold (downward dips)**  
  - Defines strictness for suppressing sudden **downward dips**.  
  - Lower values → stricter (removes even small dips).  
  - Higher values → more tolerant (keeps detail, but some dips may remain).  

&nbsp;  


## 📉 Resampling Options
Since raw signals are at **200 Hz** (very high frequency), you can reduce them into manageable summaries:  

- **Time-based**: average signal every fixed interval (e.g., 1 sec, 10 sec, 1 min).  
  *Useful for long recordings.*  

 - **Beat-based**: produces one row for **each detected beat** (separately for Finger Pressure and CBF).  
   - Each channel keeps its own beat timing (Finapress and CBF beats do not align exactly).  
   - For every beat, the value shown is the **mean of the previous N detected beats** (e.g., 5 or 10).  
   - This way, each row represents a real detected beat, with smoothed values that reflect the recent history of that signal.  
   *Useful for comparing cardiac cycles directly while accounting for natural timing differences.*  

⚠️ **Important**: Comments are preserved at their exact time, even if no other data is present.  

&nbsp;  

## 📂 Exported Excel Sheets
- **Filtered(200Hz)**  
  - Original signals at high resolution (200 Hz).  
  - Includes applied filters and comments.  

- **Resampled (Time or Beats)**  
  - Cleaner, downsampled signals for easier analysis.  
  - Includes timestamps, elapsed time, comments, and all selected signals.  

&nbsp;  

## 👩‍💻 Notes for Non-Experts
- If you are not sure which filter to use, **start with defaults**.  
- Use **No Filter** if the raw data seem clean (no obvious outliers or noise).  
- Use **Jump Filter** if you see unrealistic jumps or spikes.  
- Use **Median Filter** for CBF if there is high-frequency noise or small sharp spikes/dips.
&nbsp;  
&nbsp;  
&nbsp;  
&nbsp; 
&nbsp;  
&nbsp;  
