import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { Activity, UploadCloud, Download, Trash2, Settings2 } from 'lucide-react';
import Plot from 'react-plotly.js';
import FilterSandbox from './components/FilterSandbox';

const API_BASE_URL = 'http://127.0.0.1:8000/api';
const VIEWPORT_DEBOUNCE_MS = 300;
const SETTINGS_DEBOUNCE_MS = 600;
const TARGET_POINTS = 3000;

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [fileName, setFileName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isViewportLoading, setIsViewportLoading] = useState(false);
  const [error, setError] = useState('');
  const [plotData, setPlotData] = useState([]);
  const [layoutShapes, setLayoutShapes] = useState([]);
  const [layoutAnnotations, setLayoutAnnotations] = useState([]);
  const [availableSignals, setAvailableSignals] = useState([]);
  const [tests, setTests] = useState([]);
  const [memoryUsage, setMemoryUsage] = useState(null);
  const [memoryInfo, setMemoryInfo] = useState(null);
  const [analysisStats, setAnalysisStats] = useState([]);
  const [plotXRange, setPlotXRange] = useState({ min: 0, max: 1 });

  // Refs
  const initialPlotDataRef = useRef([]);
  const viewportTimeoutRef = useRef(null);
  const currentZoomRef = useRef(null);
  const settingsDebounceRef = useRef(null);
  const isFirstRender = useRef(true);
  const isUploadingRef = useRef(false);
  const latestSessionIdRef = useRef(null);
  const fileInputRef = useRef(null);
  const analysisStatsRef = useRef([]);
  const initialXRangeRef = useRef({ min: 0, max: 1 });

  useEffect(() => {
    analysisStatsRef.current = analysisStats;
  }, [analysisStats]);

  // Settings
  const [settings, setSettings] = useState({
    resampleMode: "Beat-based",
    resampleRateTime: 1,
    resampleRateBeat: 5,
    autoCalOption: "Auto-Detect",
    fpFilter: "Savitzky-Golay",
    fpSavgolWin: 51,
    fpSavgolPoly: 5,
    fpButterCutoff: 5.0,
    fpButterOrder: 4,
    fpHampelWin: 5,
    fpHampelSig: 3.0,
    cbfFilter: "Butterworth Low-Pass",
    cbfSavgolWin: 51,
    cbfSavgolPoly: 5,
    cbfButterCutoff: 5.0,
    cbfButterOrder: 4,
    cbfHampelWin: 5,
    cbfHampelSig: 3.0,
    analysisView: "Filtering Preview",
    baselineWinStart: -60,
    baselineWinDuration: 30,
    standingWinStart: 10,
    standingWinDuration: 30,
    analysisBaselineWindow: 60,
    baselineEndComment: "Transition",
    analysisEndMarkerWindow: 10,
    devMode: false,
    compareGaussian: false
  });

  useEffect(() => { latestSessionIdRef.current = sessionId; }, [sessionId]);

  // ---- Memory polling ------------------------------------------------
  useEffect(() => {
    const fetchMemory = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/memory`);
        setMemoryUsage(response.data.memory_mb);
        setMemoryInfo(response.data);
      } catch (err) { /* ignore */ }
    };
    fetchMemory();
    const interval = setInterval(fetchMemory, 3000);
    return () => clearInterval(interval);
  }, []);

  // ---- Auto-apply settings on change (debounced) ----------------------
  useEffect(() => {
    if (isFirstRender.current) { isFirstRender.current = false; return; }
    if (isUploadingRef.current) return;
    if (!latestSessionIdRef.current) return;

    if (settingsDebounceRef.current) clearTimeout(settingsDebounceRef.current);
    settingsDebounceRef.current = setTimeout(() => {
      processData(latestSessionIdRef.current, settings);
    }, SETTINGS_DEBOUNCE_MS);
    return () => { if (settingsDebounceRef.current) clearTimeout(settingsDebounceRef.current); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings]);

  // ---- Plotly zoom handler --------------------------------------------
  const handleRelayout = useCallback((eventData) => {
    if (!sessionId) return;

    if (eventData['xaxis.autorange']) {
      currentZoomRef.current = null;
      setPlotData([...initialPlotDataRef.current]);
      setPlotXRange(initialXRangeRef.current);
      return;
    }

    let xMin = eventData['xaxis.range[0]'];
    let xMax = eventData['xaxis.range[1]'];
    if ((xMin === undefined || xMax === undefined) && Array.isArray(eventData['xaxis.range'])) {
      xMin = eventData['xaxis.range'][0];
      xMax = eventData['xaxis.range'][1];
    }
    if (!xMin || !xMax) return;

    const xMinMs = typeof xMin === 'number' ? xMin : new Date(xMin).getTime();
    const xMaxMs = typeof xMax === 'number' ? xMax : new Date(xMax).getTime();
    if (isNaN(xMinMs) || isNaN(xMaxMs)) return;

    currentZoomRef.current = { xMinMs, xMaxMs };
    setPlotXRange({ min: xMinMs, max: xMaxMs });
    
    if (settings.analysisView === 'Supine to Standing Analysis') {
      return;
    }

    if (viewportTimeoutRef.current) clearTimeout(viewportTimeoutRef.current);
    viewportTimeoutRef.current = setTimeout(() => {
      fetchViewport(xMinMs, xMaxMs);
    }, VIEWPORT_DEBOUNCE_MS);
  }, [sessionId, settings.analysisView]);

  const fetchViewport = async (xMinMs, xMaxMs, sid = sessionId) => {
    if (!sid) return;
    setIsViewportLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/viewport`, {
        session_id: sid,
        x_min: xMinMs,
        x_max: xMaxMs,
        target_points: TARGET_POINTS,
      });
      const viewportTraces = response.data.traces;
      const viewportIds = new Set(viewportTraces.map(t => t.trace_id));
      setPlotData(prev => {
        const prevMap = new Map(prev.map(t => [t.trace_id, t]));
        
        // Preserve 'visible' property if the user toggled it in the legend
        const updatedViewportTraces = viewportTraces.map(t => {
          if (prevMap.has(t.trace_id) && prevMap.get(t.trace_id).visible !== undefined) {
            return { ...t, visible: prevMap.get(t.trace_id).visible };
          }
          return t;
        });

        const kept = prev.filter(t => !viewportIds.has(t.trace_id));
        return [...updatedViewportTraces, ...kept];
      });
    } catch (err) { console.error(err); } 
    finally { setIsViewportLoading(false); }
  };

  // ---- File upload ----------------------------------------------------
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    isUploadingRef.current = true;
    setIsLoading(true);
    setError('');
    setAnalysisStats([]);
    
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setSessionId(response.data.session_id);
      setFileName(response.data.file_name);
      setTests(response.data.tests || []);
      
      let updatedSettings = { ...settings };
      if (response.data.tests && response.data.tests.length > 0) {
        updatedSettings = { ...updatedSettings, testStartS: response.data.tests[0].start_s, testEndS: response.data.tests[0].end_s };
        setSettings(updatedSettings);
      }
      await processData(response.data.session_id, updatedSettings);
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed');
      setIsLoading(false);
    } finally {
      isUploadingRef.current = false;
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // ---- Process data ---------------------------------------------------
  const processData = async (sid = sessionId, currentSettings = settings) => {
    if (!sid) return;
    setIsLoading(true);
    setError('');
    try {
      const response = await axios.post(`${API_BASE_URL}/process`, { session_id: sid, settings: currentSettings });
      const traces = response.data.traces || [];
      setPlotData(traces);
      initialPlotDataRef.current = traces;
      setLayoutShapes(response.data.shapes || []);
      setLayoutAnnotations(response.data.annotations || []);
      setAnalysisStats(response.data.analysis_stats || []);
      
      let initialMin = null, initialMax = null;
      traces.forEach(t => {
        if (t.x && t.x.length > 0) {
          const min = t.x[0];
          const max = t.x[t.x.length - 1];
          if (initialMin === null || min < initialMin) initialMin = min;
          if (initialMax === null || max > initialMax) initialMax = max;
        }
      });
      if (initialMin !== null && initialMax !== null) {
        const rng = { min: initialMin, max: initialMax };
        initialXRangeRef.current = rng;
        if (!currentZoomRef.current) {
          setPlotXRange(rng);
        }
      }
      
      if (response.data.available_signals) {
        setAvailableSignals(response.data.available_signals);
        if (!currentSettings.selectedSignal && response.data.available_signals.length > 0) {
          const defaultSig = response.data.available_signals.find(s => s.includes('Finger Pressure')) || response.data.available_signals[0];
          setSettings(prev => ({ ...prev, selectedSignal: defaultSig }));
        }
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Processing failed');
    } finally {
      setIsLoading(false);
      if (currentZoomRef.current && currentSettings.analysisView === 'Filtering Preview') {
        setTimeout(() => { fetchViewport(currentZoomRef.current.xMinMs, currentZoomRef.current.xMaxMs, sid); }, 50);
      }
    }
  };

  // ---- Export & Clear --------------------------------------------------
  const exportData = async () => {
    if (!sessionId) return;
    setIsLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/export`, { session_id: sessionId, settings }, { responseType: 'blob' });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `matfile_analysis_${fileName.replace('.mat','')}.xlsx`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      setError('Export failed');
    } finally {
      setIsLoading(false);
    }
  };

  const clearSession = async () => {
    if (sessionId) {
      try { await axios.delete(`${API_BASE_URL}/cleanup/${sessionId}`); } catch (err) {}
    }
    setSessionId(null);
    setFileName('');
    setPlotData([]);
    initialPlotDataRef.current = [];
    currentZoomRef.current = null;
    setLayoutShapes([]);
    setLayoutAnnotations([]);
    setAvailableSignals([]);
    setTests([]);
    setError('');
  };

  const updateSetting = (k, v) => setSettings(prev => ({ ...prev, [k]: v }));
  const isFilteringPreview = settings.analysisView === 'Filtering Preview';

  const getPlottedHz = () => {
    if (!plotData || plotData.length === 0) return null;
    const rawTrace = plotData.find(t => t.trace_id === 'raw' || t.name === 'Raw Data');
    if (!rawTrace || !rawTrace.x || rawTrace.x.length < 2) return null;
    const pts = rawTrace.x.length;
    const durationS = (rawTrace.x[rawTrace.x.length - 1] - rawTrace.x[0]) / 1000;
    if (durationS <= 0) return null;
    return Math.min(200, pts / durationS).toFixed(1);
  };

  return (
    <div className="app-container">
      <main className="main-content">
        
        {/* Top Header */}
        <div className="glass-panel flex-between top-header" style={{ padding: '16px 24px' }}>
          <div className="flex-row">
            <Activity size={24} color="var(--accent-blue)" />
            <h1 style={{ margin: 0 }}>MAT Analyzer</h1>
            
            {/* Memory & Resolution Badges */}
            <div className="flex-row" style={{ marginLeft: 16 }}>
              {memoryUsage !== null && (
                <span className="badge">
                  Server: {memoryUsage.toFixed(0)} MB
                </span>
              )}
              {getPlottedHz() !== null && (
                <span className="badge badge-active">
                  Res: ~{getPlottedHz()} Hz
                </span>
              )}
            </div>
          </div>
          
          <div className="flex-row">
            <div className="flex-row" style={{ marginRight: 16 }}>
              <label style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', cursor: 'pointer', gap: 6 }}>
                <input 
                  type="checkbox" 
                  checked={settings.devMode} 
                  onChange={e => updateSetting('devMode', e.target.checked)} 
                  style={{ cursor: 'pointer' }}
                />
                Dev Mode
              </label>
            </div>
            {fileName && <span className="badge">Active File: {fileName}</span>}
            {sessionId && (
              <button className="btn btn-danger" onClick={clearSession} disabled={isLoading} style={{ marginLeft: 16, padding: '6px 12px' }}>
                <Trash2 size={16} /> Close
              </button>
            )}
          </div>
        </div>

        {error && (
          <div className="glass-panel" style={{ borderColor: 'var(--accent-red)', padding: '16px' }}>
            <p style={{ color: 'var(--accent-red)', margin: 0, fontWeight: 500 }}>{error}</p>
          </div>
        )}

        {!sessionId && (
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 1 }}>
            <input ref={fileInputRef} type="file" accept=".mat" onChange={handleFileUpload} style={{ display: 'none' }} id="file-upload" />
            <label htmlFor="file-upload" className="upload-zone" style={{ width: '100%', maxWidth: 500 }}>
              <UploadCloud className="upload-icon" />
              <div>
                <h2>Upload LabChart MAT File</h2>
                <p>Drag and drop or click to browse files</p>
              </div>
              {isLoading && (
                <div style={{ marginTop: 16, display: 'flex', alignItems: 'center', gap: 8, color: 'var(--accent-blue)', fontWeight: 600 }}>
                  <div className="spinner" style={{ width: 20, height: 20, borderWidth: 2, marginBottom: 0 }}></div>
                  Processing File...
                </div>
              )}
            </label>
          </div>
        )}

        {sessionId && (
          <>


            {/* Test Selector for Analysis Mode */}
            {!isFilteringPreview && tests && tests.length > 0 && (
              <div className="glass-panel flex-row" style={{ padding: '16px 24px' }}>
                <span style={{ fontWeight: 600 }}>Select Test:</span>
                <select 
                  style={{ minWidth: 200 }}
                  value={tests.findIndex(t => t.start_s === settings.testStartS)} 
                  onChange={e => {
                    const idx = parseInt(e.target.value);
                    if(idx >= 0 && tests[idx]) {
                      setSettings(prev => ({ ...prev, testStartS: tests[idx].start_s, testEndS: tests[idx].end_s }));
                    }
                  }}
                >
                  {tests.map((test, idx) => (
                    <option key={idx} value={idx}>Test {idx + 1} ({test.start_time})</option>
                  ))}
                </select>
              </div>
            )}

            {/* Plot Container */}
            <div className="glass-panel plot-container" style={{ padding: '24px', minWidth: 0, overflow: 'hidden' }}>
              
              {/* Overlay Loading State */}
              {(isLoading || isViewportLoading) && (
                <div className="loading-overlay">
                  <div className="spinner"></div>
                  <h3 style={{ margin: 0, color: 'var(--accent-blue)' }}>Processing Data</h3>
                  <p style={{ marginTop: 4 }}>Applying filters and recalculating...</p>
                </div>
              )}

              {/* Plot Container Header (Controls + Signals) */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 16, borderBottom: '1px solid var(--border-color)', paddingBottom: 16, marginBottom: 24 }}>
                
                {/* Left: View Toggle */}
                <div style={{ display: 'flex', background: 'rgba(0,0,0,0.05)', borderRadius: '12px', padding: '4px' }}>
                  <button
                    onClick={() => updateSetting('analysisView', 'Filtering Preview')}
                    style={{
                      padding: '6px 16px', fontSize: '13px', borderRadius: '8px', border: 'none', cursor: 'pointer',
                      background: settings.analysisView === 'Filtering Preview' ? '#fff' : 'transparent',
                      color: settings.analysisView === 'Filtering Preview' ? 'var(--text-main)' : 'var(--text-muted)',
                      fontWeight: 600,
                      boxShadow: settings.analysisView === 'Filtering Preview' ? '0 2px 4px rgba(0,0,0,0.05)' : 'none'
                    }}
                  >
                    Filtering
                  </button>
                  <button
                    onClick={() => updateSetting('analysisView', 'Supine to Standing Analysis')}
                    style={{
                      padding: '6px 16px', fontSize: '13px', borderRadius: '8px', border: 'none', cursor: 'pointer',
                      background: settings.analysisView === 'Supine to Standing Analysis' ? '#fff' : 'transparent',
                      color: settings.analysisView === 'Supine to Standing Analysis' ? 'var(--text-main)' : 'var(--text-muted)',
                      fontWeight: 600,
                      boxShadow: settings.analysisView === 'Supine to Standing Analysis' ? '0 2px 4px rgba(0,0,0,0.05)' : 'none'
                    }}
                  >
                    Analysis
                  </button>
                </div>

                {/* Center: Signal Pills */}
                {availableSignals.length > 0 && (
                  <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', flex: 1, justifyContent: 'center' }}>
                    {availableSignals
                      .filter(sig => {
                        const s = sig.toLowerCase();
                        return s.includes('finger') || s.includes('map') || s.includes('hr') || s.includes('cbf');
                      })
                      .map(sig => {
                      const isSelected = settings.selectedSignal === sig;
                      let shortName = sig;
                      if (sig.includes('Finger Pressure')) shortName = 'Finger Pressure';
                      if (sig.includes('CBF')) shortName = 'CBF';
                      if (sig.includes('MAP')) shortName = 'MAP';
                      if (sig.includes('HR')) shortName = 'HR';
                      
                      return (
                        <button
                          key={sig}
                          onClick={() => updateSetting('selectedSignal', sig)}
                          style={{
                            padding: '6px 16px', borderRadius: '20px', fontSize: '13px', fontWeight: '600', cursor: 'pointer', border: '1px solid',
                            borderColor: isSelected ? 'var(--accent-blue)' : 'var(--border-color)',
                            background: isSelected ? 'var(--accent-blue)' : '#fff',
                            color: isSelected ? '#fff' : 'var(--text-muted)',
                            transition: 'all 0.2s'
                          }}
                        >
                          {shortName}
                        </button>
                      )
                    })}
                  </div>
                )}

                {/* Right: Resampling & Export */}
                <div className="flex-row" style={{ gap: 16 }}>
                  <div className="flex-row" style={{ gap: 8 }}>
                    <select className="form-select form-select-sm" value={settings.resampleMode} onChange={e => updateSetting('resampleMode', e.target.value)} style={{ padding: '6px 12px', borderRadius: '8px', border: '1px solid var(--border-color)', outline: 'none' }}>
                      <option>Beat-based</option>
                      <option>Time-based</option>
                    </select>
                    
                    {settings.resampleMode === 'Time-based' ? (
                      <div className="flex-row" style={{ gap: 12 }}>
                        <span style={{ fontSize: 13, color: 'var(--text-muted)', width: 45, textAlign: 'right' }}>
                          {settings.resampleRateTime === 60 ? '1 min' : `${settings.resampleRateTime}s`}
                        </span>
                        <input 
                          type="range" min="0" max="5" step="1" 
                          value={[1, 5, 10, 15, 30, 60].indexOf(settings.resampleRateTime) !== -1 ? [1, 5, 10, 15, 30, 60].indexOf(settings.resampleRateTime) : 0} 
                          onChange={e => updateSetting('resampleRateTime', [1, 5, 10, 15, 30, 60][parseInt(e.target.value)])} 
                          style={{ width: 160 }} 
                        />
                      </div>
                    ) : (
                      <div 
                        className="flex-row" 
                        style={{ gap: 12 }} 
                        title="The resampling window perfectly centers on a beat (for even intervals) or between beats (for odd intervals) using N beats + 1 peak."
                      >
                        <span style={{ fontSize: 13, color: 'var(--text-muted)', width: 55, textAlign: 'right' }}>
                          {settings.resampleRateBeat} beats
                        </span>
                        <input 
                          type="range" min="0" max="3" step="1" 
                          value={[1, 2, 5, 10].indexOf(settings.resampleRateBeat) !== -1 ? [1, 2, 5, 10].indexOf(settings.resampleRateBeat) : 2} 
                          onChange={e => updateSetting('resampleRateBeat', [1, 2, 5, 10][parseInt(e.target.value)])} 
                          style={{ width: 160 }} 
                        />
                      </div>
                    )}
                  </div>

                  <div style={{ width: '1px', height: '24px', background: 'var(--border-color)' }}></div>
                  
                  <button className="btn btn-primary" onClick={exportData} disabled={isLoading}>
                    <Download size={16} /> Export
                  </button>
                </div>
              </div>
              
              {/* Analysis Settings Row */}
              {settings.analysisView === 'Supine to Standing Analysis' && (
                <div className="flex-row" style={{ width: '100%', justifyContent: 'center', marginTop: 12, gap: 24, flexWrap: 'wrap' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-muted)' }}>Baseline Window (s):</span>
                    <input 
                      type="number"
                      value={settings.analysisBaselineWindow}
                      onChange={e => updateSetting('analysisBaselineWindow', parseInt(e.target.value) || 0)}
                      style={{ width: 60, padding: '2px 6px', borderRadius: 4, border: '1px solid var(--border-color)', background: 'var(--bg-main)', color: 'var(--text-main)', fontSize: 13 }}
                    />
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-muted)' }}>Baseline Ends At:</span>
                    <select 
                      value={settings.baselineEndComment}
                      onChange={e => updateSetting('baselineEndComment', e.target.value)}
                      style={{ padding: '2px 6px', borderRadius: 4, border: '1px solid var(--border-color)', background: 'var(--bg-main)', color: 'var(--text-main)', fontSize: 13 }}
                    >
                      <option value="Transition">Transition Comment</option>
                      <option value="Standing">Standing Comment</option>
                    </select>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-muted)' }}>End Marker Window (s):</span>
                    <input 
                      type="number"
                      value={settings.analysisEndMarkerWindow}
                      onChange={e => updateSetting('analysisEndMarkerWindow', parseInt(e.target.value) || 0)}
                      style={{ width: 60, padding: '2px 6px', borderRadius: 4, border: '1px solid var(--border-color)', background: 'var(--bg-main)', color: 'var(--text-main)', fontSize: 13 }}
                    />
                  </div>

                  {settings.selectedSignal && settings.selectedSignal.includes('Finger Pressure') && (
                    <div style={{ width: '1px', height: '16px', background: 'var(--border-color)' }}></div>
                  )}

                  {settings.selectedSignal && settings.selectedSignal.includes('Finger Pressure') && (
                    <label style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', cursor: 'pointer', gap: 6 }}>
                      <input 
                        type="checkbox" 
                        checked={settings.compareGaussian} 
                        onChange={e => updateSetting('compareGaussian', e.target.checked)} 
                        style={{ cursor: 'pointer' }}
                      />
                      Compare Gaussian (1000) vs Resampled
                    </label>
                  )}
                </div>
              )}

              {plotData.length > 0 ? (
                <div style={{ width: '100%', display: 'flex', flexDirection: 'column' }}>
                  {settings.analysisView === 'Supine to Standing Analysis' && analysisStats && analysisStats.length > 0 && (
                    <div style={{ position: 'relative', height: '130px', marginLeft: '45px', marginRight: '25px', overflow: 'hidden', marginTop: '10px' }}>
                      {analysisStats.map((stat, i) => {
                        if (!stat.t_start_ms) return null;
                        const leftPct = ((stat.t_start_ms - plotXRange.min) / (plotXRange.max - plotXRange.min)) * 100;
                        if (leftPct < -100 || leftPct > 200) return null; // out of view buffer
                        
                        const _fmt = (v) => v !== null && v !== undefined ? v.toFixed(2) : "—";
                        return (
                          <div id={`stat-card-${stat.id || i}`} key={stat.id || i} style={{
                            position: 'absolute',
                            left: `${leftPct}%`,
                            top: 0,
                            transform: 'translateX(-50%)',
                            pointerEvents: 'auto',
                            width: '490px',
                            background: 'rgba(15, 15, 20, 0.92)',
                            color: 'white',
                            padding: '10px',
                            borderRadius: '8px',
                            border: '1px solid rgba(255,255,255,0.15)',
                            fontSize: '11px',
                            fontFamily: 'var(--font-sans)',
                            lineHeight: '1.4',
                            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '6px'
                          }}>
                            {/* Header Row */}
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 600, borderBottom: '1px solid rgba(255,255,255,0.15)', paddingBottom: '4px' }}>
                              <span>Base: <span style={{ color: '#60a5fa' }}>{_fmt(stat.baseline)}</span></span>
                              <span>Time: <span style={{ color: '#facc15' }}>{stat.transition_time !== null ? `${stat.transition_time.toFixed(1)}s` : "—"}</span></span>
                              <span>End: <span style={{ color: '#ef4444' }}>{_fmt(stat.end_val)}</span></span>
                            </div>
                            
                            {/* Columns */}
                            <div style={{ display: 'flex', gap: '12px' }}>
                              {/* Orange Column */}
                              <div style={{ flex: 1 }}>
                                <div style={{ color: '#f97316', fontWeight: 600, marginBottom: '2px' }}>🟠 Trans to End</div>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#9ca3af' }}>Duration:</span> <span>{stat.or_duration !== null ? `${stat.or_duration.toFixed(1)}s` : "—"}</span></div>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#9ca3af' }}>Drop:</span> <span>{_fmt(stat.or_pct_drop)}%</span></div>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#9ca3af' }}>Min:</span> <span>{_fmt(stat.or_min_val)}</span></div>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#9ca3af' }}>Area:</span> <span>{_fmt(stat.or_area_above)}/{_fmt(stat.or_area_below)}</span></div>
                              </div>
                              {/* Divider */}
                              <div style={{ width: '1px', background: 'rgba(255,255,255,0.1)' }}></div>
                              {/* Green Column */}
                              <div style={{ flex: 1 }}>
                                <div style={{ color: '#22c55e', fontWeight: 600, marginBottom: '2px' }}>🟢 Stand to End</div>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#9ca3af' }}>Duration:</span> <span>{stat.gr_duration !== null ? `${stat.gr_duration.toFixed(1)}s` : "—"}</span></div>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#9ca3af' }}>Drop:</span> <span>{_fmt(stat.gr_pct_drop)}%</span></div>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#9ca3af' }}>Min:</span> <span>{_fmt(stat.gr_min_val)}</span></div>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#9ca3af' }}>Area:</span> <span>{_fmt(stat.gr_area_above)}/{_fmt(stat.gr_area_below)}</span></div>
                              </div>
                              {/* Divider */}
                              <div style={{ width: '1px', background: 'rgba(255,255,255,0.1)' }}></div>
                              {/* Blue Column */}
                              <div style={{ flex: 1 }}>
                                <div style={{ color: '#3b82f6', fontWeight: 600, marginBottom: '2px' }}>🔵 Baseline Recovery</div>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#9ca3af' }}>Drop:</span> <span>{stat.rec_pct_drop !== null ? `${stat.rec_pct_drop.toFixed(2)}%` : "—"}</span></div>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#9ca3af' }}>Min:</span> <span>{_fmt(stat.rec_min_val)}</span></div>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#9ca3af' }}>Duration:</span> <span>{stat.rec_duration !== null ? `${stat.rec_duration.toFixed(1)}s` : "—"}</span></div>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#9ca3af' }}>Started In:</span> <span>{stat.rec_started_in || "—"}</span></div>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#9ca3af' }}>Area:</span> <span>{_fmt(stat.rec_area)}</span></div>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                  <Plot
                    data={plotData}
                    layout={{
                      plot_bgcolor: 'transparent',
                      paper_bgcolor: 'transparent',
                      font: { color: 'var(--text-main)', family: 'var(--font-sans)' },
                      margin: { l: 45, r: 25, t: 30, b: 40 },
                      shapes: layoutShapes,
                      annotations: layoutAnnotations,
                      xaxis: { title: 'Time', gridcolor: 'rgba(0,0,0,0.05)', type: 'date' },
                      yaxis: { title: 'Amplitude', gridcolor: 'rgba(0,0,0,0.05)' },
                      showlegend: true,
                      legend: { orientation: 'h', y: 1.15, x: 0.5, xanchor: 'center' },
                      uirevision: 'true',
                    }}
                    config={{ responsive: true, scrollZoom: false, displayModeBar: true }}
                    onRelayout={handleRelayout}
                    useResizeHandler={true}
                    style={{ width: '100%', height: '620px' }}
                  />
                </div>
              ) : null}
            </div>

            {/* Filter Sandboxes */}
            {isFilteringPreview && availableSignals.length > 0 && (
              <div className="glass-panel flex-row" style={{ alignItems: 'flex-start' }}>
                <div style={{ flex: 1, paddingRight: 24, borderRight: '1px solid var(--border-color)' }}>
                  <FilterSandbox prefix="fp" label="Finger Pressure Filter" settings={settings} setSettings={setSettings} />
                </div>
                <div style={{ flex: 1, paddingLeft: 24 }}>
                  <FilterSandbox prefix="cbf" label="CBF Filter" settings={settings} setSettings={setSettings} />
                </div>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}

export default App;
