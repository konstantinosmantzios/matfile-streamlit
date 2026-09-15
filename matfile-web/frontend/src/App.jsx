import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import axios from 'axios';
import { Activity, UploadCloud, Download, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, Target, X, Info } from 'lucide-react';
import Plot from 'react-plotly.js';
import FilterSandbox from './components/FilterSandbox';

// Memoized wrapper: skips Plotly's (heavy) re-render when props are unchanged,
// e.g. when unrelated UI state (loading flags, memory badge) changes.
const MemoPlot = React.memo(Plot);

const PLOT_CONFIG = { responsive: true, scrollZoom: false, displayModeBar: true };

const API_BASE_URL = 'http://localhost:8000/api';


const VIEWPORT_DEBOUNCE_MS = 300;
const SETTINGS_DEBOUNCE_MS = 600;
const TARGET_POINTS = 3000;

const getStoredDefaults = () => {
  try {
    const stored = localStorage.getItem('matAnalyzerDefaults');
    if (stored) return JSON.parse(stored);
  } catch (e) {
    console.error("Failed to load defaults", e);
  }
  return {};
};

function App() {
  const storedDefaults = getStoredDefaults();
  const [sessionId, setSessionId] = useState(null);
  const [fileName, setFileName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isViewportLoading, setIsViewportLoading] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [isConverting, setIsConverting] = useState(false);
  const [error, setError] = useState('');
  const [plotData, setPlotData] = useState([]);
  const [plotXRange, setPlotXRange] = useState(null);
  const [plotYRange, setPlotYRange] = useState(null);
  const [plotRevision, setPlotRevision] = useState(1);
  const [layoutShapes, setLayoutShapes] = useState([]);
  const [layoutAnnotations, setLayoutAnnotations] = useState([]);
  const [availableSignals, setAvailableSignals] = useState([]);
  const [tests, setTests] = useState([]);
  const [memoryUsage, setMemoryUsage] = useState(null);
  const [memoryInfo, setMemoryInfo] = useState(null);
  const [analysisStats, setAnalysisStats] = useState([]);
  
  const [analysisView, setAnalysisView] = useState("Filtering Preview");
  const [compareGaussian, setCompareGaussian] = useState(storedDefaults.compareGaussian ?? false);
  const [, forceRender] = useState({});
  
  // Force Plotly to resize when the layout changes (e.g., right panel appears/disappears)
  useEffect(() => {
    const timer = setTimeout(() => {
      window.dispatchEvent(new Event('resize'));
    }, 50);
    return () => clearTimeout(timer);
  }, [analysisView]);


  const [localAnalysisSettings, setLocalAnalysisSettings] = useState({
    analysisBaselineWindow: storedDefaults.analysisBaselineWindow ?? 30,
    analysisEndMarkerWindow: storedDefaults.analysisEndMarkerWindow ?? 10,
    useBaselineArea: storedDefaults.useBaselineArea ?? false,
    baselineEndComment: storedDefaults.baselineEndComment ?? 'Transition'
  });

  const [endMarkerOverrides, setEndMarkerOverrides] = useState({});
  const [editingTestIdx, setEditingTestIdx] = useState(null);
  const [draftEndMarker, setDraftEndMarker] = useState(null);

  // Refs
  const backendDataRef = useRef(null);
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
  const skipNextSettingsEffectRef = useRef(false);
  const plotContainerRef = useRef(null);
  const targetTestIdxRef = useRef(-1); // tracks last explicitly navigated-to test
  const jumpingToTestRef = useRef(-1); // used to provide instant UI feedback before Plotly blocks the thread

  useEffect(() => {
    analysisStatsRef.current = analysisStats;
  }, [analysisStats]);

  // Settings
  const [settings, setSettings] = useState({
    resampleMode: storedDefaults.resampleMode ?? "Beat-based",
    resampleRateTime: storedDefaults.resampleRateTime ?? 1,
    resampleRateBeat: storedDefaults.resampleRateBeat ?? 5,
    autoCalOption: storedDefaults.autoCalOption ?? "Auto-Detect",
    fpFilter: storedDefaults.fpFilter ?? "None",
    fpSavgolWin: storedDefaults.fpSavgolWin ?? 51,
    fpSavgolPoly: storedDefaults.fpSavgolPoly ?? 5,
    fpButterCutoff: storedDefaults.fpButterCutoff ?? 5.0,
    fpButterOrder: storedDefaults.fpButterOrder ?? 4,
    fpHampelWin: storedDefaults.fpHampelWin ?? 5,
    fpHampelSig: storedDefaults.fpHampelSig ?? 3.0,
    cbfFilter: storedDefaults.cbfFilter ?? "None",
    cbfSavgolWin: storedDefaults.cbfSavgolWin ?? 51,
    cbfSavgolPoly: storedDefaults.cbfSavgolPoly ?? 5,
    cbfButterCutoff: storedDefaults.cbfButterCutoff ?? 5.0,
    cbfButterOrder: storedDefaults.cbfButterOrder ?? 4,
    cbfHampelWin: storedDefaults.cbfHampelWin ?? 5,
    cbfHampelSig: storedDefaults.cbfHampelSig ?? 3.0,
    analysisBaselineWindow: storedDefaults.analysisBaselineWindow ?? 30,
    baselineEndComment: storedDefaults.baselineEndComment ?? "Transition",
    analysisEndMarkerWindow: storedDefaults.analysisEndMarkerWindow ?? 10,
    useMapGaussianForStats: storedDefaults.useMapGaussianForStats ?? false
  });

  useEffect(() => { latestSessionIdRef.current = sessionId; }, [sessionId]);

  // Cleanup session files on browser window/tab close or refresh
  useEffect(() => {
    const handleBeforeUnload = () => {
      if (sessionId) {
        const url = `${API_BASE_URL}/cleanup?session_id=${sessionId}`;
        if (navigator.sendBeacon) {
          navigator.sendBeacon(url);
        } else {
          fetch(url, { method: 'POST', keepalive: true }).catch(() => {});
        }
      }
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [sessionId]);

  const handleApplyAnalysisSettings = () => {
    setSettings(prev => ({
      ...prev,
      analysisBaselineWindow: localAnalysisSettings.analysisBaselineWindow,
      analysisEndMarkerWindow: localAnalysisSettings.analysisEndMarkerWindow,
      baselineEndComment: localAnalysisSettings.baselineEndComment
    }));
  };

  const handleApplyFilteringSettings = () => {
    setSettings(prev => ({
      ...prev,
      ...localFilteringSettings
    }));
  };

  const handleSaveDefaults = () => {
    const defaultsToSave = {
      ...settings,
      ...localAnalysisSettings,
      ...localFilteringSettings,
      compareGaussian
    };
    delete defaultsToSave.selectedSignal;
    
    localStorage.setItem('matAnalyzerDefaults', JSON.stringify(defaultsToSave));
    
    // Optional: could show a small toast, but a simple alert is fine for now
    alert('Current settings (filters, baseline, and windows) saved as defaults!');
  };

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
    // Removed setInterval to prevent terminal spam
  }, []);

  // ---- Auto-apply settings on change (debounced) ----------------------
  useEffect(() => {
    if (isFirstRender.current) { isFirstRender.current = false; return; }
    if (isUploadingRef.current) return;
    if (!latestSessionIdRef.current) return;
    if (skipNextSettingsEffectRef.current) {
      skipNextSettingsEffectRef.current = false;
      return;
    }

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
    
    const hasYAuto = eventData['yaxis.autorange'] === true;
    const hasXAuto = eventData['xaxis.autorange'] === true;
    const hasXRangeArray = Array.isArray(eventData['xaxis.range']);
    
    // A double-click on the plot area typically sends yaxis.autorange: true AND xaxis.range: [min, max]
    // A direct double-click on the X-axis sends xaxis.autorange: true
    if (hasXAuto || (hasYAuto && hasXRangeArray)) {
      currentZoomRef.current = null;
      targetTestIdxRef.current = -1;
      
      const indicator = document.getElementById('test-indicator');
      if (indicator && analysisStatsRef.current) {
        indicator.innerText = `- / ${analysisStatsRef.current.length}`;
      }

      if (viewportTimeoutRef.current) clearTimeout(viewportTimeoutRef.current);
      
      requestAnimationFrame(() => {
        setTimeout(() => {
          // Clone the objects so Plotly treats them as new traces and forces a redraw
          setPlotData(prev => initialPlotDataRef.current ? initialPlotDataRef.current.map(t => ({ ...t })) : prev);
          setPlotXRange(null);
          setPlotYRange(null);
          setPlotRevision(prev => prev + 1); // trigger Plotly update
        }, 0);
      });
      return;
    }
    
    // If only Y was autoranged (e.g., double click on Y axis specifically)
    if (hasYAuto && !hasXRangeArray && !hasXAuto) {
      setPlotYRange(null);
      setPlotRevision(prev => prev + 1);
      return;
    }

    let yMin = eventData['yaxis.range[0]'];
    let yMax = eventData['yaxis.range[1]'];
    if ((yMin === undefined || yMax === undefined) && Array.isArray(eventData['yaxis.range'])) {
      yMin = eventData['yaxis.range'][0];
      yMax = eventData['yaxis.range'][1];
    }
    if (yMin !== undefined && yMax !== undefined) {
      setPlotYRange({ min: yMin, max: yMax });
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

    if (currentZoomRef.current && 
        Math.abs(currentZoomRef.current.xMinMs - xMinMs) < 10 && 
        Math.abs(currentZoomRef.current.xMaxMs - xMaxMs) < 10) {
        return; // ignore programmatic echo
    }

    currentZoomRef.current = { xMinMs, xMaxMs };
    targetTestIdxRef.current = -1; // user manually panned, unlock target test
    setPlotXRange({ min: xMinMs, max: xMaxMs });
    
    if (analysisView === 'Supine to Standing Analysis') {
      return;
    }

    if (viewportTimeoutRef.current) clearTimeout(viewportTimeoutRef.current);
    viewportTimeoutRef.current = setTimeout(() => {
      fetchViewport(xMinMs, xMaxMs);
    }, VIEWPORT_DEBOUNCE_MS);
  }, [sessionId, analysisView]);

  const handlePlotClick = useCallback((e) => {
    if (editingTestIdx !== null && e.points && e.points.length > 0) {
      const x = e.points[0].x;
      const xMs = typeof x === 'number' ? x : new Date(x).getTime();
      setDraftEndMarker(xMs);
    }
  }, [editingTestIdx]);

  // Stable layout object — only recreated when a Plotly-relevant value actually
  // changes, so MemoPlot can skip redundant full re-renders.
  const plotLayout = useMemo(() => ({
    autosize: true,
    plot_bgcolor: 'transparent',
    paper_bgcolor: 'transparent',
    font: { color: 'var(--text-main)', family: 'var(--font-sans)' },
    margin: { l: 45, r: 25, t: 30, b: 40 },
    shapes: draftEndMarker ? [...layoutShapes, {
      type: 'line',
      x0: draftEndMarker, x1: draftEndMarker,
      y0: 0, y1: 1,
      xref: 'x', yref: 'paper',
      line: { color: '#ea580c', width: 2, dash: 'dot' }
    }] : layoutShapes,
    annotations: layoutAnnotations,
    xaxis: {
      title: 'Time',
      gridcolor: 'rgba(0,0,0,0.05)',
      type: 'date',
      ...(plotXRange && plotXRange.min && plotXRange.max ? { range: [plotXRange.min, plotXRange.max] } : { autorange: true })
    },
    yaxis: {
      title: 'Amplitude',
      gridcolor: 'rgba(0,0,0,0.05)',
      ...(plotYRange ? { range: [plotYRange.min, plotYRange.max] } : { autorange: true })
    },
    showlegend: true,
    legend: { orientation: 'h', y: 1.15, x: 0.5, xanchor: 'center' },
    uirevision: plotRevision,
  }), [layoutShapes, layoutAnnotations, plotXRange, plotYRange, plotRevision, draftEndMarker]);

  const plotStyle = useMemo(() => ({
    width: '100%',
    height: '620px',
    cursor: editingTestIdx !== null ? 'crosshair' : 'default'
  }), [editingTestIdx]);

  const jumpToTest = (targetIdx, buttonName = "Unknown") => {
    if (!analysisStats || analysisStats.length === 0) return;
    
    if (targetIdx < 0) targetIdx = 0;
    if (targetIdx >= analysisStats.length) targetIdx = analysisStats.length - 1;

    const targetStat = analysisStats[targetIdx];
    const baseWindowMs = (localAnalysisSettings.analysisBaselineWindow || 30) * 1000;
    const transMs = targetStat.t_trans_ms || targetStat.t_start_ms || 0;
    const zoomStart = transMs - baseWindowMs - 5000;
    const zoomEnd = (targetStat.t_end_ms || (targetStat.t_stand_ms + 30000)) + 10000;

    // Calculate Y auto-range for this specific X window
    let yMin = Infinity;
    let yMax = -Infinity;
    plotData.forEach(trace => {
      // Only consider visible signal/data traces, not invisible or purely marker traces
      if (trace.visible === false || trace.visible === 'legendonly') return;
      if (!trace.x || !trace.y) return;
      if (trace.name && (trace.name === 'Rec Start' || trace.name === 'Rec End' || trace.name === 'Start' || trace.name === 'End Marker')) return; // ignore visual markers for scaling
      
      for (let i = 0; i < trace.x.length; i++) {
        const xMs = typeof trace.x[i] === 'number' ? trace.x[i] : new Date(trace.x[i]).getTime();
        if (xMs >= zoomStart && xMs <= zoomEnd) {
          const y = trace.y[i];
          if (y !== null && !isNaN(y)) {
            if (y < yMin) yMin = y;
            if (y > yMax) yMax = y;
          }
        }
      }
    });
    // Update ref immediately so the text component can compute the new state
    targetTestIdxRef.current = targetIdx;
    jumpingToTestRef.current = targetIdx;
    
    // DIRECT DOM MANIPULATION for instant feedback before thread freezes
    const indicator = document.getElementById('test-indicator');
    if (indicator) {
      indicator.innerText = `${targetIdx + 1} / ${analysisStats.length}`;
    }
    
    // Yield to the browser's layout and paint cycle so the text actually appears on screen 
    // before Plotly freezes the thread with heavy rendering.
    requestAnimationFrame(() => {
      setTimeout(() => {
        setPlotXRange({ min: zoomStart, max: zoomEnd });
        currentZoomRef.current = { xMinMs: zoomStart, xMaxMs: zoomEnd };
        
        if (yMin !== Infinity && yMax !== -Infinity) {
          const margin = (yMax - yMin) * 0.1;
          setPlotYRange({ min: yMin - margin, max: yMax + margin });
        } else {
          setPlotYRange(null);
        }
        
        jumpingToTestRef.current = -1; // clear jumping flag
        setPlotRevision(prev => prev + 1); // trigger Plotly update
      }, 0);
    });
  };

  const fetchViewport = async (xMinMs, xMaxMs, sid = sessionId) => {
    if (!sid) return;
    setIsViewportLoading(true);
    const __t0 = performance.now();
    try {
      const response = await axios.post(`${API_BASE_URL}/viewport`, {
        session_id: sid,
        x_min: xMinMs,
        x_max: xMaxMs,
        target_points: TARGET_POINTS,
      });
      console.log(`[viewport] request ${(performance.now() - __t0).toFixed(0)} ms  traces=${response.data.traces.map(t => t.trace_id).join(',')}  pts=${response.data.visible_points}`);
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
    
    // Clean up previous session from server disk if uploading a new file
    if (sessionId) {
      axios.post(`${API_BASE_URL}/cleanup?session_id=${sessionId}`).catch(() => {});
    }

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
        skipNextSettingsEffectRef.current = true;
        setSettings(updatedSettings);
      }
      await processData(response.data.session_id, updatedSettings);
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed');
    } finally {
      setIsLoading(false);
      isUploadingRef.current = false;
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // ---- Process data ---------------------------------------------------
  const updatePlotFromBackendData = useCallback(() => {
    if (!backendDataRef.current) return;
    const data = backendDataRef.current;
    
    if (analysisView === 'Filtering Preview') {
      const traces = (data.filtering_traces || []).filter(t => t.trace_id !== "gauss1000" || compareGaussian);
      setPlotData(traces);
      initialPlotDataRef.current = traces;
      setLayoutShapes(data.filtering_shapes || []);
      setLayoutAnnotations(data.filtering_annotations || []);
    } else {
      const traces = (data.analysis_traces || []).filter(t => t.trace_id !== "gauss1000" || compareGaussian);
      setPlotData(traces);
      initialPlotDataRef.current = traces;
      setLayoutShapes(data.analysis_shapes || []);
      setLayoutAnnotations(data.analysis_annotations || []);
    }
    
    // Force plot update so layout.xaxis.range is strictly respected
    setPlotRevision(prev => prev + 1);
    setAnalysisStats(data.analysis_stats || []);
  }, [analysisView, compareGaussian]);

  useEffect(() => {
    updatePlotFromBackendData();
  }, [updatePlotFromBackendData]);

  const processData = async (sid = sessionId, currentSettings = settings, overrides = endMarkerOverrides) => {
    if (!sid) return;
    setIsLoading(true);
    setError('');
    const __t0 = performance.now();
    try {
      const payloadOverrides = Object.keys(overrides).length > 0 ? overrides : undefined;
      const response = await axios.post(`${API_BASE_URL}/process`, { session_id: sid, settings: currentSettings, end_marker_overrides: payloadOverrides });
      console.log(`[process] ${currentSettings.selectedSignal}  round-trip ${(performance.now() - __t0).toFixed(0)} ms  traces=${(response.data.filtering_traces || []).map(t => t.trace_id).join(',')}`);
      backendDataRef.current = response.data;
      
      const traces = analysisView === 'Filtering Preview' ? (response.data.filtering_traces || []) : (response.data.analysis_traces || []);
      const filteredTraces = traces.filter(t => t.trace_id !== "gauss1000" || compareGaussian);
      
      updatePlotFromBackendData();
      
      let initialMin = null, initialMax = null;
      filteredTraces.forEach(t => {
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
        } else if (targetTestIdxRef.current >= 0 && response.data.analysis_stats) {
          // If locked on a test, re-calculate the zoom bounds for that test
          const tStat = response.data.analysis_stats[targetTestIdxRef.current];
          if (tStat) {
             const bWin = (currentSettings.analysisBaselineWindow || 30) * 1000;
             const tMs = tStat.t_trans_ms || tStat.t_start_ms || 0;
             const zStart = tMs - bWin - 5000;
             const zEnd = (tStat.t_end_ms || (tStat.t_stand_ms + 30000)) + 10000;
             currentZoomRef.current = { xMinMs: zStart, xMaxMs: zEnd };
             setPlotXRange({ min: zStart, max: zEnd });
          }
        } else {
        }
      }
      
      if (response.data.available_signals) {
        setAvailableSignals(response.data.available_signals);
        if (!currentSettings.selectedSignal && response.data.available_signals.length > 0) {
          const defaultSig = response.data.available_signals.find(s => s.includes('Finger Pressure')) || response.data.available_signals[0];
          skipNextSettingsEffectRef.current = true;
          setSettings(prev => ({ ...prev, selectedSignal: defaultSig }));
        }
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Processing failed');
    } finally {
      setIsLoading(false);
      if (currentZoomRef.current && analysisView === 'Filtering Preview') {
        setTimeout(() => { fetchViewport(currentZoomRef.current.xMinMs, currentZoomRef.current.xMaxMs, sid); }, 50);
      }
    }
  };

  // ---- Export & Clear --------------------------------------------------
  const handleSignalSelect = (sig) => {
    if (sig === settings.selectedSignal || !latestSessionIdRef.current) return;
    // Bypass the 600ms settings debounce: request the switch immediately.
    // The backend serves it from the per-signal trace cache (no reprocessing).
    skipNextSettingsEffectRef.current = true;
    setSettings(prev => ({ ...prev, selectedSignal: sig }));
    processData(latestSessionIdRef.current, { ...settings, selectedSignal: sig });
  };

  const exportData = async () => {
    if (!sessionId) return;
    setIsExporting(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/export`, { 
        session_id: sessionId, 
        settings,
        end_marker_overrides: endMarkerOverrides
      }, { responseType: 'blob' });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      const pad = (n) => String(n).padStart(2, '0');
      const now = new Date();
      const dateStr = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}_${pad(now.getHours())}-${pad(now.getMinutes())}-${pad(now.getSeconds())}`;
      const cleanName = fileName.replace(/\.(mat|parquet|csv)$/i, '');
      link.setAttribute('download', `Analysis_${cleanName}_${dateStr}.xlsx`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      setError('Export failed');
    } finally {
      setIsExporting(false);
    }
  };



  const handleConvertMat = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setError('');
    setIsConverting(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE_URL}/convert`, formData, {
        responseType: 'blob',
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', file.name.replace('.mat', '.parquet'));
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      console.error("Convert Error:", err);
      setError(err.response?.data?.detail || err.message || 'Error converting file');
    } finally {
      setIsConverting(false);
      // clear input
      event.target.value = '';
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
  const isFilteringPreview = analysisView === 'Filtering Preview';

  const getPlottedHz = () => {
    if (!plotData || plotData.length === 0) return null;
    const rawTrace = plotData.find(t => t.trace_id === 'raw' || t.name === 'Raw Data');
    if (!rawTrace || !rawTrace.x || rawTrace.x.length < 2) return null;
    const pts = rawTrace.x.length;
    const durationS = (rawTrace.x[rawTrace.x.length - 1] - rawTrace.x[0]) / 1000;
    if (durationS <= 0) return null;
    return Math.min(200, pts / durationS).toFixed(1);
  };
  const isJumping = jumpingToTestRef.current !== -1;
  let currentTestIdx = targetTestIdxRef.current >= 0 ? targetTestIdxRef.current : -1;
  
  if (!isJumping && analysisStats && analysisStats.length > 0 && plotXRange && plotXRange.min && plotXRange.max) {
    const center = (plotXRange.min + plotXRange.max) / 2;
    // Validate that the ref index still matches the viewport (it might have been panned away)
    if (currentTestIdx >= 0 && currentTestIdx < analysisStats.length) {
      const refStat = analysisStats[currentTestIdx];
      const refTransMs = refStat.t_trans_ms || refStat.t_start_ms || 0;
      const refBaseWindowMs = (localAnalysisSettings.analysisBaselineWindow || 30) * 1000;
      const refZoomStart = refTransMs - refBaseWindowMs - 5000;
      const refZoomEnd = (refStat.t_end_ms || (refStat.t_stand_ms + 30000)) + 10000;
      
      const realStart = Math.min(refZoomStart, refZoomEnd);
      const realEnd = Math.max(refZoomStart, refZoomEnd);
      
      // If viewport center has drifted completely outside this test's viewing window, fall back
      if (center < realStart || center > realEnd) {
        currentTestIdx = -1; // will recompute below
        targetTestIdxRef.current = -1;
      }
    } else {
      currentTestIdx = -1;
    }

    if (currentTestIdx === -1) {
      let foundIdx = -1;
      let minCenterDist = Infinity;
      
      // First, try to find a test whose expected viewing window contains the current viewport center
      for (let idx = 0; idx < analysisStats.length; idx++) {
        const stat = analysisStats[idx];
        const transMs = stat.t_trans_ms || stat.t_start_ms || 0;
        const baseWindowMs = (localAnalysisSettings.analysisBaselineWindow || 30) * 1000;
        const zoomStart = transMs - baseWindowMs - 5000;
        const zoomEnd = (stat.t_end_ms || (stat.t_stand_ms + 30000)) + 10000;
        
        const realStart = Math.min(zoomStart, zoomEnd);
        const realEnd = Math.max(zoomStart, zoomEnd);
        
        if (center >= realStart && center <= realEnd) {
           const expectedCenter = (realStart + realEnd) / 2;
           const dist = Math.abs(expectedCenter - center);
           if (dist < minCenterDist) {
             minCenterDist = dist;
             foundIdx = idx;
           }
        }
      }
      
      // If no test contains the center, just find the closest test globally
      if (foundIdx === -1) {
        let bestDist = Infinity;
        analysisStats.forEach((stat, idx) => {
          const transMs = stat.t_trans_ms || stat.t_start_ms || 0;
          const baseWindowMs = (localAnalysisSettings.analysisBaselineWindow || 30) * 1000;
          const zoomStart = transMs - baseWindowMs - 5000;
          const zoomEnd = (stat.t_end_ms || (stat.t_stand_ms + 30000)) + 10000;
          const realStart = Math.min(zoomStart, zoomEnd);
          const realEnd = Math.max(zoomStart, zoomEnd);
          const expectedCenter = (realStart + realEnd) / 2;
          const dist = Math.abs(expectedCenter - center);
          if (dist < bestDist) {
            bestDist = dist;
            foundIdx = idx;
          }
        });
      }
      
      currentTestIdx = foundIdx;
      targetTestIdxRef.current = currentTestIdx;
    }
  }

  const getVisibleStats = () => {
    if (analysisView !== 'Supine to Standing Analysis' || currentTestIdx === -1) {
      return [];
    }
    
    const viewDuration = plotXRange.max - plotXRange.min;
    const targetStat = analysisStats[currentTestIdx];
    const tStart = targetStat.t_trans_ms || targetStat.t_start_ms || 0;
    const tEnd = targetStat.t_end_ms || (targetStat.t_stand_ms + 30000) || 0;
    
    const isVisible = (tStart >= plotXRange.min && tStart <= plotXRange.max) || 
                      (tEnd >= plotXRange.min && tEnd <= plotXRange.max) ||
                      (tStart <= plotXRange.min && tEnd >= plotXRange.max);
                      
    const baseWindowMs = (localAnalysisSettings.analysisBaselineWindow || 30) * 1000;
    const expectedZoomStart = tStart - baseWindowMs - 5000;
    const expectedZoomEnd = tEnd + 10000;
    const realStart = Math.min(expectedZoomStart, expectedZoomEnd);
    const realEnd = Math.max(expectedZoomStart, expectedZoomEnd);
    const isExactlyCentered = Math.abs(plotXRange.min - realStart) < 50 && Math.abs(plotXRange.max - realEnd) < 50;
    if ((viewDuration <= 150000 || isExactlyCentered) && isVisible) {
      return [targetStat];
    }
    
    return [];
  };
  const visibleStats = getVisibleStats();
  
  // If we are currently jumping, force the UI to consider the target test focused instantly
  const noTestFocused = jumpingToTestRef.current !== -1 ? false : visibleStats.length !== 1;
  const displayedText = noTestFocused ? `- / ${analysisStats ? analysisStats.length : 0}` : `${currentTestIdx + 1} / ${analysisStats ? analysisStats.length : 0}`;
  
  if (analysisStats && analysisStats.length > 0) {
  }
  
  let isCentered = false;
  if (currentTestIdx >= 0 && currentTestIdx < analysisStats.length && plotXRange && plotXRange.min != null) {
    const targetStat = analysisStats[currentTestIdx];
    const baseWindowMs = (localAnalysisSettings.analysisBaselineWindow || 30) * 1000;
    const transMs = targetStat.t_trans_ms || targetStat.t_start_ms || 0;
    const expectedZoomStart = transMs - baseWindowMs - 5000;
    const expectedZoomEnd = (targetStat.t_end_ms || (targetStat.t_stand_ms + 30000)) + 10000;
    
    // 50ms tolerance for panning floating point changes
    if (Math.abs(plotXRange.min - expectedZoomStart) < 50 && Math.abs(plotXRange.max - expectedZoomEnd) < 50) {
      isCentered = true;
    }
  }

  const disableRecenter = isCentered || noTestFocused;

  return (
    <div className="app-container">
      <main className="main-content">
        
        {/* Top Header */}
        <div className="glass-panel top-header" style={{ padding: 0, display: 'flex', flexDirection: 'column' }}>
          <div className="flex-between" style={{ padding: '16px 24px' }}>
            <div className="flex-row">
              <Activity size={24} color="var(--accent-blue)" />
              <h1 style={{ margin: 0 }}>FAMELab Mat File Viewer/Analyzer</h1>
              
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
              <div className="flex-row" style={{ marginRight: 16, gap: 16 }}>
                <button
                  className="btn btn-secondary"
                  onClick={handleSaveDefaults}
                  style={{ padding: '4px 10px', fontSize: 12, borderRadius: 6, display: 'flex', alignItems: 'center', border: '1px solid var(--border-color)', background: '#fff', color: 'var(--text-main)', cursor: 'pointer' }}
                  title="Save current filters, baseline, and end windows as default for future sessions."
                >
                  Set Defaults
                </button>
              </div>
              {fileName && (
                <span className="badge" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  Active File: {fileName}
                  <X 
                    size={14} 
                    style={{ cursor: isLoading ? 'not-allowed' : 'pointer', opacity: isLoading ? 0.5 : 1, transition: 'opacity 0.2s', marginLeft: 4 }} 
                    onClick={() => { if (!isLoading) clearSession(); }} 
                    onMouseEnter={e => { if (!isLoading) e.currentTarget.style.opacity = 0.7; }}
                    onMouseLeave={e => { if (!isLoading) e.currentTarget.style.opacity = 1; }}
                  />
                </span>
              )}
            </div>
          </div>
          <div className={`expand-row ${editingTestIdx !== null ? 'expanded' : 'collapsed'}`}>
            <span>Click on the chart to set the new end marker for this test</span>
            <div style={{ display: 'flex', gap: 8 }}>
              {draftEndMarker && (
                <button 
                  onClick={() => {
                    const newOverrides = { ...endMarkerOverrides, [editingTestIdx]: draftEndMarker };
                    setEndMarkerOverrides(newOverrides);
                    setEditingTestIdx(null);
                    setDraftEndMarker(null);
                    processData(sessionId, settings, newOverrides);
                  }}
                  style={{ background: '#16a34a', border: 'none', color: 'white', padding: '4px 12px', borderRadius: '16px', cursor: 'pointer', fontWeight: 600 }}
                >
                  Apply
                </button>
              )}
              <button 
                onClick={() => {
                  setEditingTestIdx(null);
                  setDraftEndMarker(null);
                }}
                style={{ background: 'rgba(255,255,255,0.2)', border: 'none', color: 'white', padding: '4px 12px', borderRadius: '16px', cursor: 'pointer', fontWeight: 600 }}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>

        {error && (
          <div className="glass-panel" style={{ borderColor: 'var(--accent-red)', padding: '16px' }}>
            <p style={{ color: 'var(--accent-red)', margin: 0, fontWeight: 500 }}>{error}</p>
          </div>
        )}

        {!sessionId && (
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 1, gap: 20 }}>
            {/* Primary Upload Zone */}
            <input ref={fileInputRef} type="file" accept=".mat,.parquet" onChange={handleFileUpload} style={{ display: 'none' }} id="file-upload" />
            <label htmlFor="file-upload" className="upload-zone" style={{ width: '100%', maxWidth: 500 }}>
              <UploadCloud className="upload-icon" />
              <div>
                <h2>Upload LabChart File</h2>
                <p>Select a <b>.mat</b> or <b>.parquet</b> file</p>
              </div>
              {isLoading && (
                <div style={{ marginTop: 16, display: 'flex', alignItems: 'center', gap: 8, color: 'var(--accent-blue)', fontWeight: 600 }}>
                  <div className="spinner" style={{ width: 20, height: 20, borderWidth: 2, marginBottom: 0 }}></div>
                  Processing File...
                </div>
              )}
            </label>

            <div style={{ borderTop: '1px solid var(--border-color)', width: '100%', maxWidth: 400, margin: '10px 0' }}></div>

            {/* Conversion Zone */}
            <input type="file" accept=".mat" onChange={handleConvertMat} style={{ display: 'none' }} id="convert-upload" />
            <label htmlFor="convert-upload" className="btn btn-primary" style={{ cursor: 'pointer', display: 'flex', gap: 8, padding: '10px 20px', borderRadius: '12px' }}>
              <Download size={18} />
              {isConverting ? 'Converting...' : 'Convert .MAT to .Parquet locally'}
            </label>
            <p style={{ color: 'var(--text-muted)', fontSize: 13, marginTop: -10 }}>
              Use this tool first to convert massive .mat files into optimized .parquet files for lightning fast loading.
            </p>
          </div>
        )}

        {sessionId && (
          <>


            {/* Test Selector removed: navigation handled entirely by arrow buttons now */}

            {/* Plot Container */}
            <div className="glass-panel plot-container" style={{ padding: '24px', minWidth: 0, overflow: 'hidden' }}>
              
              {/* Overlay Loading State */}
              {(isLoading || isViewportLoading || isExporting) && (
                <div className="loading-overlay">
                  <div className="spinner"></div>
                  {isExporting ? (
                    <>
                      <h3 style={{ margin: 0, color: 'var(--accent-blue)' }}>Exporting Data</h3>
                      <p style={{ marginTop: 4 }}>Computing final statistics and generating Excel file...</p>
                    </>
                  ) : (
                    <>
                      <h3 style={{ margin: 0, color: 'var(--accent-blue)' }}>Processing Data</h3>
                      <p style={{ marginTop: 4 }}>Applying filters and recalculating...</p>
                    </>
                  )}
                </div>
              )}

              {/* Plot Container Header (Controls + Signals) */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 16, borderBottom: '1px solid var(--border-color)', paddingBottom: 16, marginBottom: 24 }}>
                
                {/* Left: View Toggle */}
                <div style={{ display: 'flex', background: 'rgba(0,0,0,0.05)', borderRadius: '12px', padding: '4px' }}>
                  <button
                    onClick={() => setAnalysisView('Filtering Preview')}
                    style={{
                      padding: '6px 16px', fontSize: '13px', borderRadius: '8px', border: 'none', cursor: 'pointer',
                      background: analysisView === 'Filtering Preview' ? '#fff' : 'transparent',
                      color: analysisView === 'Filtering Preview' ? 'var(--text-main)' : 'var(--text-muted)',
                      fontWeight: 600,
                      boxShadow: analysisView === 'Filtering Preview' ? '0 2px 4px rgba(0,0,0,0.05)' : 'none'
                    }}
                  >
                    Filtering
                  </button>
                  <button
                    onClick={() => setAnalysisView('Supine to Standing Analysis')}
                    style={{
                      padding: '6px 16px', fontSize: '13px', borderRadius: '8px', border: 'none', cursor: 'pointer',
                      background: analysisView === 'Supine to Standing Analysis' ? '#fff' : 'transparent',
                      color: analysisView === 'Supine to Standing Analysis' ? 'var(--text-main)' : 'var(--text-muted)',
                      fontWeight: 600,
                      boxShadow: analysisView === 'Supine to Standing Analysis' ? '0 2px 4px rgba(0,0,0,0.05)' : 'none'
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
                          onClick={() => handleSignalSelect(sig)}
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
              
              {/* Filtering Settings Row */}
              {analysisView === 'Filtering & Preprocessing' && settings.selectedSignal && settings.selectedSignal.includes('Finger Pressure') && (
                <div className="glass-panel flex-row" style={{ width: '100%', justifyContent: 'flex-end', padding: '12px 20px', marginTop: 16 }}>
                  <label style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-main)', display: 'flex', alignItems: 'center', cursor: 'pointer', gap: 8, background: compareGaussian ? 'rgba(59, 130, 246, 0.05)' : 'transparent', padding: '6px 12px', borderRadius: 8, transition: 'all 0.2s', border: compareGaussian ? '1px solid rgba(59, 130, 246, 0.2)' : '1px solid transparent' }}>
                    <input 
                      type="checkbox" 
                      style={{ accentColor: 'var(--accent-blue)', width: '16px', height: '16px', cursor: 'pointer' }}
                      checked={compareGaussian} 
                      onChange={e => setCompareGaussian(e.target.checked)} 
                    />
                    MAP - Gaussian (5sec)
                  </label>
                </div>
              )}

              {/* Analysis Settings Row */}
              {analysisView === 'Supine to Standing Analysis' && (
                <div className="glass-panel" style={{ width: '100%', padding: '12px 20px', marginTop: 16, display: 'flex', flexDirection: 'column', gap: 16 }}>
                  {/* Row 1: Baseline settings and Apply */}
                  <div className="flex-row" style={{ gap: 24, flexWrap: 'wrap', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 4 }} title="Duration of the baseline value (blue line) before the comment selected in 'Baseline Ends At'. Used for the Baseline-based method.">
                        Baseline Window
                        <Info size={14} style={{ opacity: 0.7 }} />
                      </span>
                      <div style={{ display: 'flex', alignItems: 'center', background: 'var(--bg-main)', border: '1px solid var(--border-color)', borderRadius: 6, padding: '2px 8px' }}>
                        <input 
                          type="number"
                          value={localAnalysisSettings.analysisBaselineWindow}
                          onChange={e => setLocalAnalysisSettings(prev => ({ ...prev, analysisBaselineWindow: parseInt(e.target.value) || 0 }))}
                          style={{ width: 40, border: 'none', background: 'transparent', color: 'var(--text-main)', fontSize: 13, outline: 'none', textAlign: 'center' }}
                        />
                        <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 4 }}>sec</span>
                      </div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 4 }} title="The reference comment point to calculate the baseline value before it. Used for the Baseline-based method.">
                        Baseline Ends At
                        <Info size={14} style={{ opacity: 0.7 }} />
                      </span>
                      <select 
                        value={localAnalysisSettings.baselineEndComment}
                        onChange={e => setLocalAnalysisSettings(prev => ({ ...prev, baselineEndComment: e.target.value }))}
                        style={{ padding: '4px 8px', borderRadius: 6, border: '1px solid var(--border-color)', background: 'var(--bg-main)', color: 'var(--text-main)', fontSize: 13, outline: 'none', cursor: 'pointer' }}
                      >
                        <option value="Transition">Transition</option>
                        <option value="Standing">Standing</option>
                      </select>
                    </div>

                    <button 
                      className="btn btn-primary" 
                      onClick={handleApplyAnalysisSettings}
                      style={{ padding: '6px 12px', fontSize: 13, borderRadius: 6 }}
                    >
                      Apply Settings
                    </button>
                  </div>

                  {/* Row 2: End Window and MAP toggles */}
                  <div className="flex-row" style={{ gap: 24, flexWrap: 'wrap', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 4 }} title="Duration of the analysis window after the comment (Transition comment for Method 1 and Standing for Method 2).">
                        End Window
                        <Info size={14} style={{ opacity: 0.7 }} />
                      </span>
                      <div style={{ display: 'flex', alignItems: 'center', background: 'var(--bg-main)', border: '1px solid var(--border-color)', borderRadius: 6, padding: '2px 8px' }}>
                        <input 
                          type="number"
                          value={localAnalysisSettings.analysisEndMarkerWindow}
                          onChange={e => setLocalAnalysisSettings(prev => ({ ...prev, analysisEndMarkerWindow: parseInt(e.target.value) || 0 }))}
                          style={{ width: 40, border: 'none', background: 'transparent', color: 'var(--text-main)', fontSize: 13, outline: 'none', textAlign: 'center' }}
                        />
                        <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 4 }}>sec</span>
                      </div>
                    </div>

                    {settings.selectedSignal && settings.selectedSignal.includes('Finger Pressure') && (
                      <div className="flex-row" style={{ gap: 16 }}>
                        <label style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-main)', display: 'flex', alignItems: 'center', cursor: 'pointer', gap: 8, background: settings.useMapGaussianForStats ? 'rgba(234, 88, 12, 0.05)' : 'transparent', padding: '6px 12px', borderRadius: 8, transition: 'all 0.2s', border: settings.useMapGaussianForStats ? '1px solid rgba(234, 88, 12, 0.2)' : '1px solid transparent' }} title="Use MAP - Gaussian (5sec) for calculating minimum/maximum and endpoints.">
                          <input 
                            type="checkbox" 
                            style={{ accentColor: '#ea580c', width: '16px', height: '16px', cursor: 'pointer' }}
                            checked={settings.useMapGaussianForStats} 
                            onChange={e => {
                              const isChecked = e.target.checked;
                              setSettings(prev => ({ ...prev, useMapGaussianForStats: isChecked }));
                              if (isChecked) {
                                setCompareGaussian(true);
                              }
                            }} 
                          />
                          Use MAP - Gaussian (5sec)
                        </label>
                        <label style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-main)', display: 'flex', alignItems: 'center', cursor: 'pointer', gap: 8, background: compareGaussian ? 'rgba(59, 130, 246, 0.05)' : 'transparent', padding: '6px 12px', borderRadius: 8, transition: 'all 0.2s', border: compareGaussian ? '1px solid rgba(59, 130, 246, 0.2)' : '1px solid transparent' }} title="Show MAP - Gaussian (5sec) on the chart.">
                          <input 
                            type="checkbox" 
                            style={{ accentColor: 'var(--accent-blue)', width: '16px', height: '16px', cursor: 'pointer' }}
                            checked={compareGaussian} 
                            onChange={e => setCompareGaussian(e.target.checked)} 
                          />
                          Show MAP - Gaussian (5sec)
                        </label>
                      </div>
                    )}
                  </div>
                </div>
              )}
              {plotData.length > 0 ? (
                <div style={{ width: '100%', display: 'flex', flexDirection: 'row', gap: 24 }}>
                  <div ref={plotContainerRef} style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
                    <MemoPlot
                      data={plotData}
                      layout={plotLayout}
                      config={PLOT_CONFIG}
                      onRelayout={handleRelayout}
                      onClick={handlePlotClick}
                      useResizeHandler={true}
                      style={plotStyle}
                    />
                  </div>
                  {/* Right: Stats Table or Placeholder */}
                  {analysisView === 'Supine to Standing Analysis' && (
                    <div style={{ width: '280px', flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 16, marginTop: 30 }}>
                      {visibleStats.length === 1 ? (() => {
                        const stat = visibleStats[0];
                        const _fmt = (v, digits=2) => (v !== null && v !== undefined && typeof v === 'number') ? v.toFixed(digits) : "—";
                        return (
                          <div style={{
                            background: 'rgba(255, 255, 255, 0.85)',
                            backdropFilter: 'blur(16px)',
                            WebkitBackdropFilter: 'blur(16px)',
                            color: 'var(--text-main)',
                            padding: '16px',
                            borderRadius: '16px',
                            border: '1px solid rgba(0,0,0,0.08)',
                            fontSize: '12px',
                            fontFamily: 'var(--font-sans)',
                            boxShadow: '0 8px 32px rgba(0,0,0,0.08)',
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '16px',
                          }}>
                            {/* Header Row */}
                            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', fontWeight: 600, borderBottom: '1px solid rgba(0,0,0,0.08)', paddingBottom: '12px', fontSize: '13px' }}>
                              <span>Transition Duration: <span style={{ color: 'var(--text-main)', fontWeight: 700 }}>{stat.transition_time != null ? `${_fmt(stat.transition_time, 1)}s` : "—"}</span></span>
                            </div>
                            
                            {/* Stacked Sections */}
                            
                            {/* Orange Section */}
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                              <div style={{ color: '#ea580c', fontWeight: 700, marginBottom: '4px', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                <div style={{width: 8, height: 8, borderRadius: '50%', background: '#ea580c'}}></div>
                                Transition to End
                              </div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>Duration:</span> <span style={{fontWeight: 600}}>{stat.or_duration != null ? `${_fmt(stat.or_duration, 1)}s` : "—"}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>Drop:</span> <span style={{fontWeight: 600}}>{stat.or_pct_drop != null ? `${_fmt(stat.or_pct_drop, 2)}%` : "—"}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>Min:</span> <span style={{fontWeight: 600}}>{_fmt(stat.or_min_val)}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>AUC:</span> <span style={{fontWeight: 600}}>{_fmt(stat.or_area_below)}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>Start Value:</span> <span style={{fontWeight: 600}}>{_fmt(stat.or_trans_val)}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>End Value:</span> <span style={{fontWeight: 600}}>{_fmt(stat.end_val)}</span></div>
                            </div>
                            
                            <div style={{ height: '1px', background: 'rgba(0,0,0,0.08)' }}></div>
                            
                            {/* Green Section */}
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                              <div style={{ color: '#16a34a', fontWeight: 700, marginBottom: '4px', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                <div style={{width: 8, height: 8, borderRadius: '50%', background: '#16a34a'}}></div>
                                Stand to End
                              </div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>Duration:</span> <span style={{fontWeight: 600}}>{stat.gr_duration != null ? `${_fmt(stat.gr_duration, 1)}s` : "—"}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>Drop:</span> <span style={{fontWeight: 600}}>{stat.gr_pct_drop != null ? `${_fmt(stat.gr_pct_drop, 2)}%` : "—"}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>Min:</span> <span style={{fontWeight: 600}}>{_fmt(stat.gr_min_val)}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>AUC:</span> <span style={{fontWeight: 600}}>{_fmt(stat.gr_area_below)}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>Start Value:</span> <span style={{fontWeight: 600}}>{_fmt(stat.gr_stand_val)}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>End Value:</span> <span style={{fontWeight: 600}}>{_fmt(stat.end_val)}</span></div>
                            </div>
                            
                            <div style={{ height: '1px', background: 'rgba(0,0,0,0.08)' }}></div>
                            
                            {/* Blue Section */}
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                              <div style={{ color: '#2563eb', fontWeight: 700, marginBottom: '4px', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                <div style={{width: 8, height: 8, borderRadius: '50%', background: '#2563eb'}}></div>
                                Baseline-Based
                              </div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>Duration:</span> <span style={{fontWeight: 600}}>{stat.rec_duration != null ? `${_fmt(stat.rec_duration, 1)}s` : "—"}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>Drop:</span> <span style={{fontWeight: 600}}>{stat.rec_pct_drop != null ? `${_fmt(stat.rec_pct_drop, 2)}%` : "—"}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>Min:</span> <span style={{fontWeight: 600}}>{_fmt(stat.rec_min_val)}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>AUC:</span> <span style={{fontWeight: 600}}>{_fmt(stat.rec_area)}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>Baseline Value:</span> <span style={{fontWeight: 600}}>{_fmt(stat.baseline)}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>Started In:</span> <span style={{fontWeight: 600}}>{stat.rec_started_in || "—"}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 4 }}>
                                <span style={{ color: 'var(--text-muted)' }}>End Marker {endMarkerOverrides[stat.id] ? <span style={{color: '#ea580c'}}>(Edited)</span> : ''}:</span>
                                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                                  <span style={{fontWeight: 600}}>{stat.rec_end_ms ? new Date(stat.rec_end_ms).toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }) : "Not found"}</span>
                                  {endMarkerOverrides[stat.id] && (
                                    <button
                                      onClick={() => {
                                        const newOverrides = { ...endMarkerOverrides };
                                        delete newOverrides[stat.id];
                                        setEndMarkerOverrides(newOverrides);
                                        processData(sessionId, settings, newOverrides);
                                      }}
                                      style={{
                                        background: 'rgba(239, 68, 68, 0.1)', color: '#ef4444', border: '1px solid rgba(239, 68, 68, 0.2)', borderRadius: '4px', padding: '2px 8px', fontSize: '11px', cursor: 'pointer', fontWeight: 600
                                      }}
                                    >
                                      Reset
                                    </button>
                                  )}
                                  <button
                                    onClick={() => {
                                      setEditingTestIdx(stat.id);
                                      if (plotContainerRef.current) {
                                        plotContainerRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
                                      }
                                    }}
                                    style={{
                                      background: 'var(--accent-blue)', color: 'white', border: 'none', borderRadius: '4px', padding: '2px 8px', fontSize: '11px', cursor: 'pointer', fontWeight: 600
                                    }}
                                  >
                                    {stat.rec_end_ms ? "Edit" : "Add"}
                                  </button>
                                </div>
                              </div>
                            </div>
                            
                          </div>
                        );
                      })() : (
                        <div style={{
                          background: 'rgba(255, 255, 255, 0.4)',
                          backdropFilter: 'blur(16px)',
                          WebkitBackdropFilter: 'blur(16px)',
                          color: 'var(--text-muted)',
                          padding: '32px 16px',
                          borderRadius: '16px',
                          border: '1px dashed rgba(0,0,0,0.15)',
                          fontSize: '13px',
                          fontFamily: 'var(--font-sans)',
                          textAlign: 'center',
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          gap: '12px'
                        }}>
                          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="11" cy="11" r="8"></circle>
                            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                            <line x1="11" y1="8" x2="11" y2="14"></line>
                            <line x1="8" y1="11" x2="14" y2="11"></line>
                          </svg>
                          {visibleStats.length === 0 ? "Pan the plot to view a test's statistics." : "Zoom in to a single test to view its statistics."}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ) : null}

              {/* Navigation Buttons Centered Below Plot */}
              {analysisView === 'Supine to Standing Analysis' && analysisStats && analysisStats.length > 0 && (
                <div style={{ display: 'flex', justifyContent: 'center', marginTop: 16 }}>
                  <div className="flex-row" style={{ gap: 4, padding: '8px 16px', background: 'rgba(255,255,255,0.5)', backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)', borderRadius: 12, border: '1px solid var(--border-color)', boxShadow: '0 4px 12px rgba(0,0,0,0.03)' }}>
                    <button 
                      className="btn btn-secondary" 
                      onClick={() => jumpToTest(0)}
                      disabled={currentTestIdx <= 0 && !noTestFocused}
                      style={{ padding: '6px 8px', fontSize: 13, borderRadius: 6, display: 'flex', alignItems: 'center', border: '1px solid var(--border-color)', background: (currentTestIdx <= 0 && !noTestFocused) ? 'rgba(0,0,0,0.02)' : '#fff', color: (currentTestIdx <= 0 && !noTestFocused) ? 'var(--text-muted)' : 'var(--text-main)', cursor: (currentTestIdx <= 0 && !noTestFocused) ? 'not-allowed' : 'pointer', opacity: (currentTestIdx <= 0 && !noTestFocused) ? 0.5 : 1 }}
                      title="Go to First Test"
                    >
                      <ChevronsLeft size={16} />
                    </button>
                    <button 
                      className="btn btn-secondary" 
                      onClick={() => jumpToTest(currentTestIdx - 1)}
                      disabled={currentTestIdx <= 0 || noTestFocused}
                      style={{ padding: '6px 12px', fontSize: 13, borderRadius: 6, display: 'flex', alignItems: 'center', gap: 4, border: '1px solid var(--border-color)', background: (currentTestIdx <= 0 || noTestFocused) ? 'rgba(0,0,0,0.02)' : '#fff', color: (currentTestIdx <= 0 || noTestFocused) ? 'var(--text-muted)' : 'var(--text-main)', cursor: (currentTestIdx <= 0 || noTestFocused) ? 'not-allowed' : 'pointer', opacity: (currentTestIdx <= 0 || noTestFocused) ? 0.5 : 1 }}
                    >
                      <ChevronLeft size={16} /> Prev
                    </button>
                    
                    <div id="test-indicator" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minWidth: 60, padding: '0 4px', fontSize: 13, fontWeight: 600, color: 'var(--text-main)', margin: '0 4px' }}>
                      {noTestFocused ? `- / ${analysisStats.length}` : `${currentTestIdx + 1} / ${analysisStats.length}`}
                    </div>

                    <button 
                      className="btn btn-secondary" 
                      onClick={() => jumpToTest(currentTestIdx + 1)}
                      disabled={currentTestIdx >= analysisStats.length - 1 || noTestFocused}
                      style={{ padding: '6px 12px', fontSize: 13, borderRadius: 6, display: 'flex', alignItems: 'center', gap: 4, border: '1px solid var(--border-color)', background: (currentTestIdx >= analysisStats.length - 1 || noTestFocused) ? 'rgba(0,0,0,0.02)' : '#fff', color: (currentTestIdx >= analysisStats.length - 1 || noTestFocused) ? 'var(--text-muted)' : 'var(--text-main)', cursor: (currentTestIdx >= analysisStats.length - 1 || noTestFocused) ? 'not-allowed' : 'pointer', opacity: (currentTestIdx >= analysisStats.length - 1 || noTestFocused) ? 0.5 : 1 }}
                    >
                      Next <ChevronRight size={16} />
                    </button>
                    <button 
                      className="btn btn-secondary" 
                      onClick={() => jumpToTest(analysisStats.length - 1)}
                      disabled={currentTestIdx >= analysisStats.length - 1 && !noTestFocused}
                      style={{ padding: '6px 8px', fontSize: 13, borderRadius: 6, display: 'flex', alignItems: 'center', border: '1px solid var(--border-color)', background: (currentTestIdx >= analysisStats.length - 1 && !noTestFocused) ? 'rgba(0,0,0,0.02)' : '#fff', color: (currentTestIdx >= analysisStats.length - 1 && !noTestFocused) ? 'var(--text-muted)' : 'var(--text-main)', cursor: (currentTestIdx >= analysisStats.length - 1 && !noTestFocused) ? 'not-allowed' : 'pointer', opacity: (currentTestIdx >= analysisStats.length - 1 && !noTestFocused) ? 0.5 : 1 }}
                      title="Go to Last Test"
                    >
                      <ChevronsRight size={16} />
                    </button>

                    <div style={{ width: 1, height: 24, background: 'var(--border-color)', margin: '0 8px' }}></div>

                    <button 
                      className="btn btn-secondary" 
                      onClick={() => jumpToTest(currentTestIdx)}
                      disabled={disableRecenter}
                      style={{ padding: '6px 12px', fontSize: 13, borderRadius: 6, display: 'flex', alignItems: 'center', gap: 6, border: '1px solid var(--border-color)', background: disableRecenter ? 'rgba(0,0,0,0.02)' : '#fff', color: disableRecenter ? 'var(--text-muted)' : 'var(--text-main)', cursor: disableRecenter ? 'not-allowed' : 'pointer', opacity: disableRecenter ? 0.5 : 1 }}
                      title="Recenter Current Test"
                    >
                      <Target size={16} /> Recenter
                    </button>
                  </div>
                </div>
              )}
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
