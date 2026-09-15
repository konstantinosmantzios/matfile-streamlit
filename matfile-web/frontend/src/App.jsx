import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import axios from 'axios';
import { Activity, UploadCloud, Download, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, Target, X } from 'lucide-react';
import Plot from 'react-plotly.js';
import FilterSandbox from './components/FilterSandbox';
import Tip from './components/Tip';
import DefaultsModal from './components/DefaultsModal';
import {
  groupLabelStyle,
  toggleBtnStyle,
  settingsGroupStyle,
  settingsGroupTitleStyle,
  settingsFieldLabelStyle,
  settingsInputBoxStyle,
  settingsInputStyle,
  settingsUnitStyle,
  settingsSelectStyle,
  settingsCheckboxLabelStyle,
  settingsCheckboxStyle,
} from './components/settingsStyles';

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

const fmtClock = (ms) => {
  const d = new Date(ms);
  const pad = (n, l = 2) => String(n).padStart(l, '0');
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
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


  useEffect(() => {
    if (analysisView !== 'Supine to Standing Analysis') {
      setEditingTestIdx(null);
      setDraftEndMarker(null);
    }
  }, [analysisView]);

  const [localAnalysisSettings, setLocalAnalysisSettings] = useState({
    analysisBaselineWindow: storedDefaults.analysisBaselineWindow ?? 30,
    analysisEndMarkerWindow: storedDefaults.analysisEndMarkerWindow ?? 10,
    baselineEndComment: storedDefaults.baselineEndComment ?? 'Transition'
  });

  const [endMarkerOverrides, setEndMarkerOverrides] = useState({});
  const [editingTestIdx, setEditingTestIdx] = useState(null);
  const [draftEndMarker, setDraftEndMarker] = useState(null);
  const [defaultsOpen, setDefaultsOpen] = useState(false);
  const [storedDefaultsState, setStoredDefaultsState] = useState(storedDefaults);

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
  const manualZoomRef = useRef(false); // true when user has manually panned/zoomed; false when zoom was set by jumpToTest

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

  const buildCurrentConfig = () => {
    // Merge the live settings (filters, resampling, analysis) into one captured
    // config object. 'selectedSignal' is session-specific, so it is excluded.
    const config = {
      ...settings,
      ...localAnalysisSettings,
      compareGaussian,
    };
    delete config.selectedSignal;
    return config;
  };

  const persistDefaults = (config) => {
    const merged = { ...config };
    delete merged.selectedSignal;
    const stored = JSON.stringify(merged);
    localStorage.setItem('matAnalyzerDefaults', stored);
    setStoredDefaultsState(merged);
    return merged;
  };

  const handleSaveDefaults = () => {
    persistDefaults(buildCurrentConfig());
    setDefaultsOpen(true);
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
    const interval = setInterval(fetchMemory, 3000);
    return () => clearInterval(interval);
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
      manualZoomRef.current = false;
      
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
    manualZoomRef.current = true;
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
    if (editingTestIdx === null || !e.points || e.points.length === 0) return;
    const pt = e.points[0];
    const trace = pt.data || {};
    // A candidate marker (interpolated baseline crossing) was clicked — use its exact timepoint
    if (trace.trace_id === 'edit_candidates') {
      const cd = pt.customdata;
      const candMs = typeof cd === 'number' ? cd : (Array.isArray(cd) && cd.length ? cd[0] : undefined);
      if (typeof candMs === 'number' && !isNaN(candMs)) {
        setDraftEndMarker(candMs);
        return;
      }
    }
    const x = pt.x;
    const xMs = typeof x === 'number' ? x : new Date(x).getTime();
    if (isNaN(xMs)) return;
    // Otherwise snap to the nearest datapoint on the resampled line
    const resTrace = plotData.find(t => t.trace_id === 'resampled');
    if (resTrace && Array.isArray(resTrace.x) && Array.isArray(resTrace.y)) {
      let nearest = null;
      let best = Infinity;
      for (let i = 0; i < resTrace.x.length; i++) {
        const v = resTrace.x[i];
        const vy = resTrace.y[i];
        if (v === null || v === undefined || vy === null || vy === undefined) continue;
        const d = Math.abs(v - xMs);
        if (d < best) { best = d; nearest = v; }
      }
      if (nearest !== null && nearest !== undefined) {
        setDraftEndMarker(Number(nearest));
        return;
      }
    }
    setDraftEndMarker(xMs);
  }, [editingTestIdx, plotData]);

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

  // While editing an end marker, find the stat being edited by its backend id (comment index).
  const editingStat = useMemo(
    () => (editingTestIdx !== null ? (analysisStats.find(s => s.id === editingTestIdx) || null) : null),
    [editingTestIdx, analysisStats]
  );

  // Detect candidate timepoints where the resampled line equals the baseline value,
  // linearly interpolating between the two flanking samples. Search window:
  // [test start, standing + 30 s].
  const editCandidates = useMemo(() => {
    if (!editingStat) return [];
    const baseline = editingStat.baseline;
    if (baseline === null || baseline === undefined || !isFinite(baseline)) return [];

    const resTrace = plotData.find(t => t.trace_id === 'resampled');
    if (!resTrace || !Array.isArray(resTrace.x) || !Array.isArray(resTrace.y)) return [];

    const xs = resTrace.x;
    const ys = resTrace.y;
    const startMs = editingStat.t_trans_ms || editingStat.t_start_ms || 0;
    const endMs = (editingStat.t_stand_ms || 0) + 30000;
    const out = [];

    for (let i = 0; i < xs.length - 1; i++) {
      const x0 = xs[i], x1 = xs[i + 1];
      if (x0 === null || x0 === undefined || x1 === null || x1 === undefined) continue;
      if (x1 < startMs) continue;
      if (x0 > endMs) break;
      const y0 = ys[i], y1 = ys[i + 1];
      if (y0 === null || y0 === undefined || y1 === null || y1 === undefined) continue;
      const d0 = y0 - baseline;
      const d1 = y1 - baseline;
      if (d0 === 0) {
        if (x0 >= startMs && x0 <= endMs) out.push(x0);
        continue;
      }
      if ((d0 < 0 && d1 > 0) || (d0 > 0 && d1 < 0)) {
        const frac = -d0 / (d1 - d0);
        const xCross = x0 + frac * (x1 - x0);
        if (xCross >= startMs && xCross <= endMs) out.push(xCross);
      }
    }
    // De-duplicate nearly identical crossings
    const deduped = [];
    for (const x of out) {
      if (!deduped.length || Math.abs(deduped[deduped.length - 1] - x) > 1) deduped.push(x);
    }
    return deduped;
  }, [editingStat, plotData]);

  // Extra traces shown to the user while editing an end marker.
  const editOverlayTraces = useMemo(() => {
    if (!editingStat) return [];
    const traces = [];
    const startMs = editingStat.t_trans_ms || editingStat.t_start_ms || 0;
    const endMs = (editingStat.t_stand_ms || 0) + 30000;

    if (editCandidates.length > 0) {
      traces.push({
        trace_id: 'edit_candidates',
        x: editCandidates,
        y: editCandidates.map(() => editingStat.baseline),
        customdata: editCandidates,
        type: 'scattergl',
        mode: 'markers',
        name: 'End Marker Candidates',
        marker: { color: '#16a34a', size: 12, symbol: 'diamond', line: { color: '#fff', width: 1 } },
        showlegend: false,
        hoverinfo: 'x',
        xaxis: 'x',
        yaxis: 'y',
        hovertemplate: 'Candidate end marker: %{x}<extra></extra>'
      });
      traces.push({
        trace_id: 'edit_baseline_line',
        x: [startMs, endMs],
        y: [editingStat.baseline, editingStat.baseline],
        type: 'scattergl',
        mode: 'lines',
        name: 'Target Baseline',
        line: { color: 'rgba(37, 99, 235, 0.9)', width: 1.5, dash: 'dash' },
        showlegend: false,
        hoverinfo: 'skip'
      });
    }

    if (draftEndMarker !== null && draftEndMarker !== undefined) {
      const isCandidate = editCandidates.some(c => Math.abs(c - draftEndMarker) < 1);
      // For non-candidate (resampled datapoint) selections, place the marker at the
      // actual resampled line value so it sits on top of the plotted data.
      let selY = editingStat.baseline;
      if (!isCandidate) {
        const resTrace = plotData.find(t => t.trace_id === 'resampled');
        let bestY = null;
        let bestD = Infinity;
        if (resTrace && Array.isArray(resTrace.x) && Array.isArray(resTrace.y)) {
          for (let i = 0; i < resTrace.x.length; i++) {
            const x = resTrace.x[i];
            const y = resTrace.y[i];
            if (x === null || x === undefined || y === null || y === undefined) continue;
            const d = Math.abs(x - draftEndMarker);
            if (d < bestD) { bestD = d; bestY = y; }
          }
        }
        if (bestY !== null) selY = bestY;
      }
      traces.push({
        trace_id: 'edit_selected',
        x: [draftEndMarker],
        y: [selY],
        type: 'scattergl',
        mode: 'markers',
        name: 'Selected End Marker',
        marker: { color: '#ea580c', size: 14, symbol: 'circle', line: { color: '#fff', width: 1.5 } },
        showlegend: false,
        hoverinfo: 'skip'
      });
    }
    return traces;
  }, [editingStat, editCandidates, draftEndMarker, plotData]);

  const displayedPlotData = useMemo(
    () => (editingTestIdx !== null ? [...plotData, ...editOverlayTraces] : plotData),
    [plotData, editingTestIdx, editOverlayTraces]
  );

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
    manualZoomRef.current = false;
    
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
    if (analysisView === 'Filtering Preview' && currentZoomRef.current && latestSessionIdRef.current) {
      // Tab switched back to Filtering Preview with an active zoom: restore
      // the viewport-resolution traces for that range (updateFromBackend resets
      // plotData to the coarse full-range traces otherwise).
      const t = setTimeout(() => {
        fetchViewport(currentZoomRef.current.xMinMs, currentZoomRef.current.xMaxMs, latestSessionIdRef.current);
      }, 100);
      return () => clearTimeout(t);
    }
  }, [updatePlotFromBackendData, analysisView]);

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
        } else if (!manualZoomRef.current && targetTestIdxRef.current >= 0 && response.data.analysis_stats) {
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
                  <Tip text="Approximate memory used by the active backend session on the server (in MB). Released on session close." style={{ color: 'inherit' }}>
                    <span className="badge">
                      Server: {memoryUsage.toFixed(0)} MB
                    </span>
                  </Tip>
                )}
                {getPlottedHz() !== null && (
                  <Tip text="Approximate effective sampling rate of the currently displayed traces (points per second). Lower = coarser detail." style={{ color: 'inherit' }}>
                    <span className="badge badge-active">
                      Res: ~{getPlottedHz()} Hz
                    </span>
                  </Tip>
                )}
              </div>
            </div>
            
            <div className="flex-row">
              <div className="flex-row" style={{ marginRight: 16, gap: 16 }}>
                <button
                  className="btn btn-secondary"
                  onClick={handleSaveDefaults}
                  style={{ padding: '4px 10px', fontSize: 12, borderRadius: 6, display: 'flex', alignItems: 'center', border: '1px solid var(--border-color)', background: '#fff', color: 'var(--text-main)', cursor: 'pointer' }}
                >
                  <Tip text="Capture the current filters, resampling, and analysis settings and open the Defaults manager to view, edit, or reset them.">
                    <span style={{ display: 'inline-flex', alignItems: 'center' }}>Set Defaults</span>
                  </Tip>
                </button>
              </div>
              {fileName && (
                <Tip text="Name of the file currently loaded in this session. Click X to close and release the session." style={{ color: 'inherit' }}>
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
                </Tip>
              )}
            </div>
          </div>
          <div className={`expand-row ${editingTestIdx !== null ? 'expanded' : 'collapsed'}`}>
            <span>
              {editingTestIdx !== null
                ? (editCandidates.length > 0
                    ? `Editing end marker — ${editCandidates.length} candidate(s) detected. Use the editor panel above the chart.`
                    : 'Editing end marker — click a point directly on the resampled line, then Apply.')
                : ''}
            </span>
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
              <Tip text="Click or drag a LabChart .mat or converted .parquet file to load and plot all signals automatically." />
            </label>

            <div style={{ borderTop: '1px solid var(--border-color)', width: '100%', maxWidth: 400, margin: '10px 0' }}></div>

            {/* Conversion Zone */}
            <input type="file" accept=".mat" onChange={handleConvertMat} style={{ display: 'none' }} id="convert-upload" />
            <label htmlFor="convert-upload" className="btn btn-primary" style={{ cursor: 'pointer', display: 'flex', gap: 8, padding: '10px 20px', borderRadius: '12px' }}>
              <Download size={18} />
              {isConverting ? 'Converting...' : 'Convert .MAT to .Parquet locally'}
              <Tip text="Converts the selected .mat file into a .parquet file on your browser and downloads the result. Use before uploading for much faster load times." style={{ color: '#fff' }} />
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
              <div style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: 16, marginBottom: 20 }}>
                
                {/* Row 1: View toggle + Export */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <span style={groupLabelStyle}>View</span>
                    <div style={{ display: 'flex', background: 'rgba(0,0,0,0.05)', borderRadius: '10px', padding: '3px' }}>
                      <Tip text="Show the raw/resampled signal with the applied filters (before statistical analysis).">
                        <span style={{ display: 'inline-flex' }}>
                          <button onClick={() => setAnalysisView('Filtering Preview')} style={{ ...toggleBtnStyle, background: analysisView === 'Filtering Preview' ? '#fff' : 'transparent', color: analysisView === 'Filtering Preview' ? 'var(--text-main)' : 'var(--text-muted)', boxShadow: analysisView === 'Filtering Preview' ? '0 1px 3px rgba(0,0,0,0.08)' : 'none' }}>Filtering</button>
                        </span>
                      </Tip>
                      <Tip text="Show the analyzed signal with baseline, transition, standing, and recovery markers plus per-test statistics.">
                        <span style={{ display: 'inline-flex' }}>
                          <button onClick={() => setAnalysisView('Supine to Standing Analysis')} style={{ ...toggleBtnStyle, background: analysisView === 'Supine to Standing Analysis' ? '#fff' : 'transparent', color: analysisView === 'Supine to Standing Analysis' ? 'var(--text-main)' : 'var(--text-muted)', boxShadow: analysisView === 'Supine to Standing Analysis' ? '0 1px 3px rgba(0,0,0,0.08)' : 'none' }}>Analysis</button>
                        </span>
                      </Tip>
                    </div>
                  </div>

                  <button className="btn btn-primary" onClick={exportData} disabled={isLoading} style={{ padding: '6px 14px', fontSize: 13, borderRadius: 8 }}>
                    <Download size={14} />
                    <Tip text="Compute the final statistics for all signals and download an Excel workbook (Statistics, Resampled Data, Metadata)." style={{ color: '#fff' }}>
                      <span style={{ display: 'inline-flex', alignItems: 'center' }}>Export</span>
                    </Tip>
                  </button>
                </div>

                {/* Row 2: Signal pills + Resampling */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
                  
                  {/* Signal selector */}
                  {availableSignals.length > 0 && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={groupLabelStyle}>Signal</span>
                      <Tip text="Select which signal trace to analyze. FP = Finger Pressure, MAP = Mean Arterial Pressure, CBF = Cerebral Blood Flow, HR = Heart Rate." />
                      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                        {availableSignals
                          .filter(sig => {
                            const s = sig.toLowerCase();
                            return s.includes('finger') || s.includes('map') || s.includes('hr') || s.includes('cbf');
                          })
                          .map(sig => {
                            const isSelected = settings.selectedSignal === sig;
                            let shortName = sig;
                            if (sig.includes('Finger Pressure')) shortName = 'FP';
                            if (sig.includes('CBF')) shortName = 'CBF';
                            if (sig.includes('MAP')) shortName = 'MAP';
                            if (sig.includes('HR')) shortName = 'HR';
                            return (
                              <button
                                key={sig}
                                onClick={() => handleSignalSelect(sig)}
                                style={{
                                  padding: '5px 14px', borderRadius: '16px', fontSize: '12px', fontWeight: 600, cursor: 'pointer',
                                  border: '1px solid',
                                  borderColor: isSelected ? 'var(--accent-blue)' : 'var(--border-color)',
                                  background: isSelected ? 'var(--accent-blue)' : '#fff',
                                  color: isSelected ? '#fff' : 'var(--text-muted)',
                                  transition: 'all 0.2s'
                                }}
                              >{shortName}</button>
                            );
                          })}
                      </div>
                    </div>
                  )}

                  {/* Resampling controls */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={groupLabelStyle}>Resampling</span>
                    <Tip text="Maps the raw signal onto a uniform time (Time-based) or heartbeat-aligned (Beat-based) grid so statistics can be compared across tests." />
                    <select className="form-select form-select-sm" value={settings.resampleMode} onChange={e => updateSetting('resampleMode', e.target.value)} style={{ padding: '5px 10px', borderRadius: '8px', border: '1px solid var(--border-color)', outline: 'none', fontSize: 12, background: '#fff' }}>
                      <option>Beat-based</option>
                      <option>Time-based</option>
                    </select>
                    
                    {settings.resampleMode === 'Time-based' ? (
                      <div className="flex-row" style={{ gap: 8 }}>
                        <span style={{ fontSize: 12, color: 'var(--text-muted)', minWidth: 36, textAlign: 'right' }}>
                          {settings.resampleRateTime === 60 ? '1 min' : `${settings.resampleRateTime}s`}
                        </span>
                        <Tip text="Fixed time interval (seconds) between samples of the resampled signal.">
                          <input 
                            type="range" min="0" max="5" step="1" 
                            value={[1, 5, 10, 15, 30, 60].indexOf(settings.resampleRateTime) !== -1 ? [1, 5, 10, 15, 30, 60].indexOf(settings.resampleRateTime) : 0} 
                            onChange={e => updateSetting('resampleRateTime', [1, 5, 10, 15, 30, 60][parseInt(e.target.value)])} 
                            style={{ width: 120 }} 
                          />
                        </Tip>
                      </div>
                    ) : (
                      <div className="flex-row" style={{ gap: 8 }}>
                        <span style={{ fontSize: 12, color: 'var(--text-muted)', minWidth: 48, textAlign: 'right' }}>
                          {settings.resampleRateBeat} beats
                        </span>
                        <Tip text="The resampling window centers on a beat (even intervals) or between beats (odd intervals) using N beats + 1 peak.">
                          <input 
                            type="range" min="0" max="3" step="1" 
                            value={[1, 2, 5, 10].indexOf(settings.resampleRateBeat) !== -1 ? [1, 2, 5, 10].indexOf(settings.resampleRateBeat) : 2} 
                            onChange={e => updateSetting('resampleRateBeat', [1, 2, 5, 10][parseInt(e.target.value)])} 
                            style={{ width: 120 }} 
                          />
                        </Tip>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              
              {/* Analysis Settings Panel */}
              {analysisView === 'Supine to Standing Analysis' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 0, marginTop: 4, marginBottom: 12 }}>
                  
                  {/* Baseline group */}
                  <div style={settingsGroupStyle}>
                    <span style={settingsGroupTitleStyle}>Baseline</span>
                    <Tip text="Baseline parameters control how the blue baseline value is computed and where it ends for each test." />
                    <div className="flex-row" style={{ gap: 20, flexWrap: 'wrap' }}>
                      <div className="flex-row" style={{ gap: 8 }}>
                        <label style={settingsFieldLabelStyle}>
                          Window
                          <Tip text="Duration of the baseline value (blue line) before the comment selected in 'Baseline Ends At'. Used for the Baseline-based method." />
                        </label>
                        <div style={settingsInputBoxStyle}>
                          <input 
                            type="number"
                            value={localAnalysisSettings.analysisBaselineWindow}
                            onChange={e => setLocalAnalysisSettings(prev => ({ ...prev, analysisBaselineWindow: parseInt(e.target.value) || 0 }))}
                            style={settingsInputStyle}
                          />
                          <span style={settingsUnitStyle}>sec</span>
                        </div>
                      </div>
                      <div className="flex-row" style={{ gap: 8 }}>
                        <label style={settingsFieldLabelStyle}>
                          Ends at
                          <Tip text="The reference comment point to calculate the baseline value before it." />
                        </label>
                        <select 
                          value={localAnalysisSettings.baselineEndComment}
                          onChange={e => setLocalAnalysisSettings(prev => ({ ...prev, baselineEndComment: e.target.value }))}
                          style={settingsSelectStyle}
                        >
                          <option value="Transition">Transition</option>
                          <option value="Standing">Standing</option>
                        </select>
                      </div>
                    </div>
                  </div>

                  {/* End window group */}
                  <div style={settingsGroupStyle}>
                    <span style={settingsGroupTitleStyle}>End of Test Window</span>
                    <Tip text="Controls how the end value of the test is sampled after the end marker / transition." />
                    <div className="flex-row" style={{ gap: 8 }}>
                      <label style={settingsFieldLabelStyle}>
                        Duration
                        <Tip text="How long after the marker to sample the end value for comparison. Can be adjusted per-test from the stats card on the right." />
                      </label>
                      <div style={settingsInputBoxStyle}>
                        <input 
                          type="number"
                          value={localAnalysisSettings.analysisEndMarkerWindow}
                          onChange={e => setLocalAnalysisSettings(prev => ({ ...prev, analysisEndMarkerWindow: parseInt(e.target.value) || 0 }))}
                          style={settingsInputStyle}
                        />
                        <span style={settingsUnitStyle}>sec</span>
                      </div>
                    </div>
                  </div>

                  {/* MAP Gaussian (FP only) */}
                  {settings.selectedSignal && settings.selectedSignal.includes('Finger Pressure') && (
                    <div style={settingsGroupStyle}>
                      <span style={settingsGroupTitleStyle}>MAP - Gaussian Overlay</span>
                      <Tip text="The Mean Arterial Pressure trace smoothed with a ~5 second Gaussian kernel." />
                      <div className="flex-row" style={{ gap: 16, flexWrap: 'wrap' }}>
                        <label style={settingsCheckboxLabelStyle}>
                          <input type="checkbox" style={settingsCheckboxStyle} checked={settings.useMapGaussianForStats} onChange={e => {
                            const isChecked = e.target.checked;
                            setSettings(prev => ({ ...prev, useMapGaussianForStats: isChecked }));
                            if (isChecked) setCompareGaussian(true);
                          }} />
                          Use for stats
                          <Tip text="Use MAP - Gaussian (5sec) for calculating minimum/maximum and endpoint statistics." />
                        </label>
                        <label style={settingsCheckboxLabelStyle}>
                          <input type="checkbox" style={settingsCheckboxStyle} checked={compareGaussian} onChange={e => setCompareGaussian(e.target.checked)} />
                          Show on chart
                          <Tip text="Show the MAP - Gaussian (5sec) line on the chart." />
                        </label>
                      </div>
                    </div>
                  )}

                  {/* Apply button row */}
                  <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 4 }}>
                    <button 
                      className="btn btn-primary" 
                      onClick={handleApplyAnalysisSettings}
                      style={{ padding: '6px 14px', fontSize: 12, borderRadius: 8 }}
                    >
                      <Tip text="Apply the locally edited baseline and end-marker window values to the backend and re-run the analysis." style={{ color: '#fff' }}>
                        <span style={{ display: 'inline-flex', alignItems: 'center' }}>Apply Settings</span>
                      </Tip>
                    </button>
                  </div>
                </div>
              )}

              {/* Filter Settings — same position as the Analysis Settings panel */}
              {isFilteringPreview && availableSignals.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 0, marginTop: 4, marginBottom: 12 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                    <span style={settingsGroupTitleStyle}>Filter Settings</span>
                    <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                      — applied only in the Filtering view to the raw signal
                    </span>
                    <Tip text="Filters are applied to the raw signal for the preview shown in the Filtering view. Changes re-run the pipeline automatically." />
                  </div>
                  <div className="flex-row" style={{ alignItems: 'flex-start' }}>
                    <div style={{ flex: 1, paddingRight: 24, borderRight: '1px solid var(--border-color)' }}>
                      <FilterSandbox prefix="fp" label="Finger Pressure Filter" settings={settings} setSettings={setSettings} />
                    </div>
                    <div style={{ flex: 1, paddingLeft: 24 }}>
                      <FilterSandbox prefix="cbf" label="CBF Filter" settings={settings} setSettings={setSettings} />
                    </div>
                  </div>
                </div>
              )}

              {plotData.length > 0 ? (
                <div style={{ width: '100%', display: 'flex', flexDirection: 'row', gap: 24 }}>
                  <div ref={plotContainerRef} style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
                    {editingTestIdx !== null && (
                      <div style={{
                        background: 'var(--accent-blue)',
                        color: 'white',
                        borderRadius: '14px 14px 0 0',
                        padding: '12px 16px',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 10,
                        boxShadow: '0 -4px 16px rgba(0,0,0,0.06)'
                      }}>
                        {/* Title + actions */}
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', gap: 12 }}>
                          <span style={{ fontSize: 14, fontWeight: 700 }}>
                            {editingStat && editingStat.baseline != null
                              ? <>Set End Marker · baseline <b>{editingStat.baseline.toFixed(2)}</b></>
                              : 'Set End Marker'}
                          </span>
                          <div style={{ display: 'flex', gap: 8, flexShrink: 0 }}>
                            <button
                              onClick={() => {
                                const newOverrides = { ...endMarkerOverrides, [editingTestIdx]: draftEndMarker };
                                setEndMarkerOverrides(newOverrides);
                                setEditingTestIdx(null);
                                setDraftEndMarker(null);
                                processData(sessionId, settings, newOverrides);
                              }}
                              disabled={!draftEndMarker}
                              style={{ background: '#16a34a', border: 'none', color: 'white', padding: '6px 18px', borderRadius: '16px', cursor: draftEndMarker ? 'pointer' : 'not-allowed', fontWeight: 700, fontSize: 12 }}
                            >
                              <Tip text="Save the selected timepoint (baseline crossing or resampled point) as this test's end marker and re-run the statistics." style={{ color: '#fff' }}>
                                <span style={{ display: 'inline-flex', alignItems: 'center' }}>Apply</span>
                              </Tip>
                            </button>
                            <button
                              onClick={() => {
                                setEditingTestIdx(null);
                                setDraftEndMarker(null);
                              }}
                              style={{ background: 'rgba(255,255,255,0.2)', border: 'none', color: 'white', padding: '6px 18px', borderRadius: '16px', cursor: 'pointer', fontWeight: 700, fontSize: 12 }}
                            >
                              <Tip text="Discard the selection and close the end-marker editor without saving changes." style={{ color: '#fff' }}>
                                <span style={{ display: 'inline-flex', alignItems: 'center' }}>Cancel</span>
                              </Tip>
                            </button>
                          </div>
                        </div>

                        {/* Instructions */}
                        <span style={{ fontSize: 12, fontWeight: 500, lineHeight: 1.5, opacity: 0.98 }}>
                          {editCandidates.length > 0 ? (
                            <>
                              Detected <b>{editCandidates.length}</b> candidate{editCandidates.length === 1 ? '' : 's'} where the resampled line
                              crosses the baseline value (from test start → standing + 30 s). Green diamonds mark each crossing on the chart.
                              Click a candidate below, a <b>green diamond</b> on the chart, or any point directly on the resampled line.
                            </>
                          ) : (
                            'No baseline crossings found in the window (test start → standing + 30 s). Click any point directly on the resampled line to use as the end marker.'
                          )}
                        </span>

                        {/* Candidate chips */}
                        {editCandidates.length > 0 && (
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                            {editCandidates.map((c, idx) => {
                              const isSelected = draftEndMarker != null && Math.abs(draftEndMarker - c) < 1;
                              return (
                                <button
                                  key={idx}
                                  onClick={() => setDraftEndMarker(c)}
                                  style={{
                                    background: isSelected ? '#fff' : 'rgba(255,255,255,0.16)',
                                    color: isSelected ? '#16a34a' : 'white',
                                    border: isSelected ? '2px solid #fff' : '1px solid rgba(255,255,255,0.3)',
                                    padding: '3px 11px',
                                    borderRadius: '14px',
                                    cursor: 'pointer',
                                    fontWeight: 700,
                                    fontSize: 12
                                  }}
                                >
                                  <Tip text="Use this baseline crossing as the end marker" style={{ color: 'inherit', fontWeight: 700, fontSize: 12 }}>
                                    <span style={{ display: 'inline-flex' }}>#{idx + 1} · {fmtClock(c)}</span>
                                  </Tip>
                                </button>
                              );
                            })}
                          </div>
                        )}

                        {/* Current selection */}
                        <span style={{ fontSize: 12, fontWeight: 600 }}>
                          {draftEndMarker != null
                            ? <>Selected: <b>{fmtClock(draftEndMarker)}</b>{editCandidates.some(c => Math.abs(c - draftEndMarker) < 1) ? ' · baseline crossing' : ' · resampled point'}</>
                            : 'Nothing selected yet — make a selection above, then press Apply.'}
                        </span>
                      </div>
                    )}
                    <MemoPlot
                      data={displayedPlotData}
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
                              <span><Tip text="The time (in seconds) between the Transition and Standing comments, defining the test's supine-to-standing transition period.">Transition Duration</Tip>: <span style={{ color: 'var(--text-main)', fontWeight: 700 }}>{stat.transition_time != null ? `${_fmt(stat.transition_time, 1)}s` : "—"}</span></span>
                            </div>
                            
                            {/* Stacked Sections */}
                            
                            {/* Orange Section */}
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                              <div style={{ color: '#ea580c', fontWeight: 700, marginBottom: '4px', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                <div style={{width: 8, height: 8, borderRadius: '50%', background: '#ea580c'}}></div>
                                Transition to End
                              </div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}><Tip text="Time (in seconds) from the transition comment to the end marker for this method.">Duration</Tip>:</span> <span style={{fontWeight: 600}}>{stat.or_duration != null ? `${_fmt(stat.or_duration, 1)}s` : "—"}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}><Tip text="Percentage drop from the start value to the minimum reached during the window.">Drop</Tip>:</span> <span style={{fontWeight: 600}}>{stat.or_pct_drop != null ? `${_fmt(stat.or_pct_drop, 2)}%` : "—"}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}><Tip text="Absolute minimum value reached during the window.">Min</Tip>:</span> <span style={{fontWeight: 600}}>{_fmt(stat.or_min_val)}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}><Tip text="Area Under Curve: the total accumulated deficit below the start value during the test (area above the curve relative to start).">AUC</Tip>:</span> <span style={{fontWeight: 600}}>{_fmt(stat.or_area_below)}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}><Tip text="Value of the signal at the transition comment, used as the starting reference.">Start Value</Tip>:</span> <span style={{fontWeight: 600}}>{_fmt(stat.or_trans_val)}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}><Tip text="Value of the signal at the chosen end marker for comparison.">End Value</Tip>:</span> <span style={{fontWeight: 600}}>{_fmt(stat.end_val)}</span></div>
                            </div>
                            
                            <div style={{ height: '1px', background: 'rgba(0,0,0,0.08)' }}></div>
                            
                            {/* Green Section */}
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                              <div style={{ color: '#16a34a', fontWeight: 700, marginBottom: '4px', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                <div style={{width: 8, height: 8, borderRadius: '50%', background: '#16a34a'}}></div>
                                Stand to End
                              </div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}><Tip text="Time (in seconds) from the standing comment to the end marker for this method.">Duration</Tip>:</span> <span style={{fontWeight: 600}}>{stat.gr_duration != null ? `${_fmt(stat.gr_duration, 1)}s` : "—"}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}><Tip text="Percentage drop from the standing value to the minimum reached during the window.">Drop</Tip>:</span> <span style={{fontWeight: 600}}>{stat.gr_pct_drop != null ? `${_fmt(stat.gr_pct_drop, 2)}%` : "—"}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}><Tip text="Absolute minimum value reached during the window.">Min</Tip>:</span> <span style={{fontWeight: 600}}>{_fmt(stat.gr_min_val)}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}><Tip text="Area Under Curve: the total accumulated deficit below the standing value during the test.">AUC</Tip>:</span> <span style={{fontWeight: 600}}>{_fmt(stat.gr_area_below)}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}><Tip text="Value of the signal at the standing comment, used as the starting reference.">Start Value</Tip>:</span> <span style={{fontWeight: 600}}>{_fmt(stat.gr_stand_val)}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}><Tip text="Value of the signal at the chosen end marker for comparison.">End Value</Tip>:</span> <span style={{fontWeight: 600}}>{_fmt(stat.end_val)}</span></div>
                            </div>
                            
                            <div style={{ height: '1px', background: 'rgba(0,0,0,0.08)' }}></div>
                            
                            {/* Blue Section */}
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                              <div style={{ color: '#2563eb', fontWeight: 700, marginBottom: '4px', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                <div style={{width: 8, height: 8, borderRadius: '50%', background: '#2563eb'}}></div>
                                Baseline-Based
                              </div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}><Tip text="Time (in seconds) from the standing comment to the end marker for this method.">Duration</Tip>:</span> <span style={{fontWeight: 600}}>{stat.rec_duration != null ? `${_fmt(stat.rec_duration, 1)}s` : "—"}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}><Tip text="Percentage drop from the baseline value to the minimum reached during the window.">Drop</Tip>:</span> <span style={{fontWeight: 600}}>{stat.rec_pct_drop != null ? `${_fmt(stat.rec_pct_drop, 2)}%` : "—"}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}><Tip text="Absolute minimum value reached during the window.">Min</Tip>:</span> <span style={{fontWeight: 600}}>{_fmt(stat.rec_min_val)}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}><Tip text="Area Under Curve: the total accumulated deficit below the baseline value during the test.">AUC</Tip>:</span> <span style={{fontWeight: 600}}>{_fmt(stat.rec_area)}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}><Tip text="Mean signal value computed from the baseline window before the reference comment.">Baseline Value</Tip>:</span> <span style={{fontWeight: 600}}>{_fmt(stat.baseline)}</span></div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}><Tip text="The comment defining which segment (before it) was used as the baseline.">Started In</Tip>:</span> <span style={{fontWeight: 600}}>{stat.rec_started_in || "—"}</span></div>
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
                                      <Tip text="Discard the manual end-marker override and fall back to the auto-detected timepoint for this test." style={{ color: '#ef4444' }}>
                                        <span style={{ display: 'inline-flex', alignItems: 'center' }}>Reset</span>
                                      </Tip>
                                    </button>
                                  )}
                                  <button
                                    onClick={() => {
                                      setEditingTestIdx(stat.id);
                                      setDraftEndMarker(null);
                                      // Zoom to the candidate window (test start → standing + 30 s) so the
                                      // user immediately sees the crossing candidates on the chart.
                                      const editStart = stat.t_trans_ms || stat.t_start_ms || 0;
                                      const editEnd = (stat.t_stand_ms || 0) + 30000;
                                      const baseWin = (localAnalysisSettings.analysisBaselineWindow || 30) * 1000;
                                      const zoomS = Math.max(0, editStart - baseWin - 5000);
                                      const zoomE = editEnd + 10000;
                                      currentZoomRef.current = { xMinMs: zoomS, xMaxMs: zoomE };
                                      setPlotXRange({ min: zoomS, max: zoomE });
                                      if (plotContainerRef.current) {
                                        plotContainerRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
                                      }
                                    }}
                                    style={{
                                      background: 'var(--accent-blue)', color: 'white', border: 'none', borderRadius: '4px', padding: '2px 8px', fontSize: '11px', cursor: 'pointer', fontWeight: 600
                                    }}
                                  >
                                    <Tip text={stat.rec_end_ms ? 'Open the end-marker editor to choose a different baseline crossing or resampled point as the end marker.' : 'No auto-detected end marker exists for this test. Open the editor to add one manually.'} style={{ color: '#fff' }}>
                                      <span style={{ display: 'inline-flex', alignItems: 'center' }}>{stat.rec_end_ms ? 'Edit' : 'Add'}</span>
                                    </Tip>
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
                    >
                      <Tip text="Jump to the first test" style={{ color: 'inherit', display: 'inline-flex' }}>
                        <span style={{ display: 'inline-flex', alignItems: 'center' }}><ChevronsLeft size={16} /></span>
                      </Tip>
                    </button>
                    <button 
                      className="btn btn-secondary" 
                      onClick={() => jumpToTest(currentTestIdx - 1)}
                      disabled={currentTestIdx <= 0 || noTestFocused}
                      style={{ padding: '6px 12px', fontSize: 13, borderRadius: 6, display: 'flex', alignItems: 'center', gap: 4, border: '1px solid var(--border-color)', background: (currentTestIdx <= 0 || noTestFocused) ? 'rgba(0,0,0,0.02)' : '#fff', color: (currentTestIdx <= 0 || noTestFocused) ? 'var(--text-muted)' : 'var(--text-main)', cursor: (currentTestIdx <= 0 || noTestFocused) ? 'not-allowed' : 'pointer', opacity: (currentTestIdx <= 0 || noTestFocused) ? 0.5 : 1 }}
                    >
                      <Tip text="Jump to the previous test." style={{ color: 'inherit', display: 'inline-flex', alignItems: 'center' }}>
                        <span style={{ display: 'inline-flex', alignItems: 'center' }}><ChevronLeft size={16} /> Prev</span>
                      </Tip>
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
                      <Tip text="Jump to the next test." style={{ color: 'inherit', display: 'inline-flex', alignItems: 'center' }}>
                        <span style={{ display: 'inline-flex', alignItems: 'center' }}>Next <ChevronRight size={16} /></span>
                      </Tip>
                    </button>
                    <button 
                      className="btn btn-secondary" 
                      onClick={() => jumpToTest(analysisStats.length - 1)}
                      disabled={currentTestIdx >= analysisStats.length - 1 && !noTestFocused}
                      style={{ padding: '6px 8px', fontSize: 13, borderRadius: 6, display: 'flex', alignItems: 'center', border: '1px solid var(--border-color)', background: (currentTestIdx >= analysisStats.length - 1 && !noTestFocused) ? 'rgba(0,0,0,0.02)' : '#fff', color: (currentTestIdx >= analysisStats.length - 1 && !noTestFocused) ? 'var(--text-muted)' : 'var(--text-main)', cursor: (currentTestIdx >= analysisStats.length - 1 && !noTestFocused) ? 'not-allowed' : 'pointer', opacity: (currentTestIdx >= analysisStats.length - 1 && !noTestFocused) ? 0.5 : 1 }}
                    >
                      <Tip text="Jump to the last test." style={{ color: 'inherit', display: 'inline-flex' }}>
                        <span style={{ display: 'inline-flex', alignItems: 'center' }}><ChevronsRight size={16} /></span>
                      </Tip>
                    </button>

                    <div style={{ width: 1, height: 24, background: 'var(--border-color)', margin: '0 8px' }}></div>

                    <button 
                      className="btn btn-secondary" 
                      onClick={() => jumpToTest(currentTestIdx)}
                      disabled={disableRecenter}
                      style={{ padding: '6px 12px', fontSize: 13, borderRadius: 6, display: 'flex', alignItems: 'center', gap: 6, border: '1px solid var(--border-color)', background: disableRecenter ? 'rgba(0,0,0,0.02)' : '#fff', color: disableRecenter ? 'var(--text-muted)' : 'var(--text-main)', cursor: disableRecenter ? 'not-allowed' : 'pointer', opacity: disableRecenter ? 0.5 : 1 }}
                    >
                      <Tip text="Re-center the plot on the current test's expected viewing window (baseline through recovery)." style={{ color: 'inherit', display: 'inline-flex', alignItems: 'center' }}>
                        <span style={{ display: 'inline-flex', alignItems: 'center' }}><Target size={16} /> Recenter</span>
                      </Tip>
                    </button>
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </main>

      <DefaultsModal
        open={defaultsOpen}
        onClose={() => setDefaultsOpen(false)}
        defaults={storedDefaultsState}
        currentConfig={buildCurrentConfig()}
        onSaveDefaults={persistDefaults}
      />
    </div>
  );
}

export default App;
