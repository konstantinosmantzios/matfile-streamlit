import React, { useEffect, useState } from 'react';
import { X, RotateCcw, Save, SlidersHorizontal, Check } from 'lucide-react';
import Tip from './Tip';
import {
  settingsGroupStyle,
  settingsGroupTitleStyle,
  settingsFieldLabelStyle,
  settingsInputBoxStyle,
  settingsInputStyle,
  settingsUnitStyle,
  settingsSelectStyle,
  settingsCheckboxLabelStyle,
  settingsCheckboxStyle,
  FACTORY_DEFAULTS,
} from './settingsStyles';

// Schema describing every persisted setting so the modal can render editors for each.
const FIELD_GROUPS = [
  {
    title: 'Resampling',
    fields: [
      { key: 'resampleMode', label: 'Mode', type: 'select', options: ['Beat-based', 'Time-based'], tip: 'Beat-based resamples each window between detected heartbeats; Time-based resamples at a fixed clock rate.' },
      { key: 'resampleRateBeat', label: 'Beat Window', type: 'select', options: ['1', '2', '5', '10'], unit: 'beats', showWhen: { key: 'resampleMode', value: 'Beat-based' }, tip: 'How many beats the resampling window spans. Even counts center on a beat, odd counts sit between beats.' },
      { key: 'resampleRateTime', label: 'Time Window', type: 'select', options: ['1', '5', '10', '15', '30', '60'], unit: 's', showWhen: { key: 'resampleMode', value: 'Time-based' }, tip: 'Fixed time interval (seconds) between samples of the resampled signal.' },
    ],
  },
  {
    title: 'Filtering — Conditions (auto-calibration)',
    fields: [
      { key: 'autoCalOption', label: 'AutoCal Masking', type: 'select', options: ['Auto-Detect', 'Force Enabled (Channel)', 'Force Disabled'], tip: 'Auto-Detect applies the AutoCal channel mask automatically; Force Enabled always masks; Force Disabled never masks.' },
    ],
  },
  {
    title: 'Filtering — Finger Pressure',
    prefix: 'fp',
    fields: [
      { key: 'fpFilter', label: 'Method', type: 'select', mode: 'filter', options: ['None', 'Savitzky-Golay', 'Butterworth Low-Pass', 'Hampel'], tip: 'No filter uses the raw signal. Savitzky-Golay smooths while preserving peaks; Butterworth Low-Pass removes high-frequency noise; Hampel removes isolated spikes.' },
      { key: 'fpSavgolWin', label: 'Window Length', type: 'range', min: 5, max: 201, step: 2, showWhen: { key: 'fpFilter', value: 'Savitzky-Golay' }, tip: 'Odd number of samples used in the Savitzky-Golay smoothing window. Larger = smoother.' },
      { key: 'fpSavgolPoly', label: 'Poly Order', type: 'range', min: 1, max: 10, step: 1, showWhen: { key: 'fpFilter', value: 'Savitzky-Golay' }, tip: 'Degree of the polynomial fit inside each smoothing window.' },
      { key: 'fpButterCutoff', label: 'Cutoff', type: 'range', min: 0.5, max: 20, step: 0.5, unit: 'Hz', showWhen: { key: 'fpFilter', value: 'Butterworth Low-Pass' }, tip: 'Low-pass cutoff frequency: content above this is attenuated.' },
      { key: 'fpButterOrder', label: 'Order', type: 'range', min: 1, max: 10, step: 1, showWhen: { key: 'fpFilter', value: 'Butterworth Low-Pass' }, tip: 'Steepness of the filter roll-off. Higher = sharper, but can ring.' },
      { key: 'fpHampelWin', label: 'Window Size', type: 'range', min: 3, max: 50, step: 1, showWhen: { key: 'fpFilter', value: 'Hampel' }, tip: 'Radius of the window used to compare each sample against its neighborhood.' },
      { key: 'fpHampelSig', label: 'Sigma Threshold', type: 'range', min: 1, max: 10, step: 0.5, showWhen: { key: 'fpFilter', value: 'Hampel' }, tip: 'Number of standard deviations beyond the window median that marks a sample as a spike.' },
    ],
  },
  {
    title: 'Filtering — CBF',
    prefix: 'cbf',
    fields: [
      { key: 'cbfFilter', label: 'Method', type: 'select', mode: 'filter', options: ['None', 'Savitzky-Golay', 'Butterworth Low-Pass', 'Hampel'], tip: 'Cleans the CBF signal before analysis. Same filter options as Finger Pressure.' },
      { key: 'cbfSavgolWin', label: 'Window Length', type: 'range', min: 5, max: 201, step: 2, showWhen: { key: 'cbfFilter', value: 'Savitzky-Golay' }, tip: 'Odd number of samples used in the Savitzky-Golay smoothing window. Larger = smoother.' },
      { key: 'cbfSavgolPoly', label: 'Poly Order', type: 'range', min: 1, max: 10, step: 1, showWhen: { key: 'cbfFilter', value: 'Savitzky-Golay' }, tip: 'Degree of the polynomial fit inside each smoothing window.' },
      { key: 'cbfButterCutoff', label: 'Cutoff', type: 'range', min: 0.5, max: 20, step: 0.5, unit: 'Hz', showWhen: { key: 'cbfFilter', value: 'Butterworth Low-Pass' }, tip: 'Low-pass cutoff frequency: content above this is attenuated.' },
      { key: 'cbfButterOrder', label: 'Order', type: 'range', min: 1, max: 10, step: 1, showWhen: { key: 'cbfFilter', value: 'Butterworth Low-Pass' }, tip: 'Steepness of the filter roll-off. Higher = sharper, but can ring.' },
      { key: 'cbfHampelWin', label: 'Window Size', type: 'range', min: 3, max: 50, step: 1, showWhen: { key: 'cbfFilter', value: 'Hampel' }, tip: 'Radius of the window used to compare each sample against its neighborhood.' },
      { key: 'cbfHampelSig', label: 'Sigma Threshold', type: 'range', min: 1, max: 10, step: 0.5, showWhen: { key: 'cbfFilter', value: 'Hampel' }, tip: 'Number of standard deviations beyond the window median that marks a sample as a spike.' },
    ],
  },
  {
    title: 'Analysis',
    fields: [
      { key: 'analysisBaselineWindow', label: 'Baseline Window', type: 'number', unit: 's', tip: 'Duration of the baseline value (blue line) before the reference comment. Used by the Baseline-based method.' },
      { key: 'baselineEndComment', label: 'Baseline Ends At', type: 'select', options: ['Transition', 'Standing'], tip: 'The reference comment point used to compute the baseline value.' },
      { key: 'analysisEndMarkerWindow', label: 'End of Test Window', type: 'number', unit: 's', tip: 'How long after the end marker to sample the end value for comparison.' },
      { key: 'useMapGaussianForStats', label: 'Use MAP - Gaussian (5s) for stats', type: 'bool', tip: 'Use the Gaussian-smoothed MAP trace for minimum/maximum and endpoint statistics of Finger Pressure.' },
      { key: 'compareGaussian', label: 'Show MAP - Gaussian on chart', type: 'bool', tip: 'Overlay the Gaussian-smoothed MAP trace on the analysis chart for comparison.' },
    ],
  },
];

function FieldEditor({ field, value, onChange, indent = false }) {
  const label = (
    <label style={{ ...settingsFieldLabelStyle, ...(indent ? { paddingLeft: 16 } : {}) }}>
      {field.label}
      {field.unit ? (
        <span style={settingsUnitStyle}>{field.unit}</span>
      ) : null}
      <Tip text={field.tip} />
    </label>
  );

  if (field.type === 'bool') {
    return (
      <label style={{ ...settingsCheckboxLabelStyle, ...(indent ? { paddingLeft: 16 } : {}) }}>
        <input type="checkbox" style={settingsCheckboxStyle} checked={!!value} onChange={e => onChange(e.target.checked)} />
        {field.label}
        <Tip text={field.tip} />
      </label>
    );
  }

  if (field.type === 'select') {
    const numeric = field.options.every(o => !isNaN(Number(o)));
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, ...(indent ? { paddingLeft: 16 } : {}) }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {label}
        </div>
        <select
          value={String(value ?? field.options[0])}
          onChange={e => onChange(numeric ? Number(e.target.value) : e.target.value)}
          style={settingsSelectStyle}
        >
          {field.options.map(o => <option key={o} value={o}>{o}{field.unit ? ` ${field.unit}` : ''}</option>)}
        </select>
      </div>
    );
  }

  if (field.type === 'range') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, ...(indent ? { paddingLeft: 16 } : {}) }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          {label}
          <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-main)' }}>{value}</span>
        </div>
        <input
          type="range"
          min={field.min}
          max={field.max}
          step={field.step}
          value={value ?? field.min}
          onChange={e => onChange(field.step % 1 === 0 ? parseInt(e.target.value) : parseFloat(e.target.value))}
          style={{ width: '100%' }}
        />
      </div>
    );
  }

  // number
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, ...(indent ? { paddingLeft: 16 } : {}) }}>
      {label}
      <div style={settingsInputBoxStyle}>
        <input
          type="number"
          value={value ?? 0}
          onChange={e => onChange(parseFloat(e.target.value) || 0)}
          style={settingsInputStyle}
        />
        {field.unit ? <span style={settingsUnitStyle}>{field.unit}</span> : null}
      </div>
    </div>
  );
}

const DefaultsModal = ({ open, onClose, defaults, currentConfig, onSaveDefaults }) => {
  const [draft, setDraft] = useState({});
  const [savedFlash, setSavedFlash] = useState(false);

  useEffect(() => {
    if (open) {
      setDraft(defaults || {});
      setSavedFlash(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  if (!open) return null;

  const setKey = (key, value) => setDraft(prev => ({ ...prev, [key]: value }));

  const flash = () => {
    setSavedFlash(true);
    setTimeout(() => setSavedFlash(false), 1400);
  };

  const saveDraft = () => {
    onSaveDefaults(draft);
    flash();
  };

  const saveCurrent = () => {
    const cfg = { ...currentConfig };
    setDraft(cfg);
    onSaveDefaults(cfg);
    flash();
  };

  const resetFactory = () => {
    const factory = { ...FACTORY_DEFAULTS };
    // Preserve any keys the current config uses that aren't in the factory list.
    setDraft(factory);
    onSaveDefaults(factory);
    flash();
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 640 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px 20px', borderBottom: '1px solid var(--border-color)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <SlidersHorizontal size={16} />
            <h2 style={{ margin: 0, fontSize: 16 }}>Default Configuration</h2>
          </div>
          <button onClick={onClose} className="btn btn-secondary" style={{ padding: '4px 8px', borderRadius: 6, cursor: 'pointer', border: '1px solid var(--border-color)', background: '#fff', color: 'var(--text-main)', display: 'flex', alignItems: 'center' }}>
            <X size={14} />
          </button>
        </div>

        <p style={{ margin: '12px 20px', fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.5 }}>
          These settings are loaded automatically on the next session. Edit any value below, then press{' '}
          <b>Save Changes</b>. You can also capture your current live settings or reset to the factory defaults.
        </p>

        <div style={{ maxHeight: '50vh', overflowY: 'auto', padding: '0 20px' }}>
          {FIELD_GROUPS.map(group => (
            <div key={group.title} style={settingsGroupStyle}>
              <span style={settingsGroupTitleStyle}>{group.title}</span>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {group.fields.filter(f => !f.showWhen || draft[f.showWhen.key] === f.showWhen.value).map(field => (
                  <FieldEditor
                    key={field.key}
                    field={field}
                    value={draft[field.key]}
                    onChange={v => setKey(field.key, v)}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, padding: '16px 20px', borderTop: '1px solid var(--border-color)' }}>
          {savedFlash && (
            <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12, color: 'var(--accent-green)', fontWeight: 600 }}>
              <Check size={14} /> Saved
            </span>
          )}
          <button
            onClick={resetFactory}
            className="btn btn-secondary"
            style={{ padding: '6px 12px', fontSize: 12, borderRadius: 8, display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', background: '#fff', border: '1px solid var(--border-color)', color: 'var(--text-main)' }}
            title="Restore the built-in factory default values."
          >
            <RotateCcw size={14} /> Reset to Factory
          </button>
          <button
            onClick={saveCurrent}
            className="btn btn-secondary"
            style={{ padding: '6px 12px', fontSize: 12, borderRadius: 8, display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', background: '#fff', border: '1px solid var(--border-color)', color: 'var(--text-main)' }}
            title="Capture all currently active settings (filters, resampling, analysis) as the stored defaults."
          >
            <SlidersHorizontal size={14} /> Save Current Settings
          </button>
          <button
            onClick={saveDraft}
            className="btn btn-primary"
            style={{ padding: '6px 12px', fontSize: 12, borderRadius: 8, display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}
          >
            <Save size={14} /> Save Changes
          </button>
        </div>
      </div>
    </div>
  );
};

export default DefaultsModal;