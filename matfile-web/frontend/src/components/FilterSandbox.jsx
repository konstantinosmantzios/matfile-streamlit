import React from 'react';
import Tip from './Tip';
import {
  settingsGroupStyle,
  settingsGroupTitleStyle,
  settingsFieldLabelStyle,
  settingsSelectStyle,
  settingsRangeLabelStyle,
} from './settingsStyles';

const FilterSandbox = ({ prefix, label, settings, setSettings }) => {
  const updateSetting = (key, value) => setSettings(prev => ({ ...prev, [key]: value }));
  const methodKey = `${prefix}Filter`;
  const filterMethod = settings[methodKey];
  const helperText = prefix === 'fp'
    ? 'Removes movement artifacts before beat-peak detection and resampling.'
    : 'Cleans the CBF signal before analysis.';

  return (
    <div>
      <div style={settingsGroupStyle}>
        <span style={settingsGroupTitleStyle}>{label}</span>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12 }}>{helperText}</p>
        <div style={{ marginBottom: 16 }}>
          <label style={{ ...settingsFieldLabelStyle, marginBottom: 6 }}>
            Method
            <Tip text="No filter uses the raw signal. Savitzky-Golay smooths while preserving peaks; Butterworth Low-Pass removes high-frequency noise; Hampel removes isolated spike outliers." />
          </label>
          <select
            id={`${prefix}-filter`}
            value={filterMethod}
            onChange={(e) => updateSetting(methodKey, e.target.value)}
            style={settingsSelectStyle}
          >
            <option>None</option>
            <option>Savitzky-Golay</option>
            <option>Butterworth Low-Pass</option>
            <option>Hampel</option>
          </select>
        </div>

        {filterMethod === 'Savitzky-Golay' && (
          <>
            <label className="flex-between" style={{ ...settingsRangeLabelStyle }}>
              <span>Window
                <Tip text="Number of samples in the smoothing window. Must be odd. Larger window = smoother, but can flatten peaks." />
              </span>
              <span>{settings[`${prefix}SavgolWin`] || 51}</span>
            </label>
            <input
              type="range" min="5" max="201" step="2"
              value={settings[`${prefix}SavgolWin`] ?? 51}
              onChange={e => updateSetting(`${prefix}SavgolWin`, parseInt(e.target.value))}
              style={{ marginBottom: 12, width: '100%' }}
            />
            <label className="flex-between" style={{ ...settingsRangeLabelStyle }}>
              <span>Poly Order
                <Tip text="Degree of the polynomial fitted inside the window. Higher = finer detail, but more sensitive to noise." />
              </span>
              <span>{settings[`${prefix}SavgolPoly`] ?? 5}</span>
            </label>
            <input
              type="range" min="1" max="10" step="1"
              value={settings[`${prefix}SavgolPoly`] ?? 5}
              onChange={e => updateSetting(`${prefix}SavgolPoly`, parseInt(e.target.value))}
              style={{ marginBottom: 12, width: '100%' }}
            />
          </>
        )}

        {filterMethod === 'Butterworth Low-Pass' && (
          <>
            <label className="flex-between" style={{ ...settingsRangeLabelStyle }}>
              <span>Cutoff
                <Tip text="Low-pass cutoff frequency in Hz. Frequencies above this are attenuated. Lower cutoff = smoother signal." />
              </span>
              <span>{settings[`${prefix}ButterCutoff`] ?? 5.0} Hz</span>
            </label>
            <input
              type="range" min="0.5" max="20" step="0.5"
              value={settings[`${prefix}ButterCutoff`] ?? 5.0}
              onChange={e => updateSetting(`${prefix}ButterCutoff`, parseFloat(e.target.value))}
              style={{ marginBottom: 12, width: '100%' }}
            />
            <label className="flex-between" style={{ ...settingsRangeLabelStyle }}>
              <span>Order
                <Tip text="Filter roll-off steepness. Higher order = sharper cutoff, but can introduce ringing/instability." />
              </span>
              <span>{settings[`${prefix}ButterOrder`] ?? 4}</span>
            </label>
            <input
              type="range" min="1" max="10" step="1"
              value={settings[`${prefix}ButterOrder`] ?? 4}
              onChange={e => updateSetting(`${prefix}ButterOrder`, parseInt(e.target.value))}
              style={{ marginBottom: 12, width: '100%' }}
            />
          </>
        )}

        {filterMethod === 'Hampel' && (
          <>
            <label className="flex-between" style={{ ...settingsRangeLabelStyle }}>
              <span>Window
                <Tip text="Radius of the comparison window around each sample. Larger window = detects wider spikes." />
              </span>
              <span>{settings[`${prefix}HampelWin`] ?? 5}</span>
            </label>
            <input
              type="range" min="3" max="50" step="1"
              value={settings[`${prefix}HampelWin`] ?? 5}
              onChange={e => updateSetting(`${prefix}HampelWin`, parseInt(e.target.value))}
              style={{ marginBottom: 12, width: '100%' }}
            />
            <label className="flex-between" style={{ ...settingsRangeLabelStyle }}>
              <span>Sigma
                <Tip text="Number of standard deviations from the window median that marks an outlier. Lower = flags more samples as spikes." />
              </span>
              <span>{settings[`${prefix}HampelSig`] ?? 3.0}</span>
            </label>
            <input
              type="range" min="1" max="10" step="0.5"
              value={settings[`${prefix}HampelSig`] ?? 3.0}
              onChange={e => updateSetting(`${prefix}HampelSig`, parseFloat(e.target.value))}
              style={{ marginBottom: 12, width: '100%' }}
            />
          </>
        )}
      </div>
    </div>
  );
};

export default FilterSandbox;