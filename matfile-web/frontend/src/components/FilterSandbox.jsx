import React from 'react';

const FilterSandbox = ({ prefix, label, settings, setSettings }) => {
  const updateSetting = (key, value) => setSettings(prev => ({ ...prev, [key]: value }));
  const methodKey = `${prefix}Filter`;
  const filterMethod = settings[methodKey];

  return (
    <div>
      <h3 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: 8, color: 'var(--text-main)' }}>{label}</h3>
      <div className="input-group">
        <select value={filterMethod} onChange={(e) => updateSetting(methodKey, e.target.value)}>
          <option>None</option>
          <option>Savitzky-Golay</option>
          <option>Butterworth Low-Pass</option>
          <option>Hampel</option>
        </select>
      </div>

      {filterMethod === 'Savitzky-Golay' && (
        <>
          <div className="input-group">
            <label className="flex-between"><span>Window Length</span><span>{settings[`${prefix}SavgolWin`]}</span></label>
            <input type="range" min="5" max="201" step="2" value={settings[`${prefix}SavgolWin`]} onChange={e => updateSetting(`${prefix}SavgolWin`, parseInt(e.target.value))} />
          </div>
          <div className="input-group">
            <label className="flex-between"><span>Poly Order</span><span>{settings[`${prefix}SavgolPoly`]}</span></label>
            <input type="range" min="1" max="10" step="1" value={settings[`${prefix}SavgolPoly`]} onChange={e => updateSetting(`${prefix}SavgolPoly`, parseInt(e.target.value))} />
          </div>
        </>
      )}

      {filterMethod === 'Butterworth Low-Pass' && (
        <>
          <div className="input-group">
            <label className="flex-between"><span>Cutoff (Hz)</span><span>{settings[`${prefix}ButterCutoff`]}</span></label>
            <input type="range" min="0.5" max="20" step="0.5" value={settings[`${prefix}ButterCutoff`]} onChange={e => updateSetting(`${prefix}ButterCutoff`, parseFloat(e.target.value))} />
          </div>
          <div className="input-group">
            <label className="flex-between"><span>Order</span><span>{settings[`${prefix}ButterOrder`]}</span></label>
            <input type="range" min="1" max="10" step="1" value={settings[`${prefix}ButterOrder`]} onChange={e => updateSetting(`${prefix}ButterOrder`, parseInt(e.target.value))} />
          </div>
        </>
      )}

      {filterMethod === 'Hampel' && (
        <>
          <div className="input-group">
            <label className="flex-between"><span>Window Size</span><span>{settings[`${prefix}HampelWin`]}</span></label>
            <input type="range" min="3" max="50" step="1" value={settings[`${prefix}HampelWin`]} onChange={e => updateSetting(`${prefix}HampelWin`, parseInt(e.target.value))} />
          </div>
          <div className="input-group">
            <label className="flex-between"><span>Sigma Threshold</span><span>{settings[`${prefix}HampelSig`]}</span></label>
            <input type="range" min="1" max="10" step="0.5" value={settings[`${prefix}HampelSig`]} onChange={e => updateSetting(`${prefix}HampelSig`, parseFloat(e.target.value))} />
          </div>
        </>
      )}
    </div>
  );
};

export default FilterSandbox;
