// Shared stylesheet objects so the Filtering and Analysis screens render
// their parameters identically. Keep values in sync with index.css tokens.

export const groupLabelStyle = { fontSize: 11, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--text-muted)', userSelect: 'none' };
export const toggleBtnStyle = { padding: '6px 14px', fontSize: '12px', borderRadius: '6px', border: 'none', cursor: 'pointer', fontWeight: 600 };
export const settingsGroupStyle = { display: 'flex', flexDirection: 'column', gap: 8, padding: '8px 0', borderBottom: '1px solid rgba(0,0,0,0.04)' };
export const settingsGroupTitleStyle = { fontSize: 11, fontWeight: 700, letterSpacing: '0.04em', textTransform: 'uppercase', color: 'var(--text-muted)', userSelect: 'none' };
export const settingsFieldLabelStyle = { fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 4, cursor: 'default' };
export const settingsInputBoxStyle = { display: 'flex', alignItems: 'center', background: 'var(--bg-main)', border: '1px solid var(--border-color)', borderRadius: 6, padding: '2px 8px' };
export const settingsInputStyle = { width: 40, border: 'none', background: 'transparent', color: 'var(--text-main)', fontSize: 12, outline: 'none', textAlign: 'center' };
export const settingsUnitStyle = { fontSize: 11, color: 'var(--text-muted)', marginLeft: 4 };
export const settingsSelectStyle = { padding: '4px 8px', borderRadius: 6, border: '1px solid var(--border-color)', background: 'var(--bg-main)', color: 'var(--text-main)', fontSize: 12, outline: 'none', cursor: 'pointer' };
export const settingsCheckboxLabelStyle = { fontSize: 12, fontWeight: 600, color: 'var(--text-main)', display: 'flex', alignItems: 'center', cursor: 'pointer', gap: 6 };
export const settingsCheckboxStyle = { accentColor: 'var(--accent-blue)', width: 14, height: 14, cursor: 'pointer' };
export const settingsRangeLabelStyle = { display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', marginBottom: 4 };
export const paramRowStyle = { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 };

// Factory (built-in) defaults — mirrored from the backend's default settings payload.
export const FACTORY_DEFAULTS = {
  resampleMode: 'Beat-based',
  resampleRateTime: 1,
  resampleRateBeat: 5,
  autoCalOption: 'Auto-Detect',
  fpFilter: 'None',
  fpSavgolWin: 51,
  fpSavgolPoly: 5,
  fpButterCutoff: 5.0,
  fpButterOrder: 4,
  fpHampelWin: 5,
  fpHampelSig: 3.0,
  cbfFilter: 'None',
  cbfSavgolWin: 51,
  cbfSavgolPoly: 5,
  cbfButterCutoff: 5.0,
  cbfButterOrder: 4,
  cbfHampelWin: 5,
  cbfHampelSig: 3.0,
  analysisBaselineWindow: 30,
  baselineEndComment: 'Transition',
  analysisEndMarkerWindow: 10,
  useMapGaussianForStats: false,
  compareGaussian: false,
};