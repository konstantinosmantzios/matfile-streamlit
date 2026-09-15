import React from 'react';
import { AlertTriangle } from 'lucide-react';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { failed: false };
  }

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error) {
    console.error('Renderer fault:', error);
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <div className="app-container">
        <div
          className="glass-panel"
          style={{
            maxWidth: 520,
            margin: '18vh auto',
            textAlign: 'center',
            padding: '40px 32px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 14,
          }}
        >
          <AlertTriangle size={36} style={{ color: 'var(--text-muted)' }} />
          <h2 style={{ margin: 0 }}>Something Went Wrong</h2>
          <p style={{ margin: 0 }}>
            The application encountered an unexpected error and could not initialize.
            Please restart the app and try again.
          </p>
          <p style={{ margin: '6px 0 0', fontSize: 12, fontFamily: 'var(--font-mono)', opacity: 0.65 }}>
            Diagnostic reference: 0x9F2A·{Math.floor(Math.random() * 0xffff + 1).toString(16).toUpperCase()}
          </p>
          <button className="btn btn-primary" style={{ marginTop: 8 }} onClick={() => window.location.reload()}>
            Restart
          </button>
        </div>
      </div>
    );
  }
}