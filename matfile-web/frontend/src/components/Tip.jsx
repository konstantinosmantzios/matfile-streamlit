import React, { useState, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { Info } from 'lucide-react';

const SHOW_DELAY = 500;
const GAP = 8;
const MAX_W = 280;

const Tip = ({ text, children, size = 13, className = '', style, ...rest }) => {
  const ref = useRef(null);
  const timerRef = useRef(null);
  const [portalStyle, setPortalStyle] = useState({});
  const [visible, setVisible] = useState(false);

  const show = useCallback(() => {
    clearTimeout(timerRef.current);
    if (!text) return;
    timerRef.current = setTimeout(() => {
      const el = ref.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      let top = rect.bottom + GAP;
      let left = rect.left + rect.width / 2;
      let flipY = false;

      if (top + 80 > window.innerHeight) {
        top = rect.top - GAP;
        flipY = true;
      }
      if (left - MAX_W / 2 < 8) left = MAX_W / 2 + 8;
      if (left + MAX_W / 2 > window.innerWidth - 8) left = window.innerWidth - MAX_W / 2 - 8;

      setPortalStyle({
        top,
        left,
        transform: flipY ? 'translateX(-50%) translateY(-100%)' : 'translateX(-50%)',
      });
      setVisible(true);
    }, SHOW_DELAY);
  }, [text]);

  const hide = useCallback(() => {
    clearTimeout(timerRef.current);
    setVisible(false);
  }, []);

  return (
    <>
      <span
        ref={ref}
        className={`tip ${className}`.trim()}
        data-tip={text}
        aria-label={text || ''}
        tabIndex="0"
        style={style}
        {...rest}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
      >
        {children ?? <Info size={size} style={{ opacity: 0.6, flexShrink: 0 }} />}
      </span>
      {visible &&
        text &&
        createPortal(
          <span className="tip-portal" style={portalStyle}>
            {text}
          </span>,
          document.body
        )}
    </>
  );
};

export default Tip;
