function Icon({ name, className = "w-4 h-4", strokeWidth = 1.6 }) {
  const common = {
    viewBox: "0 0 24 24", fill: "none", stroke: "currentColor",
    strokeWidth, strokeLinecap: "round", strokeLinejoin: "round",
    className,
  };
  switch (name) {
    case 'logo': return (
      <svg {...common}>
        <circle cx="12" cy="12" r="3" />
        <path d="M5 12a7 7 0 0 1 11.5-5.4" />
        <path d="M19 12a7 7 0 0 1-11.5 5.4" />
        <circle cx="19" cy="12" r="1" fill="currentColor"/>
        <circle cx="5"  cy="12" r="1" fill="currentColor"/>
      </svg>
    );
    case 'home': return (<svg {...common}><path d="M3 11l9-7 9 7"/><path d="M5 10v10h14V10"/></svg>);
    case 'chat': return (<svg {...common}><path d="M21 12a8 8 0 1 1-3.2-6.4L21 4l-1.4 3.4A8 8 0 0 1 21 12z"/><path d="M8 12h.01M12 12h.01M16 12h.01"/></svg>);
    case 'upload': return (<svg {...common}><path d="M12 16V4"/><path d="M7 9l5-5 5 5"/><path d="M5 20h14"/></svg>);
    case 'vector': return (<svg {...common}><circle cx="5" cy="6" r="1.6"/><circle cx="19" cy="6" r="1.6"/><circle cx="5" cy="18" r="1.6"/><circle cx="19" cy="18" r="1.6"/><circle cx="12" cy="12" r="1.6"/><path d="M5 6l7 6M19 6l-7 6M5 18l7-6M19 18l-7-6"/></svg>);
    case 'settings': return (<svg {...common}><circle cx="12" cy="12" r="2.6"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 0 1-4 0v-.1a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 0 1 0-4h.1a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3 1.7 1.7 0 0 0 1-1.5V3a2 2 0 0 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8 1.7 1.7 0 0 0 1.5 1H21a2 2 0 0 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>);
    case 'search': return (<svg {...common}><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg>);
    case 'sparkle': return (<svg {...common}><path d="M12 3l1.6 4.6L18 9l-4.4 1.4L12 15l-1.6-4.6L6 9l4.4-1.4L12 3z"/><path d="M19 15l.7 1.8L21 17.5l-1.3.7L19 20l-.7-1.8L17 17.5l1.3-.7L19 15z"/></svg>);
    case 'send': return (<svg {...common}><path d="M4 12l16-8-6 18-3-7-7-3z"/></svg>);
    case 'doc': return (<svg {...common}><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/><path d="M8 13h8M8 17h5"/></svg>);
    case 'pdf': return (<svg {...common}><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/><path d="M9 14h1.2a1.3 1.3 0 0 1 0 2.6H9V14zM13 14v3M13 14h1.5M13 15.5h1.2M16.5 14h2.2M16.5 14v3M16.5 15.5h1.6"/></svg>);
    case 'globe': return (<svg {...common}><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/></svg>);
    case 'code': return (<svg {...common}><path d="M9 18l-6-6 6-6M15 6l6 6-6 6"/></svg>);
    case 'database': return (<svg {...common}><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></svg>);
    case 'plus': return (<svg {...common}><path d="M12 5v14M5 12h14"/></svg>);
    case 'check': return (<svg {...common}><path d="M5 12.5l4.5 4.5L19 7"/></svg>);
    case 'chevron-right': return (<svg {...common}><path d="M9 6l6 6-6 6"/></svg>);
    case 'chevron-down': return (<svg {...common}><path d="M6 9l6 6 6-6"/></svg>);
    case 'arrow-right': return (<svg {...common}><path d="M5 12h14M13 6l6 6-6 6"/></svg>);
    case 'arrow-up-right': return (<svg {...common}><path d="M7 17L17 7M9 7h8v8"/></svg>);
    case 'copy': return (<svg {...common}><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>);
    case 'thumb-up': return (<svg {...common}><path d="M7 11v9H4V11zM7 11l4-7a2 2 0 0 1 3.4 2L13 9h5.2a2 2 0 0 1 2 2.4l-1.4 7A2 2 0 0 1 16.8 20H7"/></svg>);
    case 'thumb-down': return (<svg {...common} className={(className||'')+' rotate-180'}><path d="M7 11v9H4V11zM7 11l4-7a2 2 0 0 1 3.4 2L13 9h5.2a2 2 0 0 1 2 2.4l-1.4 7A2 2 0 0 1 16.8 20H7"/></svg>);
    case 'menu': return (<svg {...common}><path d="M4 6h16M4 12h16M4 18h16"/></svg>);
    case 'close': return (<svg {...common}><path d="M6 6l12 12M18 6L6 18"/></svg>);
    case 'bolt': return (<svg {...common}><path d="M13 2L4 14h7l-1 8 9-12h-7l1-8z"/></svg>);
    case 'shield': return (<svg {...common}><path d="M12 3l8 3v6c0 5-3.5 8.5-8 9-4.5-.5-8-4-8-9V6l8-3z"/></svg>);
    case 'cpu': return (<svg {...common}><rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/></svg>);
    case 'link': return (<svg {...common}><path d="M10 14a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-1 1"/><path d="M14 10a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l1-1"/></svg>);
    case 'filter': return (<svg {...common}><path d="M3 5h18l-7 9v6l-4-2v-4z"/></svg>);
    case 'play': return (<svg {...common}><path d="M6 4l14 8-14 8z" fill="currentColor"/></svg>);
    case 'pause': return (<svg {...common}><rect x="6" y="5" width="4" height="14" rx="1" fill="currentColor"/><rect x="14" y="5" width="4" height="14" rx="1" fill="currentColor"/></svg>);
    case 'github': return (<svg viewBox="0 0 24 24" fill="currentColor" className={className}><path d="M12 .5C5.7.5.5 5.7.5 12c0 5.1 3.3 9.4 7.8 10.9.6.1.8-.3.8-.6v-2c-3.2.7-3.9-1.5-3.9-1.5-.5-1.3-1.3-1.7-1.3-1.7-1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1 1.8 2.8 1.3 3.5 1 .1-.8.4-1.3.7-1.6-2.6-.3-5.3-1.3-5.3-5.8 0-1.3.5-2.3 1.2-3.1-.1-.3-.5-1.5.1-3.1 0 0 1-.3 3.2 1.2.9-.3 1.9-.4 2.9-.4s2 .1 2.9.4c2.2-1.5 3.2-1.2 3.2-1.2.6 1.6.2 2.8.1 3.1.7.8 1.2 1.8 1.2 3.1 0 4.5-2.7 5.5-5.3 5.8.4.3.8 1 .8 2v2.9c0 .3.2.7.8.6 4.5-1.5 7.8-5.8 7.8-10.9C23.5 5.7 18.3.5 12 .5z"/></svg>);
    case 'discord': return (<svg viewBox="0 0 24 24" fill="currentColor" className={className}><path d="M20 4.5A18 18 0 0 0 16 3.3l-.2.4a16 16 0 0 0-5.6 0L10 3.3A18 18 0 0 0 6 4.5C2.6 9.5 1.7 14.4 2.2 19.2c1.7 1.3 3.4 2.1 5 2.6l.5-.7c-.8-.3-1.6-.7-2.3-1.2.2-.1.4-.2.5-.4 4 1.9 8.4 1.9 12.3 0 .2.1.3.3.5.4-.7.5-1.5.9-2.3 1.2l.5.7c1.7-.5 3.3-1.3 5-2.6.5-5.5-.8-10.4-3.4-14.7zM9 16.4c-1 0-1.9-1-1.9-2.2 0-1.2.9-2.2 1.9-2.2 1.1 0 1.9 1 1.9 2.2 0 1.2-.9 2.2-1.9 2.2zm6 0c-1 0-1.9-1-1.9-2.2 0-1.2.9-2.2 1.9-2.2 1.1 0 1.9 1 1.9 2.2 0 1.2-.9 2.2-1.9 2.2z"/></svg>);
    case 'dots': return (<svg {...common}><circle cx="5" cy="12" r="1.4" fill="currentColor"/><circle cx="12" cy="12" r="1.4" fill="currentColor"/><circle cx="19" cy="12" r="1.4" fill="currentColor"/></svg>);
    case 'spinner': return (<svg viewBox="0 0 24 24" className={className}><circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" opacity="0.2" fill="none"/><path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" fill="none"><animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="0.9s" repeatCount="indefinite"/></path></svg>);
    case 'kbd': return (<svg {...common}><rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 10h.01M11 10h.01M15 10h.01M7 14h10"/></svg>);
    case 'book': return (<svg {...common}><path d="M4 4h7a3 3 0 0 1 3 3v13a2 2 0 0 0-2-2H4z"/><path d="M20 4h-7a3 3 0 0 0-3 3v13a2 2 0 0 1 2-2h8z"/></svg>);
    case 'flame': return (<svg {...common}><path d="M12 2s4 4 4 8a4 4 0 0 1-8 0c0-1.4.6-2.5.6-2.5C8 9 6 11 6 14a6 6 0 0 0 12 0c0-5-6-7-6-12z"/></svg>);
    default: return null;
  }
}

function Logo({ size = 28 }) {
  return (
    <div className="flex items-center gap-2.5">
      <div className="relative" style={{width: size, height: size}}>
        <svg viewBox="0 0 32 32" width={size} height={size}>
          <defs>
            <linearGradient id="lg1" x1="0" x2="1" y1="0" y2="1">
              <stop offset="0" stopColor="#67e8f9"/>
              <stop offset="1" stopColor="#34d399"/>
            </linearGradient>
          </defs>
          <circle cx="16" cy="16" r="3.2" fill="url(#lg1)" />
          <path d="M5 16 C 8 8, 24 8, 27 16" stroke="url(#lg1)" strokeWidth="1.4" fill="none" strokeLinecap="round"/>
          <path d="M27 16 C 24 24, 8 24, 5 16" stroke="url(#lg1)" strokeWidth="1.4" fill="none" strokeLinecap="round" opacity="0.7"/>
          <circle cx="27" cy="16" r="1.6" fill="#67e8f9" />
          <circle cx="5"  cy="16" r="1.6" fill="#34d399" />
        </svg>
      </div>
      <div className="flex flex-col leading-none">
        <span className="text-[15px] font-semibold tracking-tight text-ink-50">Scrybe</span>
      </div>
    </div>
  );
}

function Button({ children, variant = 'default', size = 'md', className = '', icon, iconRight, ...props }) {
  const base = 'inline-flex items-center justify-center gap-2 font-medium rounded-lg transition-all duration-200 select-none disabled:opacity-50';
  const sizes = {
    sm: 'h-8 px-3 text-[12.5px]',
    md: 'h-9 px-3.5 text-[13px]',
    lg: 'h-11 px-5 text-[14px]',
  };
  const variants = {
    default: 'bg-ink-50 text-ink-950 hover:bg-white',
    ghost: 'text-ink-200 hover:text-ink-50 hover:bg-white/5',
    outline: 'text-ink-100 border border-white/10 hover:bg-white/5 hover:border-white/20',
    glass: 'glass text-ink-100 hover:bg-white/[0.06]',
    cyan: 'bg-cyan-soft/10 text-cyan-soft border border-cyan-soft/25 hover:bg-cyan-soft/15 hover:border-cyan-soft/40',
    danger: 'bg-rose-500/10 text-rose-300 border border-rose-500/25 hover:bg-rose-500/15',
  };
  return (
    <button className={`${base} ${sizes[size]} ${variants[variant]} ${className}`} {...props}>
      {icon ? <Icon name={icon} className="w-4 h-4" /> : null}
      {children}
      {iconRight ? <Icon name={iconRight} className="w-4 h-4" /> : null}
    </button>
  );
}

function Card({ children, className = '', as: As = 'div', ...rest }) {
  return (
    <As className={`glass rounded-2xl ring-soft ${className}`} {...rest}>{children}</As>
  );
}

function Pill({ children, tone = 'neutral', icon, className = '' }) {
  const tones = {
    neutral: 'bg-white/[0.04] text-ink-200 border-white/10',
    cyan: 'bg-cyan-soft/8 text-cyan-soft border-cyan-soft/20',
    emerald: 'bg-emerald-soft/8 text-emerald-soft border-emerald-soft/20',
    amber: 'bg-amber-300/8 text-amber-300 border-amber-300/20',
    rose: 'bg-rose-400/8 text-rose-300 border-rose-400/20',
  };
  return (
    <span className={`inline-flex items-center gap-1.5 text-[11px] font-medium px-2 py-0.5 rounded-full border ${tones[tone]} ${className}`}>
      {icon ? <Icon name={icon} className="w-3 h-3"/> : null}
      {children}
    </span>
  );
}

function SectionHeader({ eyebrow, title, subtitle, right }) {
  return (
    <div className="flex items-end justify-between gap-6 mb-5">
      <div>
        {eyebrow ? <div className="text-[11px] uppercase tracking-[0.16em] text-cyan-soft/80 font-mono mb-2">{eyebrow}</div> : null}
        <h2 className="text-[22px] font-semibold tracking-tight text-ink-50">{title}</h2>
        {subtitle ? <p className="text-ink-300 text-[13.5px] mt-1 max-w-xl">{subtitle}</p> : null}
      </div>
      {right}
    </div>
  );
}

function Kbd({ children }) {
  return (
    <kbd className="inline-flex items-center justify-center min-w-[20px] h-[20px] px-1.5 rounded-md bg-white/[0.04] border border-white/10 text-[10.5px] text-ink-300 font-mono">{children}</kbd>
  );
}

function Skeleton({ className = '' }) {
  return <div className={`skeleton rounded-md ${className}`} />;
}

function Dot({ className = 'w-1.5 h-1.5', color = 'bg-cyan-soft' }) {
  return <span className={`inline-block rounded-full ${color} ${className}`} />;
}

function Toast({ tone, children, onClose }) {
  if (!children) return null;
  const tones = {
    ok: 'border-emerald-soft/30 bg-emerald-soft/[0.06] text-emerald-soft',
    err: 'border-rose-400/30 bg-rose-400/[0.06] text-rose-300',
    info: 'border-cyan-soft/30 bg-cyan-soft/[0.06] text-cyan-soft',
  };
  return (
    <div className={`fixed bottom-20 right-6 z-40 max-w-md px-4 py-3 rounded-xl border glass ${tones[tone] || tones.info} fade-up`}>
      <div className="flex items-start gap-3">
        <span className="text-[13px] flex-1">{children}</span>
        <button onClick={onClose} className="text-ink-300 hover:text-ink-50"><Icon name="close" className="w-3.5 h-3.5"/></button>
      </div>
    </div>
  );
}

Object.assign(window, { Icon, Logo, Button, Card, Pill, SectionHeader, Kbd, Skeleton, Dot, Toast });
