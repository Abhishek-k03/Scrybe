const { useState, useEffect } = React;

const NAV = [
  { id: 'dashboard', label: 'Dashboard', icon: 'home' },
  { id: 'chat',      label: 'Ask',       icon: 'chat' },
  { id: 'ingest',    label: 'Ingest',    icon: 'upload' },
  { id: 'retrieval', label: 'Retrieval', icon: 'vector' },
  { id: 'settings',  label: 'Settings',  icon: 'settings' },
];

function useStats(refreshKey) {
  const [stats, setStats] = useState({ sources: 0, chunks: 0, urls: 0, files: 0, loading: true, error: null });
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const items = await api.listSources();
        if (cancelled) return;
        setStats({
          sources: items.length,
          chunks: items.reduce((a, s) => a + (s.chunk_count || 0), 0),
          urls: items.filter(s => s.source_type === 'url').length,
          files: items.filter(s => s.source_type === 'file').length,
          loading: false,
          error: null,
        });
      } catch (e) {
        if (cancelled) return;
        setStats(s => ({ ...s, loading: false, error: e.message }));
      }
    })();
    return () => { cancelled = true; };
  }, [refreshKey]);
  return stats;
}

function useHealth() {
  const [health, setHealth] = useState({ ok: false, version: null });
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const h = await api.health();
        if (alive) setHealth({ ok: h.status === 'ok', version: h.version });
      } catch {
        if (alive) setHealth({ ok: false, version: null });
      }
    };
    tick();
    const id = setInterval(tick, 15000);
    return () => { alive = false; clearInterval(id); };
  }, []);
  return health;
}

function CommandTriggerBar({ onClick }) {
  return (
    <button
      onClick={onClick}
      className="hidden md:flex items-center gap-2 h-9 w-[420px] px-3 rounded-lg glass hover:bg-white/[0.04] transition text-left"
    >
      <Icon name="search" className="w-4 h-4 text-ink-300"/>
      <span className="text-ink-300 text-[13px] flex-1">Jump to a section…</span>
      <span className="flex items-center gap-1">
        <Kbd>⌘</Kbd><Kbd>K</Kbd>
      </span>
    </button>
  );
}

function Sidebar({ route, setRoute, stats }) {
  return (
    <aside className="hidden lg:flex flex-col w-[260px] shrink-0 h-screen sticky top-0 border-r border-white/[0.06] bg-ink-950/40">
      <div className="h-14 px-5 flex items-center border-b border-white/[0.06]">
        <Logo/>
      </div>

      <nav className="px-3 pt-4 space-y-0.5">
        {NAV.map(item => {
          const active = route === item.id;
          return (
            <button key={item.id}
              onClick={() => setRoute(item.id)}
              className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg transition group text-[13px]
                ${active
                  ? 'bg-white/[0.05] text-ink-50 shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06)]'
                  : 'text-ink-300 hover:text-ink-100 hover:bg-white/[0.03]'}`}
            >
              <Icon name={item.icon} className={`w-4 h-4 ${active ? 'text-cyan-soft' : 'text-ink-300 group-hover:text-ink-100'}`}/>
              <span className="flex-1 text-left">{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="mt-auto p-3 space-y-3">
        <div className="glass rounded-xl p-3.5 relative overflow-hidden">
          <div className="absolute -top-8 -right-8 w-20 h-20 rounded-full bg-cyan-soft/15 blur-2xl"/>
          <div className="text-[10.5px] font-mono uppercase tracking-[0.14em] text-ink-400 mb-2">Index</div>
          <div className="flex items-baseline gap-2">
            <span className="text-[20px] font-semibold tabular-nums tracking-tight">{stats.loading ? '—' : stats.chunks}</span>
            <span className="text-[11px] text-ink-400 font-mono">chunks</span>
          </div>
          <div className="text-[11px] font-mono text-ink-400 mt-1">
            {stats.loading ? 'loading…' : `${stats.sources} source${stats.sources === 1 ? '' : 's'} · ${stats.urls} URL · ${stats.files} file`}
          </div>
        </div>
      </div>
    </aside>
  );
}

function MobileNav({ route, setRoute, open, setOpen }) {
  return (
    <div className={`lg:hidden fixed inset-0 z-40 transition ${open ? 'pointer-events-auto' : 'pointer-events-none'}`}>
      <div className={`absolute inset-0 bg-black/60 backdrop-blur-sm transition-opacity ${open ? 'opacity-100' : 'opacity-0'}`}
           onClick={() => setOpen(false)} />
      <aside className={`absolute left-0 top-0 bottom-0 w-[280px] glass border-r border-white/10 p-4 transition-transform ${open ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="flex items-center justify-between mb-6">
          <Logo/>
          <button onClick={() => setOpen(false)} className="text-ink-300 hover:text-ink-50">
            <Icon name="close" className="w-5 h-5"/>
          </button>
        </div>
        <nav className="space-y-0.5">
          {NAV.map(item => (
            <button key={item.id}
              onClick={() => { setRoute(item.id); setOpen(false); }}
              className={`w-full flex items-center gap-2.5 px-2.5 py-2.5 rounded-lg transition text-[14px]
                ${route === item.id ? 'bg-white/[0.05] text-ink-50' : 'text-ink-300'}`}>
              <Icon name={item.icon} className={`w-4 h-4 ${route === item.id ? 'text-cyan-soft' : ''}`}/>
              {item.label}
            </button>
          ))}
        </nav>
      </aside>
    </div>
  );
}

function Topbar({ openCmd, openMobile, route, health, setRoute }) {
  return (
    <header className="sticky top-0 z-30 h-14 px-4 lg:px-7 flex items-center justify-between gap-3 border-b border-white/[0.05] bg-ink-950/70 backdrop-blur-xl">
      <div className="flex items-center gap-3">
        <button onClick={openMobile} className="lg:hidden text-ink-200 hover:text-ink-50">
          <Icon name="menu" className="w-5 h-5"/>
        </button>
        <div className="hidden md:flex items-center gap-2 text-[12.5px] text-ink-300 font-mono">
          <span className="text-ink-400">weboracle</span>
          <Icon name="chevron-right" className="w-3 h-3 text-ink-500"/>
          <span className="text-ink-100 capitalize">{route}</span>
        </div>
      </div>

      <CommandTriggerBar onClick={openCmd}/>

      <div className="flex items-center gap-2">
        <div className="hidden sm:flex items-center gap-1.5 h-9 px-2.5 rounded-lg glass" title={health.ok ? `Backend healthy · v${health.version}` : 'Backend unreachable'}>
          <span className={`w-1.5 h-1.5 rounded-full ${health.ok ? 'bg-emerald-soft pulse-dot' : 'bg-rose-400'}`}/>
          <span className="text-[11.5px] text-ink-200 font-mono">{health.ok ? 'connected' : 'offline'}</span>
        </div>
        <Button variant="cyan" size="md" icon="sparkle" onClick={() => setRoute('chat')}>New query</Button>
      </div>
    </header>
  );
}

function CommandPalette({ open, onClose, setRoute }) {
  const [q, setQ] = useState('');
  const items = [
    { kind: 'Action', label: 'Open Dashboard',        icon: 'home',     go: () => setRoute('dashboard') },
    { kind: 'Action', label: 'New chat',              icon: 'chat',     go: () => setRoute('chat') },
    { kind: 'Action', label: 'Ingest a URL or file',  icon: 'upload',   go: () => setRoute('ingest') },
    { kind: 'Action', label: 'Open Retrieval inspector', icon: 'vector', go: () => setRoute('retrieval') },
    { kind: 'Action', label: 'Settings',              icon: 'settings', go: () => setRoute('settings') },
  ];
  const filtered = items.filter(i => i.label.toLowerCase().includes(q.toLowerCase()));
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[10vh] px-4">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-md" onClick={onClose}/>
      <div className="relative w-full max-w-[640px] glass rounded-2xl overflow-hidden border-white/10 fade-up">
        <div className="flex items-center gap-3 px-4 h-14 border-b border-white/[0.07]">
          <Icon name="search" className="w-4 h-4 text-ink-300"/>
          <input autoFocus value={q} onChange={e=>setQ(e.target.value)}
            placeholder="Jump to a section…"
            className="flex-1 bg-transparent outline-none text-[14px] placeholder:text-ink-400 ring-cyan"/>
        </div>
        <div className="max-h-[55vh] overflow-y-auto py-2">
          {filtered.length === 0 ? (
            <div className="px-4 py-8 text-center text-ink-400 text-[13px]">No matches.</div>
          ) : filtered.map((it, i) => (
            <button key={i}
              onClick={() => { it.go && it.go(); onClose(); }}
              className="w-full flex items-center gap-3 px-4 py-2.5 hover:bg-white/[0.04] transition text-left">
              <div className="w-7 h-7 rounded-md bg-white/[0.04] ring-1 ring-white/[0.06] flex items-center justify-center text-ink-200">
                <Icon name={it.icon} className="w-3.5 h-3.5"/>
              </div>
              <div className="flex-1">
                <div className="text-[13px] text-ink-100">{it.label}</div>
              </div>
              <span className="text-[10.5px] font-mono uppercase tracking-wider text-ink-400">{it.kind}</span>
            </button>
          ))}
        </div>
        <div className="flex items-center justify-between px-4 h-10 border-t border-white/[0.07] text-[11px] text-ink-400 font-mono">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1.5"><Kbd>↵</Kbd> open</span>
          </div>
          <span className="flex items-center gap-1.5"><Kbd>esc</Kbd> close</span>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [route, setRoute] = useState('dashboard');
  const [cmdOpen, setCmdOpen] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const stats = useStats(route);
  const health = useHealth();

  useEffect(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault(); setCmdOpen(o => !o);
      } else if (e.key === 'Escape') setCmdOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  let screen = null;
  switch (route) {
    case 'dashboard': screen = <Dashboard setRoute={setRoute} stats={stats}/>; break;
    case 'chat':      screen = <Chat/>; break;
    case 'ingest':    screen = <Ingest/>; break;
    case 'retrieval': screen = <Retrieval/>; break;
    case 'settings':  screen = <Settings health={health} stats={stats}/>; break;
    default:          screen = <Dashboard setRoute={setRoute} stats={stats}/>;
  }

  return (
    <div className="min-h-screen flex grid-backdrop">
      <Sidebar route={route} setRoute={setRoute} stats={stats}/>
      <MobileNav route={route} setRoute={setRoute} open={mobileOpen} setOpen={setMobileOpen}/>
      <div className="flex-1 min-w-0 flex flex-col">
        <Topbar
          openCmd={() => setCmdOpen(true)}
          openMobile={() => setMobileOpen(true)}
          route={route}
          health={health}
          setRoute={setRoute}/>
        <main className="flex-1 min-w-0" data-screen-label={`WebOracle · ${route}`}>
          <div key={route} className="fade-up">
            {screen}
          </div>
        </main>
      </div>
      <CommandPalette open={cmdOpen} onClose={() => setCmdOpen(false)} setRoute={setRoute}/>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App/>);
