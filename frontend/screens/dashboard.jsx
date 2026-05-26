function StatCard({ label, value, sub, icon }) {
  return (
    <Card className="p-5 relative overflow-hidden">
      <div className="absolute -top-12 -right-10 w-32 h-32 rounded-full bg-cyan-soft/5 blur-3xl"/>
      <div className="flex items-center justify-between mb-3">
        <span className="text-[11px] uppercase tracking-[0.14em] font-mono text-ink-400">{label}</span>
        <div className="w-7 h-7 rounded-md bg-white/[0.04] flex items-center justify-center text-ink-300">
          <Icon name={icon} className="w-3.5 h-3.5"/>
        </div>
      </div>
      <div className="text-[28px] font-semibold tracking-tight tabular-nums text-ink-50 leading-none">{value}</div>
      <div className="mt-2 text-[11.5px] font-mono text-ink-400">{sub}</div>
    </Card>
  );
}

function SourceLine({ s, onAsk, onDelete, busy }) {
  const isUrl = s.source_type === 'url';
  return (
    <div className="grid grid-cols-[1fr_90px_80px_120px] items-center gap-4 px-4 py-3 hover:bg-white/[0.025] transition border-b border-white/[0.04] last:border-0">
      <div className="flex items-center gap-3 min-w-0">
        <div className="w-8 h-8 rounded-md bg-white/[0.04] flex items-center justify-center text-ink-300 shrink-0">
          <Icon name={isUrl ? 'globe' : (s.source_label || '').toLowerCase().endsWith('.pdf') ? 'pdf' : 'doc'} className="w-3.5 h-3.5"/>
        </div>
        <div className="min-w-0">
          <div className="text-[13px] text-ink-100 truncate">{s.source_label}</div>
          <div className="text-[11px] text-ink-400 font-mono truncate">{s.source_id}</div>
        </div>
      </div>
      <Pill tone={isUrl ? 'cyan' : 'emerald'}>{isUrl ? 'URL' : 'File'}</Pill>
      <span className="text-[11.5px] font-mono text-ink-300 tabular-nums">{s.chunk_count} chunks</span>
      <div className="flex items-center justify-end gap-1">
        <button onClick={onAsk} className="h-7 px-2.5 rounded-md text-[12px] text-ink-300 hover:text-cyan-soft hover:bg-white/[0.04] transition inline-flex items-center gap-1.5">
          <Icon name="chat" className="w-3.5 h-3.5"/> Ask
        </button>
        <button onClick={onDelete} disabled={busy} title="Delete"
          className="h-7 w-7 rounded-md text-ink-400 hover:text-rose-300 hover:bg-white/[0.04] transition flex items-center justify-center disabled:opacity-30">
          <Icon name="close" className="w-3.5 h-3.5"/>
        </button>
      </div>
    </div>
  );
}

function Dashboard({ setRoute, stats }) {
  const [sources, setSources] = React.useState([]);
  const [loading, setLoading] = React.useState(true);
  const [busy, setBusy] = React.useState(false);

  const refresh = React.useCallback(async () => {
    try {
      setLoading(true);
      const items = await api.listSources();
      setSources(items);
    } catch (_) {
      setSources([]);
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => { refresh(); }, [refresh]);

  const remove = async (id) => {
    setBusy(true);
    try { await api.deleteSource(id); await refresh(); }
    catch (_) {}
    finally { setBusy(false); }
  };

  const pdfCount = sources.filter(s => s.source_type === 'file' && (s.source_label || '').toLowerCase().endsWith('.pdf')).length;
  const txtCount = sources.filter(s => s.source_type === 'file' && !(s.source_label || '').toLowerCase().endsWith('.pdf')).length;
  const urlCount = sources.filter(s => s.source_type === 'url').length;

  return (
    <div className="px-4 lg:px-8 py-7 max-w-[1480px] mx-auto">
      <div className="flex items-end justify-between gap-6 mb-7">
        <div>
          <div className="text-[11px] uppercase tracking-[0.16em] text-cyan-soft/80 font-mono mb-2">
            <span className="inline-flex items-center gap-1.5"><Dot color="bg-emerald-soft" className="w-1.5 h-1.5 pulse-dot"/> WebOracle · local index</span>
          </div>
          <h1 className="text-[32px] font-semibold tracking-tight text-ink-50 leading-tight">Your self-built knowledge base.</h1>
          <p className="text-ink-300 text-[14px] mt-1.5 max-w-xl">
            {stats.loading
              ? 'Connecting to backend…'
              : stats.sources === 0
                ? 'No sources indexed yet — start by adding a URL or uploading a file.'
                : <>{stats.sources} source{stats.sources === 1 ? '' : 's'} indexed across {stats.chunks} chunks. Ask anything on the chat screen.</>
            }
          </p>
        </div>
        <div className="hidden md:flex items-center gap-2">
          <Button variant="outline" icon="upload" onClick={() => setRoute('ingest')}>Ingest source</Button>
          <Button variant="cyan" icon="sparkle" onClick={() => setRoute('chat')}>Ask WebOracle</Button>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-7">
        <StatCard label="Sources" value={stats.loading ? '—' : stats.sources} sub="indexed" icon="database"/>
        <StatCard label="Chunks" value={stats.loading ? '—' : stats.chunks} sub="in ChromaDB" icon="vector"/>
        <StatCard label="URLs" value={stats.loading ? '—' : urlCount} sub="scraped pages" icon="globe"/>
        <StatCard label="Files" value={stats.loading ? '—' : stats.files} sub={`${pdfCount} PDF · ${txtCount} TXT`} icon="doc"/>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
        <Card className="xl:col-span-2 overflow-hidden">
          <div className="flex items-center justify-between px-5 py-4 border-b border-white/[0.05]">
            <div>
              <div className="text-[11px] uppercase tracking-[0.14em] font-mono text-ink-400">Library</div>
              <h3 className="text-[15.5px] font-semibold text-ink-50 mt-0.5">
                {loading ? 'Loading…' : `${sources.length} source${sources.length === 1 ? '' : 's'}`}
              </h3>
            </div>
            <div className="flex items-center gap-2">
              <Button size="sm" variant="ghost" icon="filter" onClick={refresh}>Refresh</Button>
              <Button size="sm" variant="ghost" iconRight="arrow-right" onClick={() => setRoute('ingest')}>Manage</Button>
            </div>
          </div>
          <div className="grid grid-cols-[1fr_90px_80px_120px] gap-4 px-4 py-2.5 text-[10.5px] font-mono uppercase tracking-[0.12em] text-ink-400 border-b border-white/[0.05]">
            <span>Source</span><span>Type</span><span>Chunks</span><span></span>
          </div>
          <div>
            {loading ? (
              <div className="p-6 space-y-2.5">
                <Skeleton className="h-4 w-[90%]"/>
                <Skeleton className="h-4 w-[70%]"/>
                <Skeleton className="h-4 w-[80%]"/>
              </div>
            ) : sources.length === 0 ? (
              <div className="p-10 text-center">
                <Icon name="database" className="w-7 h-7 mx-auto text-ink-400"/>
                <p className="text-[13px] text-ink-300 mt-3">No sources yet.</p>
                <Button variant="cyan" size="sm" icon="plus" className="mt-4" onClick={() => setRoute('ingest')}>
                  Add your first source
                </Button>
              </div>
            ) : (
              sources.map(s => (
                <SourceLine key={s.source_id} s={s} busy={busy}
                  onAsk={() => setRoute('chat')}
                  onDelete={() => remove(s.source_id)}/>
              ))
            )}
          </div>
        </Card>

        <div className="space-y-5">
          <Card className="p-5">
            <div className="text-[11px] uppercase tracking-[0.14em] font-mono text-ink-400 mb-2">Tech stack</div>
            <h3 className="text-[15.5px] font-semibold text-ink-50">What powers WebOracle</h3>
            <ul className="mt-3 space-y-2 text-[12.5px]">
              <li className="flex items-center justify-between">
                <span className="text-ink-200">Scraping</span>
                <span className="font-mono text-cyan-soft">Playwright</span>
              </li>
              <li className="flex items-center justify-between">
                <span className="text-ink-200">Embeddings</span>
                <span className="font-mono text-cyan-soft">jina-embeddings-v3</span>
              </li>
              <li className="flex items-center justify-between">
                <span className="text-ink-200">Vector store</span>
                <span className="font-mono text-cyan-soft">ChromaDB</span>
              </li>
              <li className="flex items-center justify-between">
                <span className="text-ink-200">LLM</span>
                <span className="font-mono text-cyan-soft">llama-3.3-70b · Groq</span>
              </li>
              <li className="flex items-center justify-between">
                <span className="text-ink-200">API</span>
                <span className="font-mono text-cyan-soft">FastAPI</span>
              </li>
            </ul>
          </Card>

          <Card className="p-5 relative overflow-hidden">
            <div className="absolute -bottom-10 -left-10 w-40 h-40 rounded-full bg-emerald-soft/10 blur-3xl"/>
            <div className="text-[11px] uppercase tracking-[0.14em] font-mono text-cyan-soft/80 mb-2">Workflow</div>
            <ol className="space-y-2 text-[12.5px] text-ink-200 list-decimal list-inside">
              <li><span className="font-medium text-ink-50">Ingest</span> a URL or file</li>
              <li>It's parsed, chunked, embedded, stored</li>
              <li>Go to <span className="font-medium text-ink-50">Ask</span> and query it</li>
              <li>Groq grounds the answer in your sources</li>
            </ol>
            <Button size="sm" variant="cyan" icon="upload" className="mt-4" onClick={() => setRoute('ingest')}>Start ingesting</Button>
          </Card>
        </div>
      </div>
    </div>
  );
}

window.Dashboard = Dashboard;
