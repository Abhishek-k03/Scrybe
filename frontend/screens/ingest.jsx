const SOURCE_TYPES = [
  { id: 'upload', label: 'Files',         icon: 'upload', desc: 'PDF or TXT, up to ~20 MB' },
  { id: 'web',    label: 'Web URL',       icon: 'globe',  desc: 'Scrape a single page (Playwright)' },
  { id: 'github', label: 'GitHub',        icon: 'github', desc: 'Not in this build' },
  { id: 'api',    label: 'API connector', icon: 'link',   desc: 'Not in this build' },
];

function SourceTab({ s, active, disabled, onClick }) {
  return (
    <button onClick={onClick} disabled={disabled}
      className={`flex items-start gap-3 p-3.5 rounded-xl text-left transition border
        ${disabled ? 'opacity-40 cursor-not-allowed bg-white/[0.015] border-white/[0.04]'
          : active ? 'bg-cyan-soft/[0.06] border-cyan-soft/30 shadow-glow'
          : 'bg-white/[0.02] border-white/[0.06] hover:bg-white/[0.04] hover:border-white/[0.1]'}`}>
      <div className={`w-9 h-9 rounded-lg flex items-center justify-center
        ${active && !disabled ? 'bg-cyan-soft/15 text-cyan-soft' : 'bg-white/[0.04] text-ink-200'}`}>
        <Icon name={s.icon} className="w-4 h-4"/>
      </div>
      <div className="min-w-0">
        <div className="text-[13px] text-ink-50 font-medium">{s.label}</div>
        <div className="text-[11.5px] text-ink-400 font-mono mt-0.5">{s.desc}</div>
      </div>
    </button>
  );
}

function Dropzone({ busy, onPickFiles, onAddUrl, urlValue, setUrlValue }) {
  const [drag, setDrag] = React.useState(false);
  const fileInput = React.useRef(null);
  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); onPickFiles(Array.from(e.dataTransfer.files || [])); }}
      className={`relative rounded-2xl overflow-hidden border border-dashed transition-colors
        ${drag ? 'border-cyan-soft/60 bg-cyan-soft/[0.05]' : 'border-white/[0.12] bg-white/[0.01]'}`}>
      <div className="absolute inset-0 dotgrid opacity-50"/>
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-soft/40 to-transparent"/>
      <input ref={fileInput} type="file" accept=".pdf,.txt" multiple hidden
        onChange={(e) => onPickFiles(Array.from(e.target.files || []))}/>
      <div className="relative px-8 py-12 flex flex-col items-center text-center">
        <div className="relative w-16 h-16 mb-4">
          <div className="absolute inset-0 rounded-full bg-cyan-soft/15 blur-xl"/>
          <div className="relative w-16 h-16 rounded-full bg-gradient-to-b from-white/[0.06] to-white/[0.02] border border-white/[0.08] flex items-center justify-center">
            <Icon name={busy ? 'spinner' : 'upload'} className="w-6 h-6 text-cyan-soft"/>
          </div>
        </div>
        <h3 className="text-[18px] font-semibold text-ink-50 tracking-tight">Drop a PDF or TXT to ingest</h3>
        <p className="text-ink-300 text-[13px] mt-1.5 max-w-md">
          Files are parsed, chunked (~800 tokens, 150 overlap), embedded with <span className="font-mono text-cyan-soft">jina-embeddings-v3</span>, and stored in ChromaDB.
        </p>
        <div className="mt-5 flex items-center gap-2">
          <Button variant="cyan" icon="plus" disabled={busy} onClick={() => fileInput.current?.click()}>
            Choose files
          </Button>
        </div>
        <div className="mt-6 w-full max-w-xl">
          <div className="flex items-center gap-2 h-11 px-3.5 rounded-xl glass">
            <Icon name="globe" className="w-4 h-4 text-cyan-soft"/>
            <input
              className="flex-1 bg-transparent outline-none text-[13.5px] text-ink-100 placeholder:text-ink-400"
              placeholder="https://example.com/article"
              value={urlValue}
              onChange={(e) => setUrlValue(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && urlValue.trim()) onAddUrl(); }}
              disabled={busy}/>
            <Button size="sm" variant="cyan" icon={busy ? 'spinner' : 'bolt'}
              disabled={busy || !urlValue.trim()} onClick={onAddUrl}>
              Add URL
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

function PipelineDiagram() {
  const steps = [
    { k: 'Scrape/Parse', d: 'Playwright · pypdf', icon: 'doc',     active: true },
    { k: 'Chunk',        d: '~800 tok · 150 ol',   icon: 'filter',  active: true },
    { k: 'Embed',        d: 'jina-embeddings-v3',  icon: 'sparkle', active: true },
    { k: 'Index',        d: 'ChromaDB · cosine',   icon: 'database',active: true },
  ];
  return (
    <Card className="p-5">
      <div className="flex items-center justify-between mb-5">
        <div>
          <div className="text-[11px] uppercase tracking-[0.14em] font-mono text-ink-400">Pipeline</div>
          <h3 className="text-[15.5px] font-semibold text-ink-50 mt-0.5">How a source flows in</h3>
        </div>
        <Pill tone="emerald"><Dot color="bg-emerald-soft" className="w-1 h-1 pulse-dot"/> live</Pill>
      </div>
      <div className="relative flex items-stretch gap-2 overflow-x-auto pb-2">
        {steps.map((s, i) => (
          <React.Fragment key={s.k}>
            <div className="flex-1 min-w-[120px] p-3 rounded-xl border border-cyan-soft/25 bg-cyan-soft/[0.04] transition">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-6 h-6 rounded-md flex items-center justify-center bg-cyan-soft/15 text-cyan-soft">
                  <Icon name={s.icon} className="w-3 h-3"/>
                </div>
                <div className="text-[12.5px] text-ink-100 font-medium">{s.k}</div>
              </div>
              <div className="text-[10.5px] font-mono text-ink-400">{s.d}</div>
            </div>
            {i < steps.length - 1 ? (
              <div className="flex items-center text-ink-500">
                <Icon name="chevron-right" className="w-3.5 h-3.5"/>
              </div>
            ) : null}
          </React.Fragment>
        ))}
      </div>
    </Card>
  );
}

function SourceRow({ s, busy, onDelete }) {
  const isUrl = s.source_type === 'url';
  return (
    <div className="grid grid-cols-[1.5fr_90px_140px_80px_36px] items-center gap-4 px-4 py-3 hover:bg-white/[0.025] transition border-b border-white/[0.04] last:border-0">
      <div className="flex items-center gap-3 min-w-0">
        <div className="w-8 h-8 rounded-md bg-white/[0.04] flex items-center justify-center text-ink-300 shrink-0">
          <Icon name={isUrl ? 'globe' : (s.source_label || '').toLowerCase().endsWith('.pdf') ? 'pdf' : 'doc'} className="w-3.5 h-3.5"/>
        </div>
        <div className="min-w-0">
          <div className="text-[13px] text-ink-100 truncate">{s.source_label}</div>
          <div className="text-[11px] font-mono text-ink-400 truncate">{s.source_id}</div>
        </div>
      </div>
      <span className="text-[11.5px] font-mono text-ink-300 uppercase">{s.source_type}</span>
      <Pill tone="emerald">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-soft"/>
        Indexed
      </Pill>
      <span className="text-[11.5px] font-mono text-ink-300 tabular-nums">{s.chunk_count}</span>
      <button onClick={() => onDelete(s.source_id)} disabled={busy}
        title="Delete source"
        className="text-ink-400 hover:text-rose-300 transition disabled:opacity-30">
        <Icon name="close" className="w-4 h-4"/>
      </button>
    </div>
  );
}

function Ingest() {
  const [tab, setTab] = React.useState('upload');
  const [chunkSize, setChunkSize] = React.useState(800);
  const [overlap, setOverlap] = React.useState(150);
  const [embed, setEmbed] = React.useState('jina-embeddings-v3');

  const [sources, setSources] = React.useState([]);
  const [loadingSources, setLoadingSources] = React.useState(true);
  const [busy, setBusy] = React.useState(false);
  const [urlValue, setUrlValue] = React.useState('');
  const [webUrl, setWebUrl] = React.useState('https://example.com');
  const [toast, setToast] = React.useState(null);

  const refresh = React.useCallback(async () => {
    try {
      setLoadingSources(true);
      const items = await api.listSources();
      setSources(items);
    } catch (e) {
      setToast({ tone: 'err', msg: `Couldn't load sources: ${e.message}` });
    } finally {
      setLoadingSources(false);
    }
  }, []);

  React.useEffect(() => { refresh(); }, [refresh]);

  const ingestFiles = async (files) => {
    const allowed = files.filter(f => /\.(pdf|txt)$/i.test(f.name));
    if (allowed.length === 0) {
      setToast({ tone: 'err', msg: 'Only .pdf and .txt files are supported.' });
      return;
    }
    setBusy(true);
    try {
      for (const f of allowed) {
        setToast({ tone: 'info', msg: `Ingesting ${f.name}…` });
        const res = await api.ingestFile(f);
        setToast({ tone: 'ok', msg: `Indexed ${res.source_label} — ${res.chunks_stored} chunks` });
      }
      await refresh();
    } catch (e) {
      setToast({ tone: 'err', msg: e.message });
    } finally {
      setBusy(false);
    }
  };

  const ingestUrl = async (url) => {
    const u = (url || '').trim();
    if (!u) return;
    setBusy(true);
    try {
      setToast({ tone: 'info', msg: `Scraping ${u}…` });
      const res = await api.ingestUrl(u);
      setToast({ tone: 'ok', msg: `Indexed ${res.source_label} — ${res.chunks_stored} chunks` });
      setUrlValue('');
      await refresh();
    } catch (e) {
      setToast({ tone: 'err', msg: e.message });
    } finally {
      setBusy(false);
    }
  };

  const removeSource = async (id) => {
    setBusy(true);
    try {
      const res = await api.deleteSource(id);
      setToast({ tone: 'ok', msg: `Removed source (${res.deleted_chunks} chunks)` });
      await refresh();
    } catch (e) {
      setToast({ tone: 'err', msg: e.message });
    } finally {
      setBusy(false);
    }
  };

  const counts = React.useMemo(() => {
    const urls = sources.filter(s => s.source_type === 'url').length;
    const files = sources.filter(s => s.source_type === 'file').length;
    const chunks = sources.reduce((a, s) => a + (s.chunk_count || 0), 0);
    return { urls, files, chunks };
  }, [sources]);

  return (
    <div className="px-4 lg:px-8 py-7 max-w-[1480px] mx-auto">
      <SectionHeader
        eyebrow="Ingest"
        title="Add sources to your knowledge base"
        subtitle="Scrape any URL or upload PDF/TXT files. Scrybe parses, chunks, embeds with Jina v3, and stores in ChromaDB for grounded Q&A."
        right={
          <div className="hidden md:flex items-center gap-2">
            <Pill tone="cyan" icon="database">{counts.chunks} chunks</Pill>
            <Pill tone="emerald">{sources.length} sources</Pill>
          </div>
        }
      />

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
        {SOURCE_TYPES.map(s => (
          <SourceTab key={s.id} s={s}
            active={tab === s.id}
            disabled={s.id === 'github' || s.id === 'api'}
            onClick={() => setTab(s.id)}/>
        ))}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[1fr_360px] gap-5">
        <div className="space-y-5">
          {tab === 'upload' ? (
            <Dropzone
              busy={busy}
              onPickFiles={ingestFiles}
              urlValue={urlValue}
              setUrlValue={setUrlValue}
              onAddUrl={() => ingestUrl(urlValue)}/>
          ) : null}

          {tab === 'web' ? (
            <Card className="p-6">
              <div className="text-[11px] uppercase tracking-[0.14em] font-mono text-ink-400 mb-2">Web URL</div>
              <h3 className="text-[18px] font-semibold tracking-tight text-ink-50">Scrape a single page</h3>
              <p className="text-[12.5px] text-ink-400 mt-1">Headless Playwright fetches the URL, strips nav/script/footer, and indexes the body text.</p>
              <div className="mt-5 space-y-3">
                <div className="flex items-center gap-2 h-11 px-3.5 rounded-xl glass">
                  <Icon name="globe" className="w-4 h-4 text-cyan-soft"/>
                  <input className="flex-1 bg-transparent outline-none text-[13.5px] text-ink-100 placeholder:text-ink-400"
                    placeholder="https://example.com/article"
                    value={webUrl}
                    onChange={(e) => setWebUrl(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') ingestUrl(webUrl); }}
                    disabled={busy}/>
                </div>
                <Button variant="cyan" icon={busy ? 'spinner' : 'bolt'} disabled={busy || !webUrl.trim()}
                  onClick={() => ingestUrl(webUrl)}>
                  {busy ? 'Scraping…' : 'Scrape and index'}
                </Button>
              </div>
            </Card>
          ) : null}

          {(tab === 'github' || tab === 'api') ? (
            <Card className="p-8 text-center">
              <Icon name={tab === 'github' ? 'github' : 'link'} className="w-8 h-8 mx-auto text-ink-400"/>
              <h3 className="text-[16px] font-semibold text-ink-50 mt-3">Not in this build</h3>
              <p className="text-[13px] text-ink-300 mt-1.5">This source type is part of the design but isn't wired to the Scrybe backend. Use Files or Web URL.</p>
            </Card>
          ) : null}

          <PipelineDiagram/>

          <Card className="overflow-hidden">
            <div className="flex items-center justify-between px-5 py-4 border-b border-white/[0.05]">
              <div>
                <div className="text-[11px] uppercase tracking-[0.14em] font-mono text-ink-400">Library</div>
                <h3 className="text-[15.5px] font-semibold text-ink-50 mt-0.5">
                  {loadingSources ? 'Loading…' : `${sources.length} source${sources.length === 1 ? '' : 's'}`}
                </h3>
              </div>
              <div className="flex items-center gap-2">
                <Pill tone="cyan">{counts.urls} URL</Pill>
                <Pill tone="emerald">{counts.files} file</Pill>
                <Button size="sm" variant="ghost" icon="filter" onClick={refresh}>Refresh</Button>
              </div>
            </div>
            <div className="grid grid-cols-[1.5fr_90px_140px_80px_36px] gap-4 px-4 py-2.5 text-[10.5px] font-mono uppercase tracking-[0.12em] text-ink-400 border-b border-white/[0.05]">
              <span>Source</span><span>Type</span><span>Status</span><span>Chunks</span><span></span>
            </div>
            <div>
              {loadingSources ? (
                <div className="p-6 space-y-2.5">
                  <Skeleton className="h-4 w-[90%]"/>
                  <Skeleton className="h-4 w-[70%]"/>
                  <Skeleton className="h-4 w-[80%]"/>
                </div>
              ) : sources.length === 0 ? (
                <div className="p-8 text-center text-ink-400 text-[13px]">
                  No sources yet. Drop a file above or paste a URL to get started.
                </div>
              ) : (
                sources.map(s => <SourceRow key={s.source_id} s={s} busy={busy} onDelete={removeSource}/>)
              )}
            </div>
          </Card>
        </div>

        <aside className="space-y-5">
          <Card className="p-5">
            <div className="text-[11px] uppercase tracking-[0.14em] font-mono text-ink-400 mb-3">Parsing config</div>
            <div className="space-y-5">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-[12.5px] text-ink-100">Chunk size</label>
                  <span className="font-mono text-[11.5px] text-cyan-soft">{chunkSize} tok</span>
                </div>
                <input type="range" min="128" max="2048" step="64" value={chunkSize} onChange={(e) => setChunkSize(+e.target.value)} className="w-full accent-cyan-soft"/>
                <div className="flex justify-between text-[10px] font-mono text-ink-400 mt-1"><span>128</span><span>2048</span></div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-[12.5px] text-ink-100">Overlap</label>
                  <span className="font-mono text-[11.5px] text-cyan-soft">{overlap} tok</span>
                </div>
                <input type="range" min="0" max="400" step="10" value={overlap} onChange={(e) => setOverlap(+e.target.value)} className="w-full accent-cyan-soft"/>
                <div className="flex justify-between text-[10px] font-mono text-ink-400 mt-1"><span>0</span><span>400</span></div>
              </div>

              <div>
                <label className="text-[12.5px] text-ink-100 mb-2 block">Embedding model</label>
                <div className="space-y-1.5">
                  {['jina-embeddings-v3'].map(m => (
                    <button key={m} onClick={() => setEmbed(m)}
                      className={`w-full flex items-center justify-between p-2.5 rounded-lg border transition text-left
                        ${embed === m ? 'border-cyan-soft/30 bg-cyan-soft/[0.05]' : 'border-white/[0.06] hover:bg-white/[0.03]'}`}>
                      <div className="flex items-center gap-2">
                        <Icon name="sparkle" className={`w-3.5 h-3.5 ${embed === m ? 'text-cyan-soft' : 'text-ink-300'}`}/>
                        <span className="text-[12.5px] text-ink-100 font-mono">{m}</span>
                      </div>
                      <Icon name="check" className="w-3.5 h-3.5 text-cyan-soft"/>
                    </button>
                  ))}
                </div>
                <div className="text-[10.5px] font-mono text-ink-400 mt-2">Config sliders are cosmetic — backend defaults are 800/150.</div>
              </div>
            </div>
          </Card>

          <Card className="p-5 relative overflow-hidden">
            <div className="absolute -bottom-10 -right-10 w-32 h-32 rounded-full bg-cyan-soft/10 blur-3xl"/>
            <div className="text-[11px] uppercase tracking-[0.14em] font-mono text-cyan-soft/80 mb-2">Tip</div>
            <h4 className="text-[14px] font-semibold text-ink-50 leading-snug">Index your resume, then ask "summarize my experience".</h4>
            <p className="text-[12.5px] text-ink-300 mt-1.5 leading-relaxed">RAG works best when the source and the question share vocabulary — long-form prose and structured docs both work.</p>
          </Card>
        </aside>
      </div>

      <Toast tone={toast?.tone} onClose={() => setToast(null)}>{toast?.msg}</Toast>
    </div>
  );
}

window.Ingest = Ingest;
