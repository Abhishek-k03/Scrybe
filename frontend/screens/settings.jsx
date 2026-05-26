function StatusRow({ label, value, tone = 'neutral', mono = true }) {
  const tones = {
    neutral: 'text-ink-100',
    ok: 'text-emerald-soft',
    err: 'text-rose-300',
    cyan: 'text-cyan-soft',
  };
  return (
    <div className="flex items-center justify-between py-3 border-b border-white/[0.05] last:border-0">
      <span className="text-[12.5px] text-ink-300">{label}</span>
      <span className={`text-[12.5px] ${mono ? 'font-mono' : ''} ${tones[tone]}`}>{value}</span>
    </div>
  );
}

function Settings({ health, stats }) {
  const [retest, setRetest] = React.useState(0);
  const [pinging, setPinging] = React.useState(false);
  const [pingResult, setPingResult] = React.useState(null);

  const ping = async () => {
    setPinging(true);
    const t0 = performance.now();
    try {
      const h = await api.health();
      setPingResult({ ok: true, ms: Math.round(performance.now() - t0), version: h.version });
    } catch (e) {
      setPingResult({ ok: false, ms: Math.round(performance.now() - t0), error: e.message });
    } finally {
      setPinging(false);
      setRetest(n => n + 1);
    }
  };

  return (
    <div className="px-4 lg:px-8 py-7 max-w-[1200px] mx-auto">
      <SectionHeader
        eyebrow="Settings"
        title="Connection & configuration"
        subtitle="WebOracle is a local prototype. Settings here are read-only — point the frontend at a different backend by setting window.WEBORACLE_API_BASE before load."
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="text-[11px] uppercase tracking-[0.14em] font-mono text-ink-400">Backend</div>
              <h3 className="text-[18px] font-semibold tracking-tight text-ink-50 mt-1">Connection</h3>
            </div>
            <Button size="sm" variant="cyan" icon={pinging ? 'spinner' : 'bolt'} onClick={ping} disabled={pinging}>
              {pinging ? 'Pinging' : 'Ping'}
            </Button>
          </div>

          <StatusRow label="API base URL" value={api?.base || 'http://localhost:8000'}/>
          <StatusRow label="Status"
            value={health.ok ? 'connected' : 'unreachable'}
            tone={health.ok ? 'ok' : 'err'}/>
          <StatusRow label="Backend version" value={health.version ? `v${health.version}` : '—'}/>
          {pingResult ? (
            <StatusRow label="Last ping"
              value={pingResult.ok ? `${pingResult.ms}ms · v${pingResult.version}` : `failed · ${pingResult.error}`}
              tone={pingResult.ok ? 'ok' : 'err'}/>
          ) : null}
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="text-[11px] uppercase tracking-[0.14em] font-mono text-ink-400">Index</div>
              <h3 className="text-[18px] font-semibold tracking-tight text-ink-50 mt-1">Storage</h3>
            </div>
            <Pill tone="emerald">ChromaDB</Pill>
          </div>

          <StatusRow label="Sources indexed" value={stats.loading ? '—' : String(stats.sources)} mono={true}/>
          <StatusRow label="Total chunks" value={stats.loading ? '—' : String(stats.chunks)} mono={true}/>
          <StatusRow label="URL sources" value={stats.loading ? '—' : String(stats.urls)} mono={true}/>
          <StatusRow label="File sources" value={stats.loading ? '—' : String(stats.files)} mono={true}/>
          {stats.error ? (
            <StatusRow label="Error" value={stats.error} tone="err"/>
          ) : null}
        </Card>

        <Card className="p-6">
          <div className="text-[11px] uppercase tracking-[0.14em] font-mono text-ink-400 mb-1">Models</div>
          <h3 className="text-[18px] font-semibold tracking-tight text-ink-50 mb-3">Pipeline</h3>
          <StatusRow label="Scraper" value="Playwright headless Chromium" tone="cyan"/>
          <StatusRow label="Chunker" value="sliding window · 800 tok · 150 overlap" tone="cyan"/>
          <StatusRow label="Embedder" value="jina-embeddings-v3 · 1024d" tone="cyan"/>
          <StatusRow label="Retriever" value="ChromaDB · cosine · top-5" tone="cyan"/>
          <StatusRow label="LLM" value="llama-3.3-70b-versatile · Groq" tone="cyan"/>
        </Card>

        <Card className="p-6">
          <div className="text-[11px] uppercase tracking-[0.14em] font-mono text-ink-400 mb-1">Keys</div>
          <h3 className="text-[18px] font-semibold tracking-tight text-ink-50 mb-3">Required environment</h3>
          <p className="text-[12.5px] text-ink-300 mb-3">Set these in <span className="font-mono text-cyan-soft">backend/.env</span> before starting uvicorn. WebOracle won't expose the values themselves — connection status above is the truth source.</p>
          <ul className="space-y-2 text-[12.5px]">
            <li className="flex items-center justify-between py-2 border-b border-white/[0.05]">
              <span className="font-mono text-ink-100">GROQ_API_KEY</span>
              <span className="text-[11px] font-mono text-ink-400">console.groq.com</span>
            </li>
            <li className="flex items-center justify-between py-2">
              <span className="font-mono text-ink-100">JINA_API_KEY</span>
              <span className="text-[11px] font-mono text-ink-400">jina.ai</span>
            </li>
          </ul>
          <div className="mt-4 text-[11.5px] text-ink-400 font-mono">
            If /api/query returns 502 with "LLM call failed", regenerate the Groq key.
          </div>
        </Card>
      </div>
    </div>
  );
}

window.Settings = Settings;
