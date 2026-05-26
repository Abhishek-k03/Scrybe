const VIEW_W = 900;
const VIEW_H = 520;

function VectorMap({ map, queryResult, busy, hoveredId, setHoveredId }) {
  const points = map?.points || [];
  const sources = map?.sources || [];
  const queryPoint = queryResult?.query_point || null;
  const hits = queryResult?.hits || [];
  const hitIds = new Set(hits.map(h => h.id));

  const hovered = points.find(p => p.id === hoveredId) || hits.find(h => h.id === hoveredId);

  if (points.length === 0 && !busy) {
    return (
      <Card className="p-10 text-center">
        <Icon name="vector" className="w-8 h-8 mx-auto text-ink-400"/>
        <h3 className="text-[16px] font-semibold text-ink-50 mt-3">No vectors yet</h3>
        <p className="text-[13px] text-ink-300 mt-1.5">
          Ingest a source on the <span className="font-mono text-cyan-soft">Ingest</span> screen to see chunks projected here.
        </p>
      </Card>
    );
  }

  return (
    <Card className="relative overflow-hidden">
      <div className="absolute inset-0 dotgrid opacity-40 pointer-events-none"/>
      <div className="absolute -top-20 -right-20 w-80 h-80 rounded-full bg-cyan-soft/[0.05] blur-3xl pointer-events-none"/>
      <div className="absolute -bottom-20 -left-20 w-80 h-80 rounded-full bg-emerald-soft/[0.05] blur-3xl pointer-events-none"/>

      <div className="relative">
        <div className="flex items-start justify-between gap-4 px-5 py-4 border-b border-white/[0.05]">
          <div>
            <div className="text-[11px] uppercase tracking-[0.14em] font-mono text-ink-400">Vector space · PCA · {map?.dim || '?'}d → 2d</div>
            <h3 className="text-[15.5px] font-semibold text-ink-50 mt-0.5">{points.length} chunks · {sources.length} source{sources.length === 1 ? '' : 's'}</h3>
          </div>
          <div className="flex items-center gap-2 flex-wrap justify-end max-w-[60%]">
            {sources.slice(0, 8).map(s => (
              <span key={s.source_id}
                className="inline-flex items-center gap-1.5 text-[11px] font-mono text-ink-300 max-w-[200px]"
                title={s.source_label}>
                <span className="w-2 h-2 rounded-full shrink-0" style={{background: s.color}}/>
                <span className="truncate">{s.source_label}</span>
                <span className="text-ink-500">({s.count})</span>
              </span>
            ))}
            {sources.length > 8 ? (
              <span className="text-[11px] font-mono text-ink-400">+{sources.length - 8}</span>
            ) : null}
          </div>
        </div>

        <div className="relative h-[480px] w-full">
          <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} className="absolute inset-0 w-full h-full" preserveAspectRatio="xMidYMid meet">
            <defs>
              <radialGradient id="haloQ" cx="50%" cy="50%">
                <stop offset="0%"  stopColor="#67e8f9" stopOpacity="0.5"/>
                <stop offset="100%" stopColor="#67e8f9" stopOpacity="0"/>
              </radialGradient>
            </defs>

            {points.map(p => {
              const isHit = hitIds.has(p.id);
              const isHovered = hoveredId === p.id;
              return (
                <circle key={p.id}
                  cx={p.x} cy={p.y}
                  r={isHovered ? 6 : isHit ? 4.5 : 3}
                  fill={p.color || '#67e8f9'}
                  opacity={queryPoint ? (isHit ? 1 : 0.35) : isHovered ? 1 : 0.85}
                  stroke={isHovered ? '#ffffff' : 'none'}
                  strokeWidth={isHovered ? 1.2 : 0}
                />
              );
            })}

            {queryPoint && hits.map((h, i) => (
              <g key={h.id}>
                <line x1={queryPoint.x} y1={queryPoint.y} x2={h.x} y2={h.y}
                  stroke="#67e8f9" strokeOpacity={hoveredId === h.id ? 0.9 : 0.35}
                  strokeWidth={hoveredId === h.id ? 1.6 : 1}/>
                <line x1={queryPoint.x} y1={queryPoint.y} x2={h.x} y2={h.y}
                  stroke="#67e8f9" strokeOpacity="0.7" strokeWidth="0.6"
                  className="beam" style={{animationDelay: `${i*0.4}s`}}/>
              </g>
            ))}

            {queryPoint ? (
              <g>
                <circle cx={queryPoint.x} cy={queryPoint.y} r="50" fill="url(#haloQ)"/>
                <circle cx={queryPoint.x} cy={queryPoint.y} r="6" fill="#e5f9ff" stroke="#67e8f9" strokeWidth="1.5"/>
                <text x={queryPoint.x + 10} y={queryPoint.y - 10} fontSize="11" fontFamily="Geist Mono, monospace" fill="#e5e5ea">query</text>
              </g>
            ) : null}

            {points.map(p => (
              <circle key={`hit-${p.id}`}
                cx={p.x} cy={p.y} r={10}
                fill="transparent"
                onMouseEnter={() => setHoveredId(p.id)}
                onMouseLeave={() => setHoveredId(null)}
                style={{ cursor: 'pointer' }}/>
            ))}
          </svg>

          {busy ? (
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-soft/60 to-transparent scanline pointer-events-none"/>
          ) : null}

          {hovered ? (
            <div className="absolute bottom-3 left-3 right-3 max-w-md px-3 py-2 rounded-lg glass border border-white/10 pointer-events-none">
              <div className="flex items-center gap-2 mb-1">
                <span className="w-2 h-2 rounded-full shrink-0" style={{background: hovered.color || sources.find(s => s.source_id === hovered.source_id)?.color || '#67e8f9'}}/>
                <span className="text-[11.5px] font-mono text-ink-200 truncate">{hovered.source_label} · chunk {hovered.chunk_index}</span>
                {hovered.distance !== undefined ? (
                  <span className="ml-auto text-[10.5px] font-mono text-cyan-soft tabular-nums">dist {hovered.distance.toFixed(3)}</span>
                ) : null}
              </div>
              <div className="text-[11.5px] text-ink-100 leading-snug line-clamp-3">{hovered.text_preview || '—'}</div>
            </div>
          ) : null}
        </div>

        <div className="flex items-center justify-between px-5 py-3 border-t border-white/[0.05] text-[11px] font-mono text-ink-400">
          <span>numpy SVD · top-2 components · per-source color</span>
          <span>{queryPoint ? `${hits.length} top hit${hits.length === 1 ? '' : 's'} linked` : 'run a query to overlay top-k'}</span>
        </div>
      </div>
    </Card>
  );
}

function HitRow({ hit, n, hoveredId, setHoveredId, color }) {
  const picked = hoveredId === hit.id;
  return (
    <button
      onMouseEnter={() => setHoveredId(hit.id)}
      onMouseLeave={() => setHoveredId(null)}
      className={`w-full text-left p-3 rounded-xl border transition
        ${picked ? 'border-cyan-soft/40 bg-cyan-soft/[0.05] shadow-glow' : 'border-white/[0.06] bg-white/[0.015] hover:bg-white/[0.03]'}`}>
      <div className="flex items-start justify-between gap-3 mb-1.5">
        <div className="flex items-center gap-2 min-w-0">
          <span className="inline-flex items-center justify-center w-[18px] h-[18px] rounded-md bg-cyan-soft/15 text-cyan-soft text-[10.5px] font-mono font-semibold">{n}</span>
          <span className="w-2 h-2 rounded-full shrink-0" style={{background: color || '#67e8f9'}}/>
          <span className="text-[11.5px] font-mono text-ink-300 truncate">{hit.source_label} · chunk {hit.chunk_index}</span>
        </div>
        <span className="text-[11px] font-mono text-cyan-soft tabular-nums shrink-0">d {hit.distance.toFixed(3)}</span>
      </div>
      <p className="text-[12.5px] text-ink-100 leading-relaxed line-clamp-3">{hit.text_preview}</p>
    </button>
  );
}

function Retrieval() {
  const [query, setQuery] = React.useState('');
  const [map, setMap] = React.useState(null);
  const [queryResult, setQueryResult] = React.useState(null);
  const [mapLoading, setMapLoading] = React.useState(true);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [hoveredId, setHoveredId] = React.useState(null);

  const loadMap = React.useCallback(async () => {
    try {
      setMapLoading(true);
      const m = await api.vectorMap();
      setMap(m);
    } catch (e) {
      setError(e.message);
    } finally {
      setMapLoading(false);
    }
  }, []);

  React.useEffect(() => { loadMap(); }, [loadMap]);

  const run = async () => {
    const q = query.trim();
    if (!q || busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.vectorMapQuery(q, 6);
      setQueryResult(res);
    } catch (e) {
      setError(e.message);
      setQueryResult(null);
    } finally {
      setBusy(false);
    }
  };

  const reset = () => { setQueryResult(null); setQuery(''); };

  const sourceColor = React.useMemo(() => {
    const m = {};
    (map?.sources || []).forEach(s => { m[s.source_id] = s.color; });
    return m;
  }, [map]);

  const stats = {
    sources: map?.sources?.length || 0,
    chunks: map?.point_count || 0,
    topK: queryResult?.hits?.length ?? '—',
    dim: map?.dim || '—',
  };

  return (
    <div className="px-4 lg:px-8 py-7 max-w-[1480px] mx-auto">
      <SectionHeader
        eyebrow="Retrieval"
        title="See exactly what WebOracle pulls back."
        subtitle="Every chunk in your index is a point in this 2-D PCA projection of the 1024-dim Jina embedding space. Hit Run to overlay your query and the top-k matches."
        right={
          <div className="hidden md:flex items-center gap-2">
            <Pill tone="cyan" icon="database">{stats.chunks} chunks</Pill>
            <Pill tone="emerald">jina-embeddings-v3 · {stats.dim}d</Pill>
            <Button size="sm" variant="ghost" icon="filter" onClick={loadMap}>Refit</Button>
          </div>
        }
      />

      <Card className="p-3 mb-5">
        <form onSubmit={(e) => { e.preventDefault(); run(); }} className="flex items-center gap-3 px-2">
          <Icon name="search" className="w-4 h-4 text-cyan-soft"/>
          <input value={query} onChange={(e) => setQuery(e.target.value)}
            className="flex-1 bg-transparent outline-none text-[14px] text-ink-100 ring-cyan py-2"
            placeholder="Type a query to project + retrieve top-k…"/>
          {queryResult ? (
            <button type="button" onClick={reset} className="text-[12px] text-ink-400 hover:text-ink-100">Clear</button>
          ) : null}
          <Button size="sm" variant="cyan" icon={busy ? 'spinner' : 'bolt'} disabled={busy || !query.trim()} type="submit">
            {busy ? 'Running' : 'Run'}
          </Button>
        </form>
      </Card>

      {error ? (
        <div className="mb-5 rounded-xl border border-rose-400/30 bg-rose-400/[0.06] p-3 text-[12.5px] text-rose-300">{error}</div>
      ) : null}

      <div className="grid grid-cols-1 xl:grid-cols-[1fr_400px] gap-5">
        <div className="space-y-5">
          {mapLoading ? (
            <Card className="p-10 text-center">
              <Skeleton className="h-[480px] rounded-xl"/>
            </Card>
          ) : (
            <VectorMap map={map} queryResult={queryResult} busy={busy} hoveredId={hoveredId} setHoveredId={setHoveredId}/>
          )}

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { k: 'Sources', v: stats.sources, d: 'indexed' },
              { k: 'Chunks', v: stats.chunks, d: 'in vector space' },
              { k: 'Top-k', v: stats.topK, d: 'last query' },
              { k: 'Dimensions', v: stats.dim, d: 'before PCA' },
            ].map(s => (
              <div key={s.k} className="glass rounded-xl p-3.5">
                <div className="text-[10.5px] uppercase tracking-[0.14em] font-mono text-ink-400">{s.k}</div>
                <div className="text-[20px] font-semibold tracking-tight tabular-nums mt-1">{s.v}</div>
                <div className="text-[10.5px] font-mono text-ink-400 mt-0.5">{s.d}</div>
              </div>
            ))}
          </div>
        </div>

        <aside className="space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-[11px] uppercase tracking-[0.14em] font-mono text-ink-400">Top matches</div>
              <h3 className="text-[15.5px] font-semibold text-ink-50 mt-0.5">
                {busy ? 'Embedding query…' : queryResult ? `${queryResult.hits.length} chunks · ranked` : 'No query yet'}
              </h3>
            </div>
          </div>
          <div className="space-y-2.5">
            {busy ? (
              <React.Fragment>
                <Skeleton className="h-16 rounded-xl"/>
                <Skeleton className="h-16 rounded-xl"/>
                <Skeleton className="h-16 rounded-xl"/>
              </React.Fragment>
            ) : queryResult?.hits?.length ? (
              queryResult.hits.map((h, i) => (
                <HitRow key={h.id} hit={h} n={i + 1}
                  hoveredId={hoveredId} setHoveredId={setHoveredId}
                  color={sourceColor[h.source_id]}/>
              ))
            ) : (
              <div className="rounded-xl border border-white/[0.06] bg-white/[0.015] p-4 text-[12.5px] text-ink-400">
                Run a query above to see ChromaDB's top-k retrieval, projected and ranked.
              </div>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

window.Retrieval = Retrieval;
