const SUGGESTIONS = [
  { q: 'Summarize what each indexed source is about', icon: 'sparkle' },
  { q: 'What is the main takeaway from my sources?', icon: 'book' },
  { q: 'List the key facts mentioned across all sources', icon: 'doc' },
  { q: 'Are there any contradictions between the sources?', icon: 'code' },
];

function MarkdownLite({ text }) {
  const lines = (text || '').split('\n');
  const out = [];
  let inTable = false;
  let tableRows = [];

  const flushTable = () => {
    if (tableRows.length === 0) return;
    const [head, _sep, ...rows] = tableRows;
    out.push(
      <div key={`tbl-${out.length}`} className="my-3 overflow-hidden rounded-xl border border-white/[0.06]">
        <table className="w-full text-[12.5px]">
          <thead>
            <tr className="bg-white/[0.025] border-b border-white/[0.06]">
              {head.map((c, i) => <th key={i} className="text-left font-medium text-ink-200 px-3.5 py-2.5">{c}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, ri) => (
              <tr key={ri} className="border-b border-white/[0.04] last:border-0">
                {r.map((c, ci) => <td key={ci} className="px-3.5 py-2 text-ink-100 font-mono text-[12px]">{c}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
    tableRows = [];
  };

  const renderInline = (s) => {
    const parts = [];
    let rest = s;
    let key = 0;
    while (rest.length) {
      const bold = rest.match(/^\*\*([^*]+)\*\*/);
      const code = rest.match(/^`([^`]+)`/);
      if (bold) { parts.push(<strong key={key++} className="text-ink-50 font-semibold">{bold[1]}</strong>); rest = rest.slice(bold[0].length); }
      else if (code) { parts.push(<code key={key++} className="px-1.5 py-0.5 rounded-md bg-white/[0.05] text-cyan-soft font-mono text-[12px] mx-0.5">{code[1]}</code>); rest = rest.slice(code[0].length); }
      else { parts.push(rest[0]); rest = rest.slice(1); }
    }
    return parts;
  };

  lines.forEach((raw, idx) => {
    if (raw.startsWith('|')) {
      const cells = raw.split('|').slice(1, -1).map(s => s.trim());
      tableRows.push(cells);
      inTable = true;
    } else {
      if (inTable) { flushTable(); inTable = false; }
      if (raw.trim() === '') {
        out.push(<div key={idx} className="h-2"/>);
      } else if (raw.startsWith('# ')) {
        out.push(<h1 key={idx} className="text-[20px] font-semibold text-ink-50 mt-3 mb-1">{renderInline(raw.slice(2))}</h1>);
      } else if (raw.startsWith('## ')) {
        out.push(<h2 key={idx} className="text-[17px] font-semibold text-ink-50 mt-3 mb-1">{renderInline(raw.slice(3))}</h2>);
      } else if (/^[-*]\s/.test(raw)) {
        out.push(<li key={idx} className="text-[14px] leading-[1.7] text-ink-100 ml-5 list-disc">{renderInline(raw.replace(/^[-*]\s/, ''))}</li>);
      } else {
        out.push(<p key={idx} className="text-[14px] leading-[1.7] text-ink-100">{renderInline(raw)}</p>);
      }
    }
  });
  if (inTable) flushTable();

  return <div className="space-y-1">{out}</div>;
}

function CitationCard({ c, n }) {
  const isUrl = c.type === 'url';
  return (
    <a
      href={isUrl ? c.label : undefined}
      target={isUrl ? '_blank' : undefined}
      rel="noreferrer"
      className={`block text-left w-full p-3 rounded-xl border transition group
        border-white/[0.06] bg-white/[0.015] hover:bg-white/[0.03] ${isUrl ? 'cursor-pointer' : 'cursor-default'}`}>
      <div className="flex items-start justify-between gap-2 mb-1.5">
        <div className="inline-flex items-center gap-2">
          <span className="inline-flex items-center justify-center w-[18px] h-[18px] rounded-md bg-cyan-soft/15 text-cyan-soft text-[10.5px] font-mono font-semibold">{n}</span>
          <Pill tone={isUrl ? 'cyan' : 'emerald'}>{isUrl ? 'URL' : 'File'}</Pill>
        </div>
        {isUrl ? <Icon name="arrow-up-right" className="w-3.5 h-3.5 text-ink-400 group-hover:text-cyan-soft transition"/> : null}
      </div>
      <div className="text-[12.5px] text-ink-100 leading-snug break-all">{c.label}</div>
    </a>
  );
}

function Composer({ value, onChange, onSubmit, disabled }) {
  const taRef = React.useRef(null);
  React.useEffect(() => {
    if (!taRef.current) return;
    taRef.current.style.height = 'auto';
    taRef.current.style.height = Math.min(taRef.current.scrollHeight, 180) + 'px';
  }, [value]);
  return (
    <form onSubmit={(e) => { e.preventDefault(); onSubmit(); }}
      className="glass rounded-2xl p-3 relative overflow-hidden">
      <div className="pointer-events-none absolute inset-0 rounded-2xl opacity-60" style={{
        background: 'radial-gradient(600px 100px at 50% 100%, rgba(103,232,249,0.08), transparent 70%)'
      }}/>
      <textarea
        ref={taRef}
        rows="1"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); onSubmit(); } }}
        placeholder="Ask anything about your indexed sources…"
        className="w-full bg-transparent outline-none resize-none text-[14.5px] text-ink-50 placeholder:text-ink-400 px-2 py-2 ring-cyan"
      />
      <div className="flex items-center justify-between mt-1 px-1.5">
        <div className="flex items-center gap-1">
          <span className="h-8 px-2.5 rounded-md text-[12px] text-ink-300 inline-flex items-center gap-1.5">
            <Icon name="cpu" className="w-3.5 h-3.5"/> Model
            <span className="text-ink-500">·</span>
            <span className="text-emerald-soft font-mono">llama-3.3-70b · Groq</span>
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="hidden md:flex items-center gap-1 text-[10.5px] font-mono text-ink-400">
            <Kbd>⌘</Kbd><Kbd>↵</Kbd> to send
          </span>
          <Button size="sm" variant="cyan" icon={disabled ? 'spinner' : 'send'} disabled={!value.trim() || disabled} type="submit">
            {disabled ? 'Querying' : 'Send'}
          </Button>
        </div>
      </div>
    </form>
  );
}

function ChatHistory({ chats, activeId, onPick, onNew, onDelete, configured, loading, currentDirty }) {
  return (
    <aside className="hidden xl:flex flex-col w-[260px] shrink-0 border-r border-white/[0.05] h-[calc(100vh-56px)] sticky top-14 bg-ink-950/30">
      <div className="px-4 py-3 border-b border-white/[0.05] flex items-center justify-between">
        <div className="text-[11px] uppercase tracking-[0.14em] font-mono text-ink-400">Chats</div>
        <button onClick={onNew} disabled={!configured && !currentDirty}
          className="text-ink-300 hover:text-cyan-soft transition inline-flex items-center gap-1 text-[11.5px] disabled:opacity-40" title="New chat">
          <Icon name="plus" className="w-3.5 h-3.5"/> New
        </button>
      </div>
      <div className="p-2 flex-1 overflow-y-auto no-scrollbar">
        {!configured ? (
          <div className="px-3 py-4 text-[12px] text-ink-400 leading-relaxed">
            Supabase is not configured — chats persist only in this tab.
            <div className="text-[11px] font-mono text-cyan-soft mt-2">See Settings → Connection.</div>
          </div>
        ) : loading ? (
          <div className="px-3 py-4 space-y-2">
            <Skeleton className="h-8 rounded-md"/>
            <Skeleton className="h-8 rounded-md"/>
            <Skeleton className="h-8 rounded-md"/>
          </div>
        ) : chats.length === 0 ? (
          <div className="px-3 py-4 text-[12px] text-ink-400">No saved chats yet. Send your first message.</div>
        ) : chats.map(c => (
          <div key={c.id}
            className={`group flex items-center gap-1 px-2 py-2 rounded-lg transition mb-0.5
              ${c.id === activeId ? 'bg-white/[0.05] text-ink-50' : 'text-ink-200 hover:bg-white/[0.025]'}`}>
            <button onClick={() => onPick(c.id)} className="flex-1 text-left min-w-0">
              <div className="text-[12.5px] line-clamp-2">{c.title}</div>
              <div className="text-[10.5px] font-mono text-ink-400 mt-0.5">{c.message_count || 0} msg</div>
            </button>
            <button onClick={() => onDelete(c.id)} title="Delete chat"
              className="opacity-0 group-hover:opacity-100 text-ink-400 hover:text-rose-300 transition p-1">
              <Icon name="close" className="w-3.5 h-3.5"/>
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}

function QABlock({ msg, n }) {
  return (
    <div className="border-t border-white/[0.05] first:border-t-0 pt-7 first:pt-0 mt-7 first:mt-0 scroll-mt-24" data-qa-index={n - 1}>
      <div className="text-[11px] uppercase tracking-[0.16em] text-cyan-soft/80 font-mono mb-2 flex items-center gap-2">
        <Dot color="bg-cyan-soft" className="w-1.5 h-1.5"/>
        query #{n}{msg.pending ? ' · pending' : msg.error ? ' · failed' : ` · ${msg.latency}ms${msg.chunks_used ? ` · ${msg.chunks_used} chunks` : ''}`}
      </div>
      <h2 className="text-[20px] md:text-[22px] font-semibold tracking-tight text-ink-50 leading-snug mb-5">
        {msg.question}
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_280px] gap-7">
        <div>
          {msg.error ? (
            <div className="rounded-xl border border-rose-400/30 bg-rose-400/[0.06] p-4 text-[13px] text-rose-300">
              <div className="font-mono text-[11px] uppercase tracking-[0.14em] mb-1">Query failed</div>
              {msg.error}
            </div>
          ) : msg.pending ? (
            <div className="space-y-2.5">
              <Skeleton className="h-3 w-[92%]"/>
              <Skeleton className="h-3 w-[78%]"/>
              <Skeleton className="h-3 w-[85%]"/>
              <Skeleton className="h-3 w-[60%]"/>
              <div className="mt-3 text-[11px] font-mono text-ink-400">
                Embedding query → searching ChromaDB → calling Groq…
              </div>
            </div>
          ) : (
            <React.Fragment>
              <MarkdownLite text={msg.answer}/>
              <div className="mt-5 flex items-center justify-between">
                <button onClick={() => navigator.clipboard?.writeText(msg.answer)}
                  className="h-8 w-8 rounded-md text-ink-300 hover:text-ink-50 hover:bg-white/[0.04] transition flex items-center justify-center" title="Copy">
                  <Icon name="copy" className="w-4 h-4"/>
                </button>
                <span className="text-[10.5px] font-mono text-ink-400">finished · {msg.latency}ms</span>
              </div>
            </React.Fragment>
          )}
        </div>

        <aside>
          <div className="text-[11px] uppercase tracking-[0.14em] font-mono text-ink-400 mb-2 flex items-center justify-between">
            <span>Sources · {msg.sources?.length || 0}</span>
            <Icon name="link" className="w-3.5 h-3.5"/>
          </div>
          <div className="space-y-2">
            {(msg.sources || []).map((c, i) => (
              <CitationCard key={c.source_id || i} c={c} n={i + 1}/>
            ))}
            {!msg.pending && (!msg.sources || msg.sources.length === 0) && !msg.error ? (
              <div className="text-[12px] text-ink-400 p-3 rounded-xl border border-white/[0.06]">
                No sources cited.
              </div>
            ) : null}
          </div>
        </aside>
      </div>
    </div>
  );
}

function Chat() {
  const [input, setInput] = React.useState('');
  const [messages, setMessages] = React.useState([]);
  const [busy, setBusy] = React.useState(false);

  const [chats, setChats] = React.useState([]);
  const [chatId, setChatId] = React.useState(null);
  const [chatsLoading, setChatsLoading] = React.useState(true);
  const [configured, setConfigured] = React.useState(false);
  const [error, setError] = React.useState(null);

  const scrollRef = React.useRef(null);
  const endRef = React.useRef(null);

  React.useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages.length, busy]);

  const loadList = React.useCallback(async () => {
    try {
      const items = await api.listChats();
      setChats(items);
    } catch (e) {
      setError(e.message);
    } finally {
      setChatsLoading(false);
    }
  }, []);

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const s = await api.chatsStatus();
        if (cancelled) return;
        setConfigured(s.configured);
        if (s.configured) await loadList();
        else setChatsLoading(false);
      } catch (e) {
        if (!cancelled) {
          setConfigured(false);
          setChatsLoading(false);
        }
      }
    })();
    return () => { cancelled = true; };
  }, [loadList]);

  const openChat = async (id) => {
    if (busy) return;
    setError(null);
    setBusy(true);
    try {
      const chat = await api.getChat(id);
      const loaded = [];
      const msgs = chat.messages || [];
      for (let i = 0; i < msgs.length; i++) {
        const m = msgs[i];
        if (m.role === 'user') {
          const next = msgs[i + 1];
          if (next && next.role === 'assistant') {
            loaded.push({
              question: m.content,
              answer: next.content,
              sources: next.sources || [],
              latency: next.latency_ms || 0,
              chunks_used: next.chunks_used || 0,
              error: next.error || null,
              pending: false,
            });
            i++;
          } else {
            loaded.push({ question: m.content, answer: '', sources: [], latency: 0, pending: false });
          }
        }
      }
      setMessages(loaded);
      setChatId(id);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const newChat = () => {
    if (busy) return;
    setMessages([]);
    setChatId(null);
    setInput('');
  };

  const removeChat = async (id) => {
    try {
      await api.deleteChat(id);
      if (id === chatId) newChat();
      await loadList();
    } catch (e) {
      setError(e.message);
    }
  };

  const submit = async () => {
    const q = input.trim();
    if (!q || busy) return;
    const t0 = performance.now();
    setBusy(true);
    setError(null);
    setInput('');

    let activeChatId = chatId;
    if (configured && !activeChatId) {
      try {
        const created = await api.createChat();
        activeChatId = created.id;
        setChatId(activeChatId);
      } catch (e) {
        setError(`Persistence: ${e.message}`);
      }
    }

    setMessages(prev => [...prev, { question: q, answer: '', sources: [], latency: 0, pending: true }]);

    try {
      const res = await api.query(q, 5, activeChatId);
      const latency = Math.round(performance.now() - t0);
      setMessages(prev => prev.map((m, i) =>
        i === prev.length - 1
          ? { ...m, answer: res.answer, sources: res.sources, latency, chunks_used: res.chunks_used, pending: false }
          : m
      ));
      if (configured) await loadList();
    } catch (e) {
      const latency = Math.round(performance.now() - t0);
      setMessages(prev => prev.map((m, i) =>
        i === prev.length - 1 ? { ...m, error: e.message, latency, pending: false } : m
      ));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex">
      <ChatHistory
        chats={chats} activeId={chatId}
        onPick={openChat} onNew={newChat} onDelete={removeChat}
        configured={configured} loading={chatsLoading}
        currentDirty={messages.length > 0}
      />
      <div className="flex-1 min-w-0 flex flex-col">
        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          <div className="max-w-[1100px] mx-auto px-4 lg:px-8 py-7">
            {error ? (
              <div className="mb-5 rounded-xl border border-rose-400/30 bg-rose-400/[0.06] p-3 text-[12.5px] text-rose-300">
                {error}
              </div>
            ) : null}

            {messages.length === 0 ? (
              <div className="fade-up">
                <div className="text-[11px] uppercase tracking-[0.16em] text-cyan-soft/80 font-mono mb-2.5 flex items-center gap-2">
                  <Dot color="bg-cyan-soft" className="w-1.5 h-1.5 pulse-dot"/>
                  ready{configured ? ' · chats saved to Supabase' : ' · running without persistence'}
                </div>
                <h1 className="text-[26px] md:text-[30px] font-semibold tracking-tight text-ink-50 leading-tight">
                  Ask anything about your indexed sources.
                </h1>
                <p className="text-ink-300 text-[14px] mt-3 max-w-xl">
                  Scrybe retrieves the top chunks from ChromaDB, then asks Groq's <span className="font-mono text-cyan-soft">llama-3.3-70b</span> to ground an answer with citations.
                </p>

                <div className="mt-8">
                  <div className="text-[11px] uppercase tracking-[0.14em] font-mono text-ink-400 mb-3">Try one</div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                    {SUGGESTIONS.map((s, i) => (
                      <button key={i}
                        onClick={() => setInput(s.q)}
                        className="text-left p-3 rounded-xl border border-white/[0.06] hover:border-cyan-soft/30 hover:bg-cyan-soft/[0.03] transition group">
                        <div className="flex items-center gap-2.5">
                          <Icon name={s.icon} className="w-3.5 h-3.5 text-cyan-soft"/>
                          <span className="text-[12.5px] text-ink-100 flex-1">{s.q}</span>
                          <Icon name="arrow-right" className="w-3.5 h-3.5 text-ink-400 group-hover:text-cyan-soft transition"/>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Pill tone="neutral" icon="chat">{messages.length} message{messages.length === 1 ? '' : 's'}</Pill>
                    {chatId ? <Pill tone="cyan">saved</Pill> : configured ? null : <Pill tone="amber">unsaved</Pill>}
                  </div>
                  <button onClick={newChat} disabled={busy}
                    className="text-[12px] text-ink-300 hover:text-cyan-soft transition inline-flex items-center gap-1.5 disabled:opacity-40">
                    <Icon name="plus" className="w-3.5 h-3.5"/> New chat
                  </button>
                </div>
                {messages.map((m, i) => <QABlock key={i} msg={m} n={i + 1}/>)}
                <div ref={endRef}/>
              </div>
            )}
          </div>
        </div>

        <div className="border-t border-white/[0.05] bg-ink-950/60 backdrop-blur-xl">
          <div className="max-w-[1100px] mx-auto px-4 lg:px-8 py-4">
            <Composer value={input} onChange={setInput} onSubmit={submit} disabled={busy}/>
            <div className="mt-2 text-[10.5px] font-mono text-ink-400 flex items-center justify-between">
              <span>Scrybe can be wrong. Verify critical claims against citations.</span>
              <span className="flex items-center gap-3">
                <span className="inline-flex items-center gap-1.5">
                  <Dot color={configured ? 'bg-emerald-soft' : 'bg-amber-300'} className="w-1 h-1 pulse-dot"/>
                  {configured ? 'persistence on' : 'persistence off'}
                </span>
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

window.Chat = Chat;
