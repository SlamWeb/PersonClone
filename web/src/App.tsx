import { FormEvent, useEffect, useMemo, useState } from 'react';
import { Activity, Check, Clock3, Copy, ExternalLink, Plus, Settings2, SlidersHorizontal, Trash2, X } from 'lucide-react';
import {
  AuthorJob,
  cancelAuthorJob,
  ChatSessionSummary,
  deleteSession,
  fetchAuthorJobs,
  fetchPersonas,
  fetchSession,
  fetchSessions,
  fetchSuggestions,
  fetchTrace,
  PersonaInfo,
  retryAuthorJob,
  Source,
  streamChat,
  TraceStage,
  TracePayload
} from './api';
import {
  AddAuthorModal,
  AuthorLibraryPage,
  PersonaSwitcher
} from './AuthorManager';

type Message = {
  id: string;
  role: 'user' | 'assistant' | 'error';
  text: string;
  sources?: Source[] | null;
  traceId?: string | null;
};

type WriterPrompt = 'current' | 'strong_identity' | 'persona_pack';

const USER_AVATAR = '你';
const OPENING_LINE = '今天想聊点什么？';

function initialWriterPrompt(): WriterPrompt {
  const saved = localStorage.getItem('pf-writer-prompt');
  return saved === 'current' || saved === 'persona_pack' || saved === 'strong_identity'
    ? saved
    : 'strong_identity';
}

export default function App() {
  const [personas, setPersonas] = useState<PersonaInfo[]>([]);
  const [author, setAuthor] = useState('');
  const [sessions, setSessions] = useState<ChatSessionSummary[]>([]);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [queryMode, setQueryMode] = useState<'raw' | 'grounded'>('grounded');
  const [writerPrompt, setWriterPrompt] = useState<WriterPrompt>(initialWriterPrompt);
  const [parentTopK, setParentTopK] = useState(20);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [trace, setTrace] = useState<TracePayload | null>(null);
  const [traceOpen, setTraceOpen] = useState(false);
  const [traceLoading, setTraceLoading] = useState(false);
  const [traceError, setTraceError] = useState('');
  const [liveStatus, setLiveStatus] = useState<string | null>(null);
  const [developerMode, setDeveloperMode] = useState(() => localStorage.getItem('pf-developer-mode') === 'true');
  const [traceCapture, setTraceCapture] = useState<'summary' | 'full'>(() =>
    localStorage.getItem('pf-trace-capture') === 'full' ? 'full' : 'summary'
  );
  const [status, setStatus] = useState('Loading local personas...');
  const [busy, setBusy] = useState(false);
  const [authorJobs, setAuthorJobs] = useState<AuthorJob[]>([]);
  const [authorLibraryOpen, setAuthorLibraryOpen] = useState(() => window.location.pathname === '/authors');
  const [addAuthorOpen, setAddAuthorOpen] = useState(false);

  const selectedPersona = useMemo(
    () => personas.find((item) => item.author === author) || null,
    [personas, author]
  );
  const canSend = useMemo(() => Boolean(author && input.trim() && !busy), [author, input, busy]);

  useEffect(() => {
    Promise.all([fetchPersonas(), fetchAuthorJobs()])
      .then(([payload, jobs]) => {
        setPersonas(payload.personas);
        setAuthorJobs(jobs);
        const selected = payload.default_author || payload.personas[0]?.author || '';
        setAuthor(selected);
        setStatus(selected ? `Ready: ${selected}` : 'No local persona index found.');
      })
      .catch((error) => {
        setStatus(String(error.message || error));
      });
  }, []);

  useEffect(() => {
    const timer = window.setInterval(async () => {
      try {
        const [payload, jobs] = await Promise.all([fetchPersonas(), fetchAuthorJobs()]);
        setPersonas(payload.personas);
        setAuthorJobs(jobs);
        setAuthor((current) => current || payload.default_author || payload.personas[0]?.author || '');
      } catch {
        // A transient polling failure should not interrupt an active chat.
      }
    }, 3000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    function handlePopState() {
      setAuthorLibraryOpen(window.location.pathname === '/authors');
    }
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  useEffect(() => {
    localStorage.setItem('pf-developer-mode', String(developerMode));
  }, [developerMode]);

  useEffect(() => {
    localStorage.setItem('pf-trace-capture', traceCapture);
  }, [traceCapture]);

  useEffect(() => {
    localStorage.setItem('pf-writer-prompt', writerPrompt);
  }, [writerPrompt]);

  useEffect(() => {
    if (selectedPersona && !selectedPersona.persona_pack_available && writerPrompt === 'persona_pack') {
      setWriterPrompt('strong_identity');
    }
  }, [selectedPersona, writerPrompt]);

  useEffect(() => {
    if (!author) return;
    refreshSessions(author);
    refreshSuggestions(author);
    setCurrentSessionId(null);
    setMessages([]);
    setLiveStatus(null);
  }, [author]);

  async function refreshSessions(targetAuthor = author) {
    if (!targetAuthor) return;
    try {
      setSessions(await fetchSessions(targetAuthor));
    } catch (error) {
      setStatus(String((error as Error).message || error));
    }
  }

  async function refreshSuggestions(targetAuthor = author) {
    if (!targetAuthor) return;
    try {
      setSuggestions(await fetchSuggestions(targetAuthor));
    } catch {
      setSuggestions([]);
    }
  }

  async function openSession(sessionId: string) {
    if (!author || busy) return;
    try {
      const session = await fetchSession(author, sessionId);
      setCurrentSessionId(session.id);
      setMessages(
        session.messages.map((message) => ({
          id: makeId(),
          role: message.role,
          text: message.text,
          sources: message.sources,
          traceId: message.trace_id
        }))
      );
      setStatus(`Ready: ${author}`);
    } catch (error) {
      setStatus(String((error as Error).message || error));
    }
  }

  async function removeSession(sessionId: string) {
    if (!author || busy) return;
    await deleteSession(author, sessionId);
    if (currentSessionId === sessionId) {
      newChat();
    }
    await refreshSessions(author);
  }

  function newChat() {
    setCurrentSessionId(null);
    setMessages([]);
    setInput('');
    setTrace(null);
    setTraceOpen(false);
    setLiveStatus(null);
  }

  function showAuthorLibrary() {
    window.history.pushState({}, '', '/authors');
    setAuthorLibraryOpen(true);
  }

  function showChat() {
    window.history.pushState({}, '', '/');
    setAuthorLibraryOpen(false);
  }

  function selectAuthor(nextAuthor: string) {
    setAuthor(nextAuthor);
    showChat();
  }

  function upsertAuthorJob(job: AuthorJob) {
    setAuthorJobs((items) => [job, ...items.filter((item) => item.id !== job.id)]);
  }

  async function cancelJob(jobId: string) {
    try {
      upsertAuthorJob(await cancelAuthorJob(jobId));
    } catch (error) {
      setStatus(String((error as Error).message || error));
    }
  }

  async function retryJob(jobId: string) {
    try {
      upsertAuthorJob(await retryAuthorJob(jobId));
    } catch (error) {
      setStatus(String((error as Error).message || error));
    }
  }

  async function openTrace(traceId: string) {
    if (!author) return;
    setTraceOpen(true);
    setTrace(null);
    setTraceError('');
    setTraceLoading(true);
    try {
      setTrace(await fetchTrace(author, traceId));
    } catch (error) {
      setTraceError(String((error as Error).message || error));
    } finally {
      setTraceLoading(false);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = input.trim();
    if (!text || !author) return;

    const userId = makeId();
    const assistantId = makeId();
    let answerStarted = false;
    setMessages((items) => [...items, { id: userId, role: 'user', text }]);
    setInput('');
    setBusy(true);
    setLiveStatus(queryMode === 'grounded' ? '正在理解问题' : '正在检索历史表达');
    setStatus('Retrieving and generating...');

    try {
      await streamChat(
        {
          author,
          session_id: currentSessionId,
          query: text,
          query_mode: queryMode,
          writer_prompt: writerPrompt,
          parent_top_k: parentTopK,
          trace_capture: developerMode ? traceCapture : 'summary'
        },
        {
          onMeta: (payload) => {
            const sessionId = String(payload.session_id || '');
            if (sessionId) setCurrentSessionId(sessionId);
          },
          onStatus: (payload) => {
            if (payload.label) setLiveStatus(payload.label);
          },
          onToken: (token) => {
            if (!answerStarted) {
              answerStarted = true;
              setLiveStatus(null);
              setMessages((items) => [...items, { id: assistantId, role: 'assistant', text: token }]);
              return;
            }
            setMessages((items) =>
              items.map((message) =>
                message.id === assistantId ? { ...message, text: message.text + token } : message
              )
            );
          },
          onDone: async (payload) => {
            setCurrentSessionId(payload.session_id);
            setLiveStatus(null);
            setMessages((items) =>
              items.some((message) => message.id === assistantId)
                ? items.map((message) =>
                    message.id === assistantId
                      ? {
                          ...message,
                          text: payload.answer || message.text,
                          sources: payload.sources,
                          traceId: payload.trace_id
                        }
                      : message
                  )
                : [
                    ...items,
                    {
                      id: assistantId,
                      role: 'assistant',
                      text: payload.answer,
                      sources: payload.sources,
                      traceId: payload.trace_id
                    }
                  ]
            );
            setStatus(`Ready: ${author}`);
            await refreshSessions(author);
          },
          onError: (message) => {
            throw new Error(message);
          }
        }
      );
    } catch (error) {
      setLiveStatus(null);
      setMessages((items) => [
        ...items.filter((message) => message.id !== assistantId),
        { id: makeId(), role: 'error', text: String((error as Error).message || error) }
      ]);
      setStatus('Error');
    } finally {
      setBusy(false);
    }
  }

  if (authorLibraryOpen) {
    return (
      <>
        <AuthorLibraryPage
          personas={personas}
          jobs={authorJobs}
          onBack={showChat}
          onAdd={() => setAddAuthorOpen(true)}
          onSelect={selectAuthor}
          onCancel={cancelJob}
          onRetry={retryJob}
        />
        <AddAuthorModal
          open={addAuthorOpen}
          jobs={authorJobs}
          onClose={() => setAddAuthorOpen(false)}
          onJobCreated={upsertAuthorJob}
          onReady={(nextAuthor) => {
            setAddAuthorOpen(false);
            selectAuthor(nextAuthor);
          }}
        />
      </>
    );
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <PersonaSwitcher
          personas={personas}
          jobs={authorJobs}
          selectedPersona={selectedPersona}
          onSelect={setAuthor}
          onAdd={() => setAddAuthorOpen(true)}
          onManage={showAuthorLibrary}
        />

        <button className="new-chat-button" type="button" onClick={newChat}>
          <Plus size={16} />
          新对话
        </button>

        <section className="session-section">
          <div className="section-label">历史对话</div>
          <div className="session-list">
            {sessions.length === 0 ? (
              <div className="muted-empty">暂无历史会话</div>
            ) : (
              sessions.map((session) => (
                <div className={`session-item ${session.id === currentSessionId ? 'active' : ''}`} key={session.id}>
                  <button className="session-open" type="button" onClick={() => openSession(session.id)}>
                    <span>{session.title}</span>
                    <small>{session.message_count} 条消息</small>
                  </button>
                  <button
                    className="delete-session"
                    type="button"
                    title="删除会话"
                    onClick={() => removeSession(session.id)}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))
            )}
          </div>
        </section>

        <details className="advanced-settings">
          <summary>
            <Settings2 size={15} />
            检索与生成设置
          </summary>
          <div className="control-grid">
            <label>
              RAG
              <select value={queryMode} onChange={(event) => setQueryMode(event.target.value as 'raw' | 'grounded')}>
                <option value="grounded">Grounded</option>
                <option value="raw">Raw</option>
              </select>
            </label>
            <label>
              TopK
              <input
                type="number"
                min={1}
                max={40}
                value={parentTopK}
                onChange={(event) => setParentTopK(Number(event.target.value) || 20)}
              />
            </label>
          </div>

          {developerMode ? (
            <label>
              Trace 记录
              <select value={traceCapture} onChange={(event) => setTraceCapture(event.target.value as 'summary' | 'full')}>
                <option value="summary">摘要</option>
                <option value="full">完整本地记录</option>
              </select>
            </label>
          ) : null}
        </details>

        <button
          className={`developer-mode-toggle ${developerMode ? 'enabled' : ''}`}
          type="button"
          onClick={() => setDeveloperMode((enabled) => !enabled)}
          aria-pressed={developerMode}
        >
          <SlidersHorizontal size={15} />
          {developerMode ? '开发者模式已开启' : '开发者模式'}
        </button>

        <div className="sidebar-footer">
          <span>PersonaForge</span>
          <span>{status === 'Error' ? '出现错误' : busy ? '生成中' : '本地运行中'}</span>
        </div>
      </aside>

      <main className="chat-panel">
        <section className="messages">
          {messages.length === 0 ? (
            <OpeningMessage persona={selectedPersona} />
          ) : (
            messages.map((message) => (
              <ChatBubble
                key={message.id}
                message={message}
                persona={selectedPersona}
                onOpenTrace={openTrace}
                showTrace={developerMode}
              />
            ))
          )}
          {liveStatus ? <LiveStatus persona={selectedPersona} label={liveStatus} /> : null}
        </section>

        <div className="composer-area">
          <WriterModeSelector
            value={writerPrompt}
            onChange={setWriterPrompt}
            personaPackAvailable={Boolean(selectedPersona?.persona_pack_available)}
            disabled={busy}
          />
          {messages.length === 0 && suggestions.length ? (
            <SuggestionChips suggestions={suggestions} onPickSuggestion={setInput} />
          ) : null}
          <form className="composer" onSubmit={handleSubmit}>
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="例如：如何看待女生常说的配得感？"
            />
            <button type="submit" disabled={!canSend}>
              {busy ? '生成中' : '发送'}
            </button>
          </form>
        </div>
      </main>
      <TraceDrawer
        open={traceOpen}
        trace={trace}
        loading={traceLoading}
        error={traceError}
        onClose={() => setTraceOpen(false)}
      />
      <AddAuthorModal
        open={addAuthorOpen}
        jobs={authorJobs}
        onClose={() => setAddAuthorOpen(false)}
        onJobCreated={upsertAuthorJob}
        onReady={(nextAuthor) => {
          setAddAuthorOpen(false);
          setAuthor(nextAuthor);
        }}
      />
    </div>
  );
}

function WriterModeSelector({
  value,
  onChange,
  personaPackAvailable,
  disabled
}: {
  value: WriterPrompt;
  onChange: (value: WriterPrompt) => void;
  personaPackAvailable: boolean;
  disabled: boolean;
}) {
  const modes: Array<{
    value: WriterPrompt;
    label: string;
    title: string;
    available: boolean;
  }> = [
    {
      value: 'current',
      label: '定向提示',
      title: '使用当前人工调优的 Writer 提示词',
      available: true
    },
    {
      value: 'strong_identity',
      label: '强身份',
      title: '仅使用通用强身份提示与 RAG20',
      available: true
    },
    {
      value: 'persona_pack',
      label: 'Persona Pack',
      title: personaPackAvailable
        ? '在强身份模式上加入证据化作者画像'
        : '这个作者还没有 Persona Pack',
      available: personaPackAvailable
    }
  ];

  return (
    <div className="writer-mode-bar">
      <span className="writer-mode-label">回答模式</span>
      <div className="writer-mode-segments" role="radiogroup" aria-label="回答模式">
        {modes.map((mode) => (
          <button
            key={mode.value}
            className={value === mode.value ? 'active' : ''}
            type="button"
            role="radio"
            aria-checked={value === mode.value}
            disabled={disabled || !mode.available}
            title={mode.title}
            onClick={() => onChange(mode.value)}
          >
            {mode.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function OpeningMessage({ persona }: { persona: PersonaInfo | null }) {
  return (
    <div className="opening-wrap">
      <article className="chat-row from-persona opening-message">
        <Avatar label={persona?.display_name || 'PF'} src={persona?.avatar_url || undefined} />
        <div className="bubble-stack">
          <div className="chat-bubble">
            <div className="message-text">{OPENING_LINE}</div>
          </div>
        </div>
      </article>
    </div>
  );
}

function LiveStatus({ persona, label }: { persona: PersonaInfo | null; label: string }) {
  return (
    <article className="live-status-row" aria-live="polite">
      <Avatar label={persona?.display_name || 'PF'} src={persona?.avatar_url || undefined} />
      <span className="live-status-text">{label}</span>
    </article>
  );
}

function SuggestionChips({
  suggestions,
  onPickSuggestion
}: {
  suggestions: string[];
  onPickSuggestion: (question: string) => void;
}) {
  return (
    <div className="suggestion-list" aria-label="建议问题">
      {suggestions.slice(0, 4).map((item) => (
        <button className="suggestion-chip" type="button" key={item} onClick={() => onPickSuggestion(item)}>
          {item}
        </button>
      ))}
    </div>
  );
}

function ChatBubble({
  message,
  persona,
  onOpenTrace,
  showTrace
}: {
  message: Message;
  persona: PersonaInfo | null;
  onOpenTrace: (traceId: string) => void;
  showTrace: boolean;
}) {
  const isUser = message.role === 'user';
  const isError = message.role === 'error';
  return (
    <article className={`chat-row ${isUser ? 'from-user' : 'from-persona'} ${isError ? 'error' : ''}`}>
      {!isUser ? <Avatar label={persona?.display_name || 'PF'} src={persona?.avatar_url || undefined} /> : null}
      <div className="bubble-stack">
        <div className="chat-bubble">
          <CopyButton text={message.text} />
          <div className="message-text">{message.text}</div>
          {message.sources ? <Sources sources={message.sources} /> : null}
        </div>
        {!isUser && !isError && message.traceId && showTrace ? (
          <button className="trace-button" type="button" onClick={() => onOpenTrace(message.traceId || '')}>
            <Activity size={14} />
            查看过程
          </button>
        ) : null}
      </div>
      {isUser ? <Avatar label={USER_AVATAR} /> : null}
    </article>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      setCopied(false);
    }
  }
  return (
    <button className="copy-button" type="button" title="复制" onClick={copy}>
      {copied ? <Check size={14} /> : <Copy size={14} />}
    </button>
  );
}

function Avatar({ label, src }: { label: string; src?: string }) {
  const initials = label.trim().slice(0, 2).toUpperCase() || 'PF';
  if (src) {
    return <img className="avatar" src={src} alt={label} />;
  }
  return <div className="avatar avatar-fallback">{initials}</div>;
}

function Sources({ sources }: { sources: Source[] }) {
  return (
    <details className="sources">
      <summary>引用来源 · {sources.length} 篇</summary>
      <ol className="source-list">
        {sources.map((source) => (
          <li key={`${source.rank}-${source.parent_id}`}>
            {source.url ? (
              <a href={source.url} target="_blank" rel="noreferrer">
                <span>{source.title || '查看原文'}</span>
                <ExternalLink size={13} aria-hidden="true" />
              </a>
            ) : (
              <span>{source.title || source.parent_id}</span>
            )}
          </li>
        ))}
      </ol>
    </details>
  );
}

function TraceDrawer({
  open,
  trace,
  loading,
  error,
  onClose
}: {
  open: boolean;
  trace: TracePayload | null;
  loading: boolean;
  error: string;
  onClose: () => void;
}) {
  if (!open) return null;
  const understanding = trace?.query_understanding;
  const searchPlan = understanding?.trace?.search_plan;
  const searchResults = understanding?.trace?.search_results || [];
  const retrieval = trace?.retrieval;
  const writer = trace?.writer;
  const generation = trace?.generation;

  return (
    <div className="trace-overlay" role="presentation" onMouseDown={onClose}>
      <aside
        className="trace-drawer"
        role="dialog"
        aria-modal="true"
        aria-label="本次回答的运行过程"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="trace-header">
          <div>
            <div className="trace-kicker">运行过程</div>
            <h2>这次回答是怎么来的</h2>
          </div>
          <button className="trace-close" type="button" title="关闭" onClick={onClose}>
            <X size={18} />
          </button>
        </header>

        <div className="trace-body">
          {loading ? <div className="trace-state">正在读取本地 trace...</div> : null}
          {error ? <div className="trace-error">{error}</div> : null}
          {trace ? (
            <>
              <section className="trace-overview">
                <div className="trace-status">
                  <Activity size={15} />
                  <span>{trace.status === 'completed' ? '已完成' : trace.status === 'failed' ? '运行失败' : '准备中'}</span>
                </div>
                <p>{trace.input.query}</p>
                <div className="trace-stats">
                  <span>
                    <Clock3 size={13} />
                    {formatDuration(trace.timing?.total_duration_ms)}
                  </span>
                  <span>{trace.input.query_mode === 'grounded' ? '联网理解 + RAG' : '直接 RAG'}</span>
                </div>
              </section>

              <section className="trace-timeline" aria-label="节点时间线">
                <div className="trace-timeline-heading">
                  <span>节点时间线</span>
                  <small>{trace.capture?.mode === 'full' ? '完整本地记录' : '摘要记录'}</small>
                </div>
                {trace.stages?.length ? (
                  trace.stages.map((stage) => <TraceStageRow key={`${stage.order}-${stage.id}`} stage={stage} />)
                ) : (
                  <div className="trace-state">这是旧版 trace，尚未记录细分节点。</div>
                )}
              </section>

              <details className="trace-section" open>
                <summary>题目理解与检索改写</summary>
                <div className="trace-section-body">
                  <TraceFact label="是否联网" value={searchPlan ? (searchPlan.needs_web ? '需要' : '不需要') : '未启用'} />
                  <TraceFact label="本阶段耗时" value={formatDuration(understanding?.duration_ms)} />
                  {searchPlan?.search_queries?.length ? (
                    <TraceList label="搜索词" items={searchPlan.search_queries} />
                  ) : null}
                  {understanding?.objective_background ? (
                    <div className="trace-background">
                      <span>客观背景</span>
                      <p>{understanding.objective_background}</p>
                    </div>
                  ) : null}
                  {searchResults.length ? (
                    <div className="trace-search-results">
                      <span>联网来源</span>
                      {searchResults.slice(0, 5).map((item) => (
                        <a href={item.url} key={`${item.query}-${item.url}`} target="_blank" rel="noreferrer">
                          {item.title || item.url}
                        </a>
                      ))}
                    </div>
                  ) : null}
                  <TraceQueryList queries={retrieval?.retrieval_queries || []} />
                </div>
              </details>

              <details className="trace-section" open>
                <summary>检索与 Parent 聚合</summary>
                <div className="trace-section-body">
                  <TraceFact label="检索耗时" value={formatDuration(retrieval?.duration_ms)} />
                  <TraceFact label="最终回填" value={`${retrieval?.parents.length || 0} 篇作者历史内容`} />
                  <div className="trace-parent-list">
                    {(retrieval?.parents || []).map((parent) => (
                      <div className="trace-parent" key={parent.parent_id}>
                        <span className="trace-rank">{parent.rank}</span>
                        <div>
                          {parent.url ? (
                            <a className="trace-parent-link" href={parent.url} target="_blank" rel="noreferrer">
                              <strong>{parent.title || parent.parent_id}</strong>
                              <ExternalLink size={12} aria-hidden="true" />
                            </a>
                          ) : (
                            <strong>{parent.title || parent.parent_id}</strong>
                          )}
                          <small>{parent.first_hits.map((hit) => `${hit.route} #${hit.rank}`).join(' · ')}</small>
                        </div>
                      </div>
                    ))}
                  </div>
                  {retrieval?.routes ? <TraceRouteHits routes={retrieval.routes} /> : null}
                </div>
              </details>

              <details className="trace-section">
                <summary>写作与生成</summary>
                <div className="trace-section-body">
                  <TraceFact label="Writer 变体" value={writer?.variant || '未知'} />
                  <TraceFact label="Writer 上下文" value={`${writer?.total_characters || 0} 字符`} />
                  <TraceFact label="生成模型" value={generation?.model || generation?.provider || '未知'} />
                  <TraceFact label="生成参数" value={`temperature ${generation?.temperature ?? '-'} · 上限 ${generation?.max_tokens ?? '-'} tokens`} />
                  <TraceFact label="生成耗时" value={formatDuration(generation?.duration_ms)} />
                  <TraceFact label="输出长度" value={`${generation?.answer_characters || 0} 字符`} />
                  {trace.error ? <div className="trace-error">{trace.error.type}: {trace.error.message}</div> : null}
                </div>
              </details>

              <div className="trace-id">{trace.trace_id}</div>
            </>
          ) : null}
        </div>
      </aside>
    </div>
  );
}

function TraceStageRow({ stage }: { stage: TraceStage }) {
  const usage = stage.usage;
  const tokenText = usage
    ? usage.source === 'provider'
      ? `${usage.total_tokens ?? '-'} tokens`
      : `约 ${usage.estimated_tokens ?? '-'} tokens`
    : null;
  return (
    <details className={`trace-stage status-${stage.status}`}>
      <summary>
        <span className="trace-stage-marker" aria-hidden="true" />
        <strong>{stage.label}</strong>
        <span>{formatDuration(stage.duration_ms)}</span>
      </summary>
      <div className="trace-stage-detail">
        {tokenText ? (
          <p>
            Token：{tokenText}
            {usage?.source === 'estimated' ? '（估算）' : ''}
          </p>
        ) : null}
        {stage.details ? <pre>{JSON.stringify(stage.details, null, 2)}</pre> : null}
        {usage?.note ? <small>{usage.note}</small> : null}
      </div>
    </details>
  );
}

function TraceFact({ label, value }: { label: string; value: string }) {
  return (
    <div className="trace-fact">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function TraceList({ label, items }: { label: string; items: string[] }) {
  return (
    <div className="trace-list-wrap">
      <span>{label}</span>
      <ul className="trace-list">
        {items.map((item) => <li key={item}>{item}</li>)}
      </ul>
    </div>
  );
}

function TraceQueryList({ queries }: { queries: Array<{ route: string; query: string }> }) {
  if (!queries.length) return null;
  return (
    <div className="trace-query-list">
      <span>检索 query</span>
      {queries.map((item) => (
        <div className="trace-query" key={item.route}>
          <small>{item.route}</small>
          <p>{item.query}</p>
        </div>
      ))}
    </div>
  );
}

function TraceRouteHits({ routes }: { routes: Record<string, Array<{ rank: number; title: string; node_type: string }>> }) {
  return (
    <details className="trace-route-hits">
      <summary>查看各路 child 命中</summary>
      <div>
        {Object.entries(routes).map(([route, hits]) => (
          <section key={route}>
            <strong>{route}</strong>
            <span>{hits.length} 个节点</span>
            <ol>
              {hits.slice(0, 8).map((hit) => (
                <li key={`${route}-${hit.rank}`}>#{hit.rank} · {hit.node_type} · {hit.title}</li>
              ))}
            </ol>
          </section>
        ))}
      </div>
    </details>
  );
}

function formatDuration(value?: number): string {
  if (value === undefined || value === null) return '-';
  if (value < 1000) return `${value} ms`;
  return `${(value / 1000).toFixed(1)} s`;
}

function makeId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}
