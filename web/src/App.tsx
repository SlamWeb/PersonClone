import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import { Activity, Brain, Check, Clock3, Copy, ExternalLink, LogOut, Pencil, Pin, Plus, Save, Settings2, SlidersHorizontal, Trash2, X } from 'lucide-react';
import {
  AuthState,
  AuthorJob,
  cancelAuthorJob,
  ChatMessage as ApiChatMessage,
  ChatSessionSummary,
  deleteSession,
  fetchAuthorJobs,
  fetchAuthState,
  fetchMemories,
  fetchMemorySettings,
  fetchPersonas,
  fetchSession,
  fetchSessions,
  fetchSuggestions,
  fetchTurn,
  fetchTrace,
  logoutAuth,
  forgetMemory,
  clearMemories,
  PersonaInfo,
  retryAuthorJob,
  retryTurn,
  Source,
  streamChat,
  TraceStage,
  TracePayload,
  UserMemory,
  UserMemorySettings,
  updateMemory,
  updateMemorySettings
} from './api';
import { AuthScreen } from './AuthScreen';
import {
  AddAuthorModal,
  AuthorLibraryPage,
  PersonaSwitcher
} from './AuthorManager';

type Message = {
  id: string;
  role: 'user' | 'assistant' | 'error';
  text: string;
  status?: 'queued' | 'running' | 'completed' | 'failed' | 'interrupted';
  sources?: Source[] | null;
  traceId?: string | null;
  turnId?: string | null;
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
  const [authState, setAuthState] = useState<AuthState | null>(null);
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
  const [runningConversations, setRunningConversations] = useState<Record<string, string>>({});
  const [pendingNewConversation, setPendingNewConversation] = useState(false);
  const [authorJobs, setAuthorJobs] = useState<AuthorJob[]>([]);
  const [authorLibraryOpen, setAuthorLibraryOpen] = useState(() => window.location.pathname === '/authors');
  const [addAuthorOpen, setAddAuthorOpen] = useState(false);
  const [memoryOpen, setMemoryOpen] = useState(false);
  const currentSessionRef = useRef<string | null>(null);
  const currentAuthorRef = useRef('');
  const pollingTurnsRef = useRef<Set<string>>(new Set());
  const messagesRef = useRef<HTMLElement | null>(null);
  const keepMessagesPinnedRef = useRef(true);

  const selectedPersona = useMemo(
    () => personas.find((item) => item.author === author) || null,
    [personas, author]
  );
  const currentRunLabel = currentSessionId
    ? runningConversations[currentSessionId] || null
    : pendingNewConversation
      ? '正在创建回答任务'
      : null;
  const busy = Boolean(currentRunLabel);
  const canSend = useMemo(() => Boolean(author && input.trim() && !busy), [author, input, busy]);

  useEffect(() => {
    fetchAuthState()
      .then(setAuthState)
      .catch(() => setAuthState({ configured: true, authenticated: false, user: null }));
    function handleExpiredSession() {
      setAuthState((current) => ({
        configured: current?.configured ?? true,
        authenticated: false,
        user: null
      }));
    }
    window.addEventListener('pf-auth-expired', handleExpiredSession);
    return () => window.removeEventListener('pf-auth-expired', handleExpiredSession);
  }, []);

  useEffect(() => {
    currentSessionRef.current = currentSessionId;
  }, [currentSessionId]);

  useEffect(() => {
    currentAuthorRef.current = author;
  }, [author]);

  useEffect(() => {
    if (!keepMessagesPinnedRef.current) return;
    const node = messagesRef.current;
    if (!node) return;
    const frame = window.requestAnimationFrame(() => {
      node.scrollTo({ top: node.scrollHeight, behavior: 'auto' });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [messages, liveStatus, currentSessionId]);

  useEffect(() => {
    if (!authState?.authenticated) return;
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
  }, [authState?.authenticated]);

  useEffect(() => {
    if (!authState?.authenticated) return;
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
  }, [authState?.authenticated]);

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
    if (!author) return;
    try {
      keepMessagesPinnedRef.current = true;
      const session = await fetchSession(author, sessionId);
      currentSessionRef.current = session.id;
      setCurrentSessionId(session.id);
      setMessages(session.messages.map(chatMessageToMessage));
      const pending = session.messages.find(
        (message) =>
          message.role === 'assistant' &&
          (message.status === 'queued' || message.status === 'running') &&
          message.turn_id
      );
      if (pending?.turn_id) {
        setRunningConversations((items) => ({
          ...items,
          [session.id]: '正在恢复生成进度'
        }));
        void watchTurn(author, session.id, pending.turn_id);
      } else {
        setLiveStatus(null);
      }
      setStatus(`Ready: ${author}`);
    } catch (error) {
      setStatus(String((error as Error).message || error));
    }
  }

  async function removeSession(sessionId: string) {
    if (!author) return;
    if (runningConversations[sessionId]) {
      setStatus('这段对话仍在生成，请完成后再删除。');
      return;
    }
    await deleteSession(author, sessionId);
    if (currentSessionId === sessionId) {
      newChat();
    }
    await refreshSessions(author);
  }

  function newChat() {
    keepMessagesPinnedRef.current = true;
    currentSessionRef.current = null;
    setCurrentSessionId(null);
    setMessages([]);
    setInput('');
    setTrace(null);
    setTraceOpen(false);
    setLiveStatus(null);
  }

  async function watchTurn(targetAuthor: string, sessionId: string, turnId: string) {
    if (pollingTurnsRef.current.has(turnId)) return;
    pollingTurnsRef.current.add(turnId);
    let terminal = false;
    let failures = 0;
    try {
      while (!terminal && failures < 3) {
        try {
          const turn = await fetchTurn(turnId);
          failures = 0;
          terminal = turn.status === 'completed' || turn.status === 'failed' || turn.status === 'interrupted';
          if (terminal) {
            setRunningConversations((items) => omitKey(items, sessionId));
          } else {
            setRunningConversations((items) => ({ ...items, [sessionId]: turn.label }));
          }

          if (currentAuthorRef.current === targetAuthor && currentSessionRef.current === sessionId) {
            const session = await fetchSession(targetAuthor, sessionId);
            setMessages(session.messages.map(chatMessageToMessage));
            setLiveStatus(terminal ? null : turn.label);
          }
          if (!terminal) {
            await sleep(700);
          }
        } catch {
          failures += 1;
          if (failures < 3) await sleep(1000);
        }
      }
    } finally {
      pollingTurnsRef.current.delete(turnId);
      if (terminal) {
        await refreshSessions(targetAuthor);
      }
    }
  }

  async function retryFailedTurn(turnId: string) {
    if (!author || !currentSessionId) return;
    try {
      const turn = await retryTurn(turnId);
      setRunningConversations((items) => ({
        ...items,
        [turn.conversation_id]: turn.label
      }));
      setMessages((items) =>
        items.map((message) =>
          message.turnId === turnId
            ? { ...message, text: '', status: 'queued', traceId: null, sources: null }
            : message
        )
      );
      setLiveStatus(turn.label);
      void watchTurn(author, turn.conversation_id, turn.id);
    } catch (error) {
      setStatus(String((error as Error).message || error));
    }
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
    if (!text || !author || busy) return;

    const userId = makeId();
    const assistantId = makeId();
    const requestAuthor = author;
    const submissionSessionId = currentSessionId;
    let requestSessionId = currentSessionId;
    let requestTurnId = '';
    const isVisible = () =>
      currentAuthorRef.current === requestAuthor &&
      currentSessionRef.current === requestSessionId;

    keepMessagesPinnedRef.current = true;
    setMessages((items) => [
      ...items,
      { id: userId, role: 'user', text, status: 'completed' }
    ]);
    setInput('');
    if (!submissionSessionId) setPendingNewConversation(true);
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
          onAccepted: (payload) => {
            requestSessionId = payload.session_id;
            requestTurnId = payload.turn_id;
            setPendingNewConversation(false);
            setRunningConversations((items) => ({
              ...items,
              [payload.session_id]: payload.label || '正在等待生成'
            }));
            if (
              currentAuthorRef.current === requestAuthor &&
              currentSessionRef.current === submissionSessionId
            ) {
              currentSessionRef.current = payload.session_id;
              setCurrentSessionId(payload.session_id);
              setLiveStatus(payload.label || '正在等待生成');
              setMessages((items) =>
                items.some((message) => message.turnId === payload.turn_id)
                  ? items
                  : [
                      ...items,
                      {
                        id: assistantId,
                        role: 'assistant',
                        text: '',
                        status: 'queued',
                        turnId: payload.turn_id
                      }
                    ]
              );
            }
          },
          onMeta: (payload) => {
            const sessionId = String(payload.session_id || '');
            const traceId = String(payload.trace_id || '');
            if (sessionId) requestSessionId = sessionId;
            if (isVisible() && traceId) {
              setMessages((items) =>
                items.map((message) =>
                  message.turnId === requestTurnId
                    ? { ...message, traceId }
                    : message
                )
              );
            }
          },
          onStatus: (payload) => {
            if (payload.label && requestSessionId) {
              setRunningConversations((items) => ({
                ...items,
                [requestSessionId || '']: payload.label
              }));
            }
            if (payload.label && isVisible()) setLiveStatus(payload.label);
          },
          onToken: (token) => {
            if (isVisible()) {
              setLiveStatus(null);
              setMessages((items) =>
                appendAssistantToken(items, requestTurnId, assistantId, token)
              );
            }
          },
          onDone: async (payload) => {
            requestSessionId = payload.session_id;
            requestTurnId = payload.turn_id || requestTurnId;
            setRunningConversations((items) => omitKey(items, payload.session_id));
            if (isVisible()) {
              setLiveStatus(null);
              setMessages((items) =>
                finishAssistantMessage(
                  items,
                  requestTurnId,
                  assistantId,
                  payload.answer,
                  payload.sources,
                  payload.trace_id
                )
              );
              setStatus(`Ready: ${requestAuthor}`);
            }
            await refreshSessions(requestAuthor);
          },
          onError: (message) => {
            throw new Error(message);
          }
        }
      );
    } catch (error) {
      setPendingNewConversation(false);
      if (requestSessionId) {
        setRunningConversations((items) => omitKey(items, requestSessionId || ''));
      }
      if (requestSessionId && isVisible()) {
        try {
          const session = await fetchSession(requestAuthor, requestSessionId);
          setMessages(session.messages.map(chatMessageToMessage));
        } catch {
          setMessages((items) => [
            ...items.filter((message) => message.id !== assistantId),
            { id: makeId(), role: 'error', text: String((error as Error).message || error) }
          ]);
        }
      } else if (currentAuthorRef.current === requestAuthor && currentSessionRef.current === submissionSessionId) {
        setMessages((items) => [
          ...items.filter((message) => message.id !== assistantId),
          { id: makeId(), role: 'error', text: String((error as Error).message || error) }
        ]);
      }
      setLiveStatus(null);
      setStatus('Error');
    }
  }

  async function signOut() {
    try {
      await logoutAuth();
    } finally {
      setAuthState({ configured: true, authenticated: false, user: null });
      setPersonas([]);
      setAuthor('');
      setSessions([]);
      setMessages([]);
      setCurrentSessionId(null);
      setRunningConversations({});
    }
  }

  if (authState === null) {
    return (
      <main className="auth-page">
        <div className="auth-loading">正在检查登录状态</div>
      </main>
    );
  }

  if (!authState.authenticated) {
    return <AuthScreen configured={authState.configured} onAuthenticated={setAuthState} />;
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
                    <small>
                      {runningConversations[session.id]
                        ? '正在生成'
                        : `${session.message_count} 条消息`}
                    </small>
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
          <button
            className="memory-entry"
            type="button"
            title="管理我的记忆"
            onClick={() => setMemoryOpen(true)}
          >
            <Brain size={14} />
            <span className="signed-in-user">{authState.user?.display_name || authState.user?.username}</span>
          </button>
          <button className="logout-button" type="button" title="退出登录" onClick={signOut}>
            <LogOut size={14} />
          </button>
        </div>
      </aside>

      <main className="chat-panel">
        <section
          className="messages"
          ref={messagesRef}
          onScroll={(event) => {
            const node = event.currentTarget;
            keepMessagesPinnedRef.current =
              node.scrollHeight - node.scrollTop - node.clientHeight < 120;
          }}
        >
          {messages.length === 0 ? (
            <OpeningMessage persona={selectedPersona} />
          ) : (
            messages.map((message) => (
              <ChatBubble
                key={message.id}
                message={message}
                persona={selectedPersona}
                onOpenTrace={openTrace}
                onRetryTurn={retryFailedTurn}
                showTrace={developerMode}
              />
            ))
          )}
          {liveStatus || currentRunLabel ? (
            <LiveStatus persona={selectedPersona} label={liveStatus || currentRunLabel || '正在生成'} />
          ) : null}
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
      <MemoryDrawer open={memoryOpen} onClose={() => setMemoryOpen(false)} />
    </div>
  );
}

function MemoryDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [memories, setMemories] = useState<UserMemory[]>([]);
  const [settings, setSettings] = useState<UserMemorySettings>({ enabled: true, auto_write: true });
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function refresh() {
    setLoading(true);
    setError('');
    try {
      const [items, nextSettings] = await Promise.all([fetchMemories(), fetchMemorySettings()]);
      setMemories(items);
      setSettings(nextSettings);
    } catch (reason) {
      setError(String((reason as Error).message || reason));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (open) void refresh();
  }, [open]);

  if (!open) return null;

  async function patchSettings(patch: Partial<UserMemorySettings>) {
    try {
      setSettings(await updateMemorySettings(patch));
    } catch (reason) {
      setError(String((reason as Error).message || reason));
    }
  }

  async function togglePinned(memory: UserMemory) {
    try {
      const updated = await updateMemory(memory.id, { pinned: !memory.pinned });
      setMemories((items) => items.map((item) => item.id === memory.id ? updated : item));
    } catch (reason) {
      setError(String((reason as Error).message || reason));
    }
  }

  async function saveCorrection(memory: UserMemory) {
    if (!draft.trim()) return;
    try {
      const updated = await updateMemory(memory.id, { content: draft.trim() });
      setMemories((items) => items.map((item) => item.id === memory.id ? updated : item));
      setEditingId(null);
    } catch (reason) {
      setError(String((reason as Error).message || reason));
    }
  }

  async function forget(memoryId: string) {
    try {
      await forgetMemory(memoryId);
      setMemories((items) => items.filter((item) => item.id !== memoryId));
    } catch (reason) {
      setError(String((reason as Error).message || reason));
    }
  }

  async function clearAll() {
    if (!window.confirm('清空后，已有对话仍保留，但这些长期记忆不会再被召回。确定继续吗？')) return;
    try {
      await clearMemories();
      setMemories([]);
    } catch (reason) {
      setError(String((reason as Error).message || reason));
    }
  }

  return (
    <div className="trace-overlay" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <aside className="memory-drawer" aria-label="我的记忆">
        <header className="memory-header">
          <div>
            <span className="memory-eyebrow">跨会话上下文</span>
            <h2>我的记忆</h2>
            <p>只保存你的信息，不会把作者回答写成你的事实。</p>
          </div>
          <button className="icon-button" type="button" onClick={onClose} title="关闭"><X size={18} /></button>
        </header>

        <section className="memory-settings">
          <label>
            <span><strong>使用记忆</strong><small>在相关的新对话中召回</small></span>
            <input type="checkbox" checked={settings.enabled} onChange={(event) => patchSettings({ enabled: event.target.checked })} />
          </label>
          <label>
            <span><strong>自动形成记忆</strong><small>回答完成后异步审查写入</small></span>
            <input type="checkbox" checked={settings.auto_write} disabled={!settings.enabled} onChange={(event) => patchSettings({ auto_write: event.target.checked })} />
          </label>
        </section>

        <div className="memory-toolbar">
          <span>{memories.length} 条有效记忆</span>
          {memories.length ? <button type="button" onClick={clearAll}>全部清空</button> : null}
        </div>
        {error ? <div className="memory-error">{error}</div> : null}
        <section className="memory-list">
          {loading ? <div className="memory-empty">正在读取记忆</div> : null}
          {!loading && memories.length === 0 ? (
            <div className="memory-empty"><Brain size={22} /><strong>还没有长期记忆</strong><span>有价值的用户信息会在回答完成后异步形成。</span></div>
          ) : null}
          {memories.map((memory) => (
            <article className="memory-item" key={memory.id}>
              <div className="memory-meta">
                <span>{memoryKindLabel(memory.kind)}</span>
                {memory.sensitivity === 'restricted' ? <span>敏感概括</span> : null}
                {memory.pinned ? <span>已置顶</span> : null}
              </div>
              {editingId === memory.id ? (
                <textarea value={draft} onChange={(event) => setDraft(event.target.value)} autoFocus />
              ) : <p>{memory.content}</p>}
              <div className="memory-actions">
                <button type="button" title={memory.pinned ? '取消置顶' : '置顶'} onClick={() => togglePinned(memory)}><Pin size={14} />{memory.pinned ? '取消置顶' : '置顶'}</button>
                {editingId === memory.id ? (
                  <button type="button" onClick={() => saveCorrection(memory)}><Save size={14} />保存修正</button>
                ) : (
                  <button type="button" onClick={() => { setEditingId(memory.id); setDraft(memory.content); }}><Pencil size={14} />纠正</button>
                )}
                <button type="button" onClick={() => forget(memory.id)}><Trash2 size={14} />遗忘</button>
              </div>
            </article>
          ))}
        </section>
      </aside>
    </div>
  );
}

function memoryKindLabel(kind: UserMemory['kind']): string {
  if (kind === 'semantic') return '个人信息';
  if (kind === 'procedural') return '协作偏好';
  return '持续事件';
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
  onRetryTurn,
  showTrace
}: {
  message: Message;
  persona: PersonaInfo | null;
  onOpenTrace: (traceId: string) => void;
  onRetryTurn: (turnId: string) => void;
  showTrace: boolean;
}) {
  const isUser = message.role === 'user';
  const isError = message.role === 'error';
  const isPending = message.status === 'queued' || message.status === 'running';
  const canRetry =
    !isUser &&
    Boolean(message.turnId) &&
    (message.status === 'failed' || message.status === 'interrupted');
  if (!isUser && isPending && !message.text) return null;
  const displayText =
    message.text ||
    (message.status === 'interrupted' ? '这次回答因服务重启而中断。' : '这次回答生成失败。');
  return (
    <article className={`chat-row ${isUser ? 'from-user' : 'from-persona'} ${isError || canRetry ? 'error' : ''}`}>
      {!isUser ? <Avatar label={persona?.display_name || 'PF'} src={persona?.avatar_url || undefined} /> : null}
      <div className="bubble-stack">
        <div className="chat-bubble">
          <CopyButton text={displayText} />
          <div className="message-text">{displayText}</div>
          {message.sources ? <Sources sources={message.sources} /> : null}
        </div>
        {canRetry ? (
          <button
            className="retry-turn-button"
            type="button"
            onClick={() => onRetryTurn(message.turnId || '')}
          >
            重新生成
          </button>
        ) : null}
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
  const turnPlanner = trace?.turn_planner;
  const conversationContext = trace?.conversation_context;
  const memoryUpdate = trace?.memory_update;
  const memoryRecall = trace?.user_memory_recall;

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
                <summary>对话决策与记忆</summary>
                <div className="trace-section-body">
                  <TraceFact
                    label="本轮类型"
                    value={formatTurnType(turnPlanner?.turn_type)}
                  />
                  <TraceFact
                    label="检索策略"
                    value={formatRetrievalPolicy(turnPlanner?.retrieval_policy)}
                  />
                  <TraceFact
                    label="回答深度"
                    value={formatResponseDepth(turnPlanner?.response_depth)}
                  />
                  {turnPlanner?.resolved_question ? (
                    <div className="trace-background">
                      <span>独立语义问题</span>
                      <p>{turnPlanner.resolved_question}</p>
                    </div>
                  ) : null}
                  <TraceFact
                    label="历史上下文"
                    value={
                      conversationContext?.used_full_short_history
                        ? '短对话全量保留'
                        : `最近 ${conversationContext?.recent_turn_ids?.length || 0} 轮 + 相关 ${conversationContext?.relevant_turn_ids?.length || 0} 轮`
                    }
                  />
                  <TraceFact
                    label="会话摘要"
                    value={
                      conversationContext?.summary_version
                        ? `v${conversationContext.summary_version}`
                        : '尚未生成'
                    }
                  />
                  <TraceFact
                    label="长期记忆召回"
                    value={
                      memoryRecall?.selected_ids?.length
                        ? `从 ${memoryRecall.candidate_ids?.length || 0} 条候选中使用 ${memoryRecall.selected_ids.length} 条`
                        : `候选 ${memoryRecall?.candidate_ids?.length || 0} 条，本轮未使用`
                    }
                  />
                  {turnPlanner?.evidence_source_turn_id ? (
                    <TraceFact
                      label="复用证据轮次"
                      value={turnPlanner.evidence_source_turn_id}
                    />
                  ) : null}
                  {memoryUpdate ? (
                    <TraceFact
                      label="记忆更新"
                      value={
                        memoryUpdate.status === 'failed'
                          ? '失败'
                          : memoryUpdate.status === 'skipped'
                            ? '本轮无需更新'
                            : `长期记忆 ${memoryUpdate.user_memory?.operations?.length || 0} 项变更 · 会话摘要${memoryUpdate.conversation_summary?.status === 'completed' ? ` v${memoryUpdate.conversation_summary.summary_version}` : '未变更'}`
                      }
                    />
                  ) : null}
                </div>
              </details>

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

function formatTurnType(value?: string): string {
  return {
    new_topic: '新话题',
    follow_up: '继续追问',
    explain_previous: '解释上一回答',
    casual: '轻量对话',
    unclear: '需要澄清'
  }[value || ''] || '旧版 Trace 未记录';
}

function formatRetrievalPolicy(value?: string): string {
  return {
    new: '重新检索',
    reuse: '复用既有证据',
    none: '无需检索'
  }[value || ''] || '未记录';
}

function formatResponseDepth(value?: string): string {
  return {
    brief: '简短',
    normal: '正常',
    deep: '深入'
  }[value || ''] || '未记录';
}

function chatMessageToMessage(message: ApiChatMessage): Message {
  return {
    id: message.id || makeId(),
    role: message.role,
    text: message.text,
    status: message.status || 'completed',
    sources: message.sources,
    traceId: message.trace_id,
    turnId: message.turn_id
  };
}

function appendAssistantToken(
  messages: Message[],
  turnId: string,
  fallbackId: string,
  token: string
): Message[] {
  const index = messages.findIndex(
    (message) => (turnId && message.turnId === turnId) || message.id === fallbackId
  );
  if (index < 0) {
    return [
      ...messages,
      {
        id: fallbackId,
        role: 'assistant',
        text: token,
        status: 'running',
        turnId: turnId || null
      }
    ];
  }
  return messages.map((message, itemIndex) =>
    itemIndex === index
      ? { ...message, text: message.text + token, status: 'running', turnId: turnId || message.turnId }
      : message
  );
}

function finishAssistantMessage(
  messages: Message[],
  turnId: string,
  fallbackId: string,
  answer: string,
  sources: Source[],
  traceId?: string
): Message[] {
  const index = messages.findIndex(
    (message) => (turnId && message.turnId === turnId) || message.id === fallbackId
  );
  const completed: Message = {
    id: index >= 0 ? messages[index].id : fallbackId,
    role: 'assistant',
    text: answer,
    status: 'completed',
    sources,
    traceId,
    turnId: turnId || null
  };
  if (index < 0) return [...messages, completed];
  return messages.map((message, itemIndex) => itemIndex === index ? completed : message);
}

function omitKey(values: Record<string, string>, key: string): Record<string, string> {
  const next = { ...values };
  delete next[key];
  return next;
}

function sleep(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function makeId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}
