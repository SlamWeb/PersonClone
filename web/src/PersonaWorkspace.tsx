import { ReactNode } from 'react';
import {
  Brain,
  Library,
  LogOut,
  Menu,
  MessageSquarePlus,
  Plus,
  SlidersHorizontal,
  Trash2,
  X
} from 'lucide-react';
import { ChatSessionSummary, PersonaInfo } from './api';

type SessionGroup = {
  label: string;
  sessions: ChatSessionSummary[];
};

export function PersonaDock({
  personas,
  selectedAuthor,
  hasActiveJobs,
  onSelect,
  onAdd,
  onManage,
  onOpenSessions
}: {
  personas: PersonaInfo[];
  selectedAuthor: string;
  hasActiveJobs: boolean;
  onSelect: (author: string) => void;
  onAdd: () => void;
  onManage: () => void;
  onOpenSessions: () => void;
}) {
  return (
    <aside className="persona-dock" aria-label="分身切换">
      <button className="dock-mobile-menu" type="button" title="打开会话" onClick={onOpenSessions}>
        <Menu size={19} />
      </button>
      <div className="persona-dock-list">
        {personas.map((persona) => (
          <button
            className={`persona-dock-item ${persona.author === selectedAuthor ? 'active' : ''}`}
            type="button"
            key={persona.author}
            title={persona.display_name}
            aria-label={`切换到 ${persona.display_name}`}
            aria-pressed={persona.author === selectedAuthor}
            onClick={() => onSelect(persona.author)}
          >
            <PersonaPortrait persona={persona} />
          </button>
        ))}
      </div>
      <div className="persona-dock-actions">
        <button className="dock-action" type="button" title="添加作者" onClick={onAdd}>
          <Plus size={18} />
          {hasActiveJobs ? <span className="dock-job-dot" aria-label="作者正在构建" /> : null}
        </button>
        <button className="dock-action" type="button" title="分身库" onClick={onManage}>
          <Library size={18} />
        </button>
      </div>
    </aside>
  );
}

export function ConversationSidebar({
  open,
  persona,
  sessions,
  currentSessionId,
  runningConversations,
  userName,
  experimentPanel,
  onClose,
  onNewChat,
  onOpenSession,
  onDeleteSession,
  onOpenMemory,
  onLogout
}: {
  open: boolean;
  persona: PersonaInfo | null;
  sessions: ChatSessionSummary[];
  currentSessionId: string | null;
  runningConversations: Record<string, string>;
  userName: string;
  experimentPanel: ReactNode;
  onClose: () => void;
  onNewChat: () => void;
  onOpenSession: (sessionId: string) => void;
  onDeleteSession: (sessionId: string) => void;
  onOpenMemory: () => void;
  onLogout: () => void;
}) {
  const groups = groupSessions(sessions);
  return (
    <>
      {open ? <button className="sidebar-scrim" type="button" aria-label="关闭会话栏" onClick={onClose} /> : null}
      <aside className={`conversation-sidebar ${open ? 'mobile-open' : ''}`}>
        <header className="persona-panel-header">
          {persona ? <PersonaPortrait persona={persona} size="large" /> : <PersonaPortraitFallback />}
          <div className="persona-panel-identity">
            <strong>{persona?.display_name || '选择作者'}</strong>
            <span><i aria-hidden="true" />已就绪</span>
          </div>
          <button className="sidebar-close" type="button" title="关闭" onClick={onClose}>
            <X size={18} />
          </button>
        </header>

        <button className="new-chat-button" type="button" onClick={onNewChat}>
          <span className="command-icon"><MessageSquarePlus size={16} /></span>
          <span>开始新对话</span>
        </button>

        <section className="session-section">
          <div className="section-label">对话</div>
          <div className="session-list">
            {!sessions.length ? <div className="muted-empty">还没有对话</div> : null}
            {groups.map((group) => (
              <div className="session-group" key={group.label}>
                <div className="session-group-label">{group.label}</div>
                {group.sessions.map((session) => (
                  <div className={`session-item ${session.id === currentSessionId ? 'active' : ''}`} key={session.id}>
                    <button className="session-open" type="button" onClick={() => onOpenSession(session.id)}>
                      <span>{session.title}</span>
                      {runningConversations[session.id] ? <small>正在生成</small> : null}
                    </button>
                    <button
                      className="delete-session"
                      type="button"
                      title="删除会话"
                      onClick={() => onDeleteSession(session.id)}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </section>

        <div className="persona-panel-tools">
          <button className="panel-tool" type="button" onClick={onOpenMemory}>
            <Brain size={16} />
            <span>它记住的你</span>
          </button>
          <details className="experiment-panel">
            <summary>
              <SlidersHorizontal size={16} />
              <span>实验台</span>
            </summary>
            <div className="experiment-panel-body">{experimentPanel}</div>
          </details>
        </div>

        <footer className="sidebar-footer">
          <span className="signed-in-user">{userName}</span>
          <button className="logout-button" type="button" title="退出登录" onClick={onLogout}>
            <LogOut size={15} />
          </button>
        </footer>
      </aside>
    </>
  );
}

export function personaTheme(persona: PersonaInfo | null): Record<string, string> {
  const palettes = [
    ['#4f766d', '#edf4f1'],
    ['#58739a', '#eef3f8'],
    ['#806a8e', '#f4f0f6'],
    ['#8a6b55', '#f6f1ed'],
    ['#55748a', '#eef4f7'],
    ['#756f55', '#f4f3ec']
  ];
  const identity = persona?.avatar_url || persona?.author || 'personaforge';
  let hash = 0;
  for (let index = 0; index < identity.length; index += 1) {
    hash = ((hash << 5) - hash + identity.charCodeAt(index)) | 0;
  }
  const [accent, soft] = palettes[Math.abs(hash) % palettes.length];
  return {
    '--persona-accent': accent,
    '--persona-accent-soft': soft
  };
}

function PersonaPortrait({ persona, size = 'normal' }: { persona: PersonaInfo; size?: 'normal' | 'large' }) {
  const initials = persona.display_name.trim().slice(0, 2).toUpperCase() || 'PF';
  return persona.avatar_url ? (
    <img className={`persona-portrait ${size}`} src={persona.avatar_url} alt={persona.display_name} />
  ) : (
    <span className={`persona-portrait persona-portrait-fallback ${size}`}>{initials}</span>
  );
}

function PersonaPortraitFallback() {
  return <span className="persona-portrait persona-portrait-fallback large">PF</span>;
}

function groupSessions(sessions: ChatSessionSummary[]): SessionGroup[] {
  const now = new Date();
  const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const startWeek = startToday - 6 * 24 * 60 * 60 * 1000;
  const values: SessionGroup[] = [
    { label: '今天', sessions: [] },
    { label: '最近 7 天', sessions: [] },
    { label: '更早', sessions: [] }
  ];
  sessions.forEach((session) => {
    const timestamp = new Date(session.updated_at).getTime();
    if (timestamp >= startToday) values[0].sessions.push(session);
    else if (timestamp >= startWeek) values[1].sessions.push(session);
    else values[2].sessions.push(session);
  });
  return values.filter((group) => group.sessions.length);
}
