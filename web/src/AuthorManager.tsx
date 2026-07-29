import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowLeft,
  Check,
  ChevronDown,
  ExternalLink,
  Library,
  Plus,
  RefreshCw,
  X
} from 'lucide-react';
import {
  AuthorJob,
  AuthorPreview,
  createAuthorJob,
  PersonaInfo,
  previewAuthor
} from './api';

type ContentKind = 'answer' | 'article' | 'pin';

export function PersonaSwitcher({
  personas,
  jobs,
  selectedPersona,
  onSelect,
  onAdd,
  onManage
}: {
  personas: PersonaInfo[];
  jobs: AuthorJob[];
  selectedPersona: PersonaInfo | null;
  onSelect: (author: string) => void;
  onAdd: () => void;
  onManage: () => void;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const activeJobs = latestJobsByAuthor(jobs).filter(
    (job) => isActiveJob(job) && !personas.some((persona) => persona.author === job.author)
  );

  useEffect(() => {
    function closeOnOutside(event: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === 'Escape') setOpen(false);
    }
    document.addEventListener('mousedown', closeOnOutside);
    document.addEventListener('keydown', closeOnEscape);
    return () => {
      document.removeEventListener('mousedown', closeOnOutside);
      document.removeEventListener('keydown', closeOnEscape);
    };
  }, []);

  return (
    <div className="persona-switcher" ref={rootRef}>
      <button
        className="persona-switcher-trigger"
        type="button"
        aria-expanded={open}
        aria-haspopup="menu"
        onClick={() => setOpen((value) => !value)}
      >
        <AuthorAvatar
          label={selectedPersona?.display_name || 'PersonaForge'}
          src={selectedPersona?.avatar_url}
          size="medium"
        />
        <span>{selectedPersona?.display_name || '选择作者'}</span>
        <ChevronDown size={16} aria-hidden="true" />
      </button>

      {open ? (
        <div className="persona-popover" role="menu">
          <div className="persona-popover-label">作者</div>
          <div className="persona-popover-list">
            {personas.map((persona) => (
              <button
                className={`persona-option ${selectedPersona?.author === persona.author ? 'active' : ''}`}
                type="button"
                role="menuitem"
                key={persona.author}
                onClick={() => {
                  onSelect(persona.author);
                  setOpen(false);
                }}
              >
                <AuthorAvatar label={persona.display_name} src={persona.avatar_url} size="small" />
                <span>{persona.display_name}</span>
                {selectedPersona?.author === persona.author ? <Check size={15} /> : null}
              </button>
            ))}
            {activeJobs.map((job) => (
              <div className="persona-option pending" key={job.id}>
                <AuthorAvatar label={job.display_name || job.author} src={job.avatar_url} size="small" />
                <span>
                  <strong>{job.display_name || job.author}</strong>
                  <small>{job.label}</small>
                </span>
                <span className="status-dot running" aria-hidden="true" />
              </div>
            ))}
            {!personas.length && !activeJobs.length ? (
              <div className="persona-popover-empty">还没有可用作者</div>
            ) : null}
          </div>
          <div className="persona-popover-actions">
            <button
              type="button"
              onClick={() => {
                onAdd();
                setOpen(false);
              }}
            >
              <Plus size={16} />
              添加作者
            </button>
            <button
              type="button"
              onClick={() => {
                onManage();
                setOpen(false);
              }}
            >
              <Library size={16} />
              管理作者库
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export function AuthorLibraryPage({
  personas,
  jobs,
  onBack,
  onAdd,
  onSelect,
  onCancel,
  onRetry
}: {
  personas: PersonaInfo[];
  jobs: AuthorJob[];
  onBack: () => void;
  onAdd: () => void;
  onSelect: (author: string) => void;
  onCancel: (jobId: string) => void;
  onRetry: (jobId: string) => void;
}) {
  const latestJobs = latestJobsByAuthor(jobs);
  const personaAuthors = new Set(personas.map((persona) => persona.author));
  const jobOnly = latestJobs.filter((job) => !personaAuthors.has(job.author));

  return (
    <main className="author-library-page">
      <header className="author-library-header">
        <div className="author-library-heading">
          <button className="icon-button" type="button" title="返回聊天" onClick={onBack}>
            <ArrowLeft size={19} />
          </button>
          <div>
            <h1>作者库</h1>
            <p>管理已创建的作者与后台构建任务</p>
          </div>
        </div>
        <button className="primary-command" type="button" onClick={onAdd}>
          <Plus size={17} />
          添加作者
        </button>
      </header>

      <section className="author-library-list" aria-label="作者列表">
        {personas.map((persona) => (
          <AuthorRow
            key={persona.author}
            persona={persona}
            job={latestJobs.find((job) => job.author === persona.author)}
            onSelect={onSelect}
            onCancel={onCancel}
            onRetry={onRetry}
          />
        ))}
        {jobOnly.map((job) => (
          <AuthorRow key={job.id} job={job} onSelect={onSelect} onCancel={onCancel} onRetry={onRetry} />
        ))}
        {!personas.length && !jobOnly.length ? (
          <div className="author-library-empty">
            <div className="empty-avatar-stack">
              <span />
              <span />
              <span />
            </div>
            <h2>还没有作者</h2>
            <p>添加一个知乎作者，PersonaForge 会在后台抓取、解析并建立本地索引。</p>
            <button className="primary-command" type="button" onClick={onAdd}>
              <Plus size={17} />
              添加第一位作者
            </button>
          </div>
        ) : null}
      </section>
    </main>
  );
}

function AuthorRow({
  persona,
  job,
  onSelect,
  onCancel,
  onRetry
}: {
  persona?: PersonaInfo;
  job?: AuthorJob;
  onSelect: (author: string) => void;
  onCancel: (jobId: string) => void;
  onRetry: (jobId: string) => void;
}) {
  const author = persona?.author || job?.author || '';
  const displayName = persona?.display_name || job?.display_name || author;
  const ready = Boolean(persona);
  const status = job && isActiveJob(job) ? job : null;
  const failure = job && ['failed', 'cancelled', 'interrupted'].includes(job.status) ? job : null;
  const contentCount = persona?.content_count ?? job?.item_count;

  return (
    <article className="author-library-row">
      <div className="author-identity">
        <AuthorAvatar label={displayName} src={persona?.avatar_url || job?.avatar_url} size="large" />
        <div>
          <strong>{displayName}</strong>
          <span>@{author}</span>
        </div>
      </div>
      <div className="author-library-meta">
        <span>{contentCount ? `${contentCount} 篇内容` : '尚无内容'}</span>
        <small>{persona?.last_synced_at ? `同步于 ${formatDate(persona.last_synced_at)}` : '尚未完成同步'}</small>
      </div>
      <div className={`author-status ${status ? 'working' : failure ? 'failed' : 'ready'}`}>
        <span className={`status-dot ${status ? 'running' : failure ? 'failed' : 'ready'}`} aria-hidden="true" />
        <span>{status?.label || failure?.label || '已就绪'}</span>
      </div>
      <div className="author-row-actions">
        {status ? (
          <button className="quiet-command" type="button" onClick={() => onCancel(status.id)}>
            取消
          </button>
        ) : failure ? (
          <button className="quiet-command" type="button" onClick={() => onRetry(failure.id)}>
            <RefreshCw size={15} />
            重试
          </button>
        ) : null}
        {ready ? (
          <button className="quiet-command emphasis" type="button" onClick={() => onSelect(author)}>
            进入对话
          </button>
        ) : null}
        {(persona?.profile_url || job?.profile_url) ? (
          <a
            className="icon-button"
            title="打开知乎主页"
            href={persona?.profile_url || job?.profile_url}
            target="_blank"
            rel="noreferrer"
          >
            <ExternalLink size={16} />
          </a>
        ) : null}
      </div>
      {failure?.error_message ? <p className="author-row-error">{failure.error_message}</p> : null}
    </article>
  );
}

export function AddAuthorModal({
  open,
  jobs,
  onClose,
  onJobCreated,
  onReady
}: {
  open: boolean;
  jobs: AuthorJob[];
  onClose: () => void;
  onJobCreated: (job: AuthorJob) => void;
  onReady: (author: string) => void;
}) {
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [value, setValue] = useState('');
  const [preview, setPreview] = useState<AuthorPreview | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [kinds, setKinds] = useState<ContentKind[]>(['answer', 'article', 'pin']);
  const [limitEnabled, setLimitEnabled] = useState(false);
  const [maxItems, setMaxItems] = useState(100);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const job = jobs.find((item) => item.id === jobId) || null;

  useEffect(() => {
    if (!open) return;
    setStep(1);
    setValue('');
    setPreview(null);
    setJobId(null);
    setKinds(['answer', 'article', 'pin']);
    setLimitEnabled(false);
    setMaxItems(100);
    setLoading(false);
    setError('');
  }, [open]);

  const stageOrder = ['queued', 'crawling', 'building', 'indexing', 'activating', 'ready'];
  const activeStageIndex = job ? stageOrder.indexOf(job.stage) : -1;

  async function handlePreview(event: FormEvent) {
    event.preventDefault();
    if (!value.trim()) return;
    setLoading(true);
    setError('');
    try {
      setPreview(await previewAuthor(value.trim()));
      setStep(2);
    } catch (caught) {
      setError(String((caught as Error).message || caught));
    } finally {
      setLoading(false);
    }
  }

  async function handleCreate() {
    if (!preview || !kinds.length) return;
    setLoading(true);
    setError('');
    try {
      const created = await createAuthorJob({
        author: preview.author,
        kinds,
        max_items: limitEnabled ? maxItems : null
      });
      setJobId(created.id);
      onJobCreated(created);
      setStep(3);
    } catch (caught) {
      setError(String((caught as Error).message || caught));
    } finally {
      setLoading(false);
    }
  }

  function toggleKind(kind: ContentKind) {
    setKinds((items) => items.includes(kind) ? items.filter((item) => item !== kind) : [...items, kind]);
  }

  if (!open) return null;

  return (
    <div className="modal-overlay" role="presentation" onMouseDown={onClose}>
      <section
        className="add-author-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="add-author-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="add-author-header">
          <div>
            <span>创建本地作者</span>
            <h2 id="add-author-title">添加知乎作者</h2>
          </div>
          <button className="icon-button" type="button" title="关闭" onClick={onClose}>
            <X size={18} />
          </button>
        </header>

        <div className="add-author-steps" aria-label="添加步骤">
          {[1, 2, 3].map((item) => (
            <span className={item === step ? 'active' : item < step ? 'complete' : ''} key={item} />
          ))}
        </div>

        <div className="add-author-body">
          {step === 1 ? (
            <form className="author-input-step" onSubmit={handlePreview}>
              <label htmlFor="author-input">知乎用户名或主页链接</label>
              <input
                id="author-input"
                autoFocus
                value={value}
                onChange={(event) => setValue(event.target.value)}
                placeholder="例如：wu-ren-jun-28"
              />
              <p>只读取公开资料。需要登录时，后台会使用服务端已有的知乎登录态。</p>
              {error ? <div className="form-error">{error}</div> : null}
              <div className="modal-actions">
                <button className="quiet-command" type="button" onClick={onClose}>取消</button>
                <button className="primary-command" type="submit" disabled={loading || !value.trim()}>
                  {loading ? '正在读取资料' : '继续'}
                </button>
              </div>
            </form>
          ) : null}

          {step === 2 && preview ? (
            <div className="author-confirm-step">
              <div className="author-preview-card">
                <AuthorAvatar label={preview.display_name} src={preview.avatar_url} size="large" />
                <div>
                  <strong>{preview.display_name}</strong>
                  <span>@{preview.author}</span>
                  {preview.headline ? <p>{preview.headline}</p> : null}
                </div>
                <a href={preview.profile_url} target="_blank" rel="noreferrer" title="打开知乎主页">
                  <ExternalLink size={16} />
                </a>
              </div>
              {preview.ready ? (
                <div className="inline-note">这位作者已经在作者库中，本次操作会同步公开内容并重建索引。</div>
              ) : null}
              <div className="crawl-summary">
                <span>抓取范围</span>
                <strong>全部公开内容</strong>
                <small>回答、文章和想法</small>
              </div>
              <details className="crawl-advanced">
                <summary>高级设置</summary>
                <div className="kind-options">
                  {([
                    ['answer', '回答'],
                    ['article', '文章'],
                    ['pin', '想法']
                  ] as Array<[ContentKind, string]>).map(([kind, label]) => (
                    <label key={kind}>
                      <input
                        type="checkbox"
                        checked={kinds.includes(kind)}
                        onChange={() => toggleKind(kind)}
                      />
                      {label}
                    </label>
                  ))}
                </div>
                <label className="limit-option">
                  <input
                    type="checkbox"
                    checked={limitEnabled}
                    onChange={(event) => setLimitEnabled(event.target.checked)}
                  />
                  限制抓取数量
                </label>
                {limitEnabled ? (
                  <input
                    type="number"
                    min={1}
                    max={10000}
                    value={maxItems}
                    onChange={(event) => setMaxItems(Math.max(1, Number(event.target.value) || 100))}
                  />
                ) : null}
              </details>
              {error ? <div className="form-error">{error}</div> : null}
              <div className="modal-actions">
                <button className="quiet-command" type="button" onClick={() => setStep(1)}>上一步</button>
                <button className="primary-command" type="button" disabled={loading || !kinds.length} onClick={handleCreate}>
                  {loading ? '正在创建任务' : preview.ready ? '开始同步' : '开始创建'}
                </button>
              </div>
            </div>
          ) : null}

          {step === 3 ? (
            <div className="author-progress-step">
              <div className={`progress-hero ${job?.status || 'queued'}`}>
                <AuthorAvatar
                  label={job?.display_name || preview?.display_name || 'PF'}
                  src={job?.avatar_url || preview?.avatar_url}
                  size="large"
                />
                <div>
                  <strong>{job?.display_name || preview?.display_name}</strong>
                  <span>{job?.label || '等待后台处理'}</span>
                </div>
              </div>
              <div className="job-stage-list">
                {[
                  ['crawling', '抓取公开内容'],
                  ['building', '解析作者材料'],
                  ['indexing', '创建检索索引'],
                  ['ready', '作者可以对话']
                ].map(([stage, label]) => {
                  const index = stageOrder.indexOf(stage);
                  const complete = job?.status === 'ready' || activeStageIndex > index;
                  const active = activeStageIndex === index && job?.status === 'running';
                  return (
                    <div className={complete ? 'complete' : active ? 'active' : ''} key={stage}>
                      <span>{complete ? <Check size={13} /> : null}</span>
                      <strong>{label}</strong>
                    </div>
                  );
                })}
              </div>
              {job?.item_count ? <p className="job-count">已获得 {job.item_count} 篇公开内容</p> : null}
              {job?.error_message ? <div className="form-error">{job.error_message}</div> : null}
              <div className="modal-actions">
                {job?.status === 'ready' ? (
                  <>
                    <button className="quiet-command" type="button" onClick={onClose}>稍后再说</button>
                    <button className="primary-command" type="button" onClick={() => onReady(job.author)}>
                      开始对话
                    </button>
                  </>
                ) : (
                  <button className="quiet-command emphasis" type="button" onClick={onClose}>
                    在后台继续
                  </button>
                )}
              </div>
            </div>
          ) : null}
        </div>
      </section>
    </div>
  );
}

function AuthorAvatar({
  label,
  src,
  size
}: {
  label: string;
  src?: string | null;
  size: 'small' | 'medium' | 'large';
}) {
  const initials = label.trim().slice(0, 2).toUpperCase() || 'PF';
  return src ? (
    <img className={`author-avatar ${size}`} src={src} alt={label} />
  ) : (
    <span className={`author-avatar author-avatar-fallback ${size}`}>{initials}</span>
  );
}

function latestJobsByAuthor(jobs: AuthorJob[]): AuthorJob[] {
  const values = new Map<string, AuthorJob>();
  jobs.forEach((job) => {
    if (!values.has(job.author)) values.set(job.author, job);
  });
  return [...values.values()];
}

function isActiveJob(job: AuthorJob): boolean {
  return job.status === 'queued' || job.status === 'running';
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat('zh-CN', {
    month: 'numeric',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  }).format(date);
}
