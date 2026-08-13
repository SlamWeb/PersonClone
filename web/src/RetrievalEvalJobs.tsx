import { type ChangeEvent, type ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  CircleDollarSign,
  Download,
  FileUp,
  LoaderCircle,
  Play,
  RefreshCw
} from 'lucide-react';
import {
  AuthUser,
  createRetrievalEvalJob,
  downloadRetrievalEvalHandoff,
  fetchRetrievalEvalJobs,
  importRetrievalEvalReview,
  PersonaInfo,
  resumeRetrievalEvalJob,
  RetrievalEvalJob
} from './api';

type Props = {
  viewSwitcher: ReactNode;
  user: AuthUser;
  personas: PersonaInfo[];
  authorScope: string | null;
};

const ACTIVE_STATUSES = new Set(['queued', 'running']);

const STATUS_LABELS: Record<RetrievalEvalJob['status'], string> = {
  queued: '等待执行',
  running: '正在执行',
  awaiting_codex: '等待导入',
  paused_budget: '预算暂停',
  completed: '已完成',
  failed: '失败',
  interrupted: '已中断'
};

const LABELER_LABELS: Record<RetrievalEvalJob['labeler'], string> = {
  deepseek_api: 'DeepSeek API',
  codex_handoff: 'Codex handoff',
  manual_import: '人工导入'
};

export function RetrievalEvalJobs({ viewSwitcher, user, personas, authorScope }: Props) {
  const [jobs, setJobs] = useState<RetrievalEvalJob[]>([]);
  const [author, setAuthor] = useState(
    personas.find((item) => item.author === authorScope)?.author || personas[0]?.author || ''
  );
  const [labeler, setLabeler] = useState<RetrievalEvalJob['labeler']>('codex_handoff');
  const [split, setSplit] = useState<RetrievalEvalJob['split']>('dev');
  const [budget, setBudget] = useState(5);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const previousAuthorScope = useRef(authorScope);

  const refresh = async () => {
    try {
      setJobs(await fetchRetrievalEvalJobs());
      setError('');
    } catch (reason) {
      setError(String((reason as Error).message || reason));
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  useEffect(() => {
    if (
      authorScope &&
      authorScope !== previousAuthorScope.current &&
      personas.some((item) => item.author === authorScope)
    ) {
      setAuthor(authorScope);
    }
    previousAuthorScope.current = authorScope;
  }, [authorScope, personas]);

  useEffect(() => {
    if (!personas.some((item) => item.author === author)) {
      setAuthor(personas[0]?.author || '');
    }
  }, [author, personas]);

  const hasActiveJob = jobs.some((job) => ACTIVE_STATUSES.has(job.status));
  useEffect(() => {
    if (!hasActiveJob) return;
    const timer = window.setInterval(() => void refresh(), 1800);
    return () => window.clearInterval(timer);
  }, [hasActiveJob]);

  const selectedPersona = useMemo(
    () => personas.find((persona) => persona.author === author),
    [author, personas]
  );
  const visibleJobs = useMemo(
    () => author ? jobs.filter((job) => job.author === author) : jobs,
    [author, jobs]
  );

  const createJob = async () => {
    if (!author) return;
    setBusy(true);
    setError('');
    setNotice('');
    try {
      const job = await createRetrievalEvalJob({ author, labeler, split, budget_cny: budget });
      setJobs((current) => [job, ...current.filter((item) => item.id !== job.id)]);
      setNotice('任务已创建，后台正在准备时间切分和候选池。');
    } catch (reason) {
      setError(String((reason as Error).message || reason));
    } finally {
      setBusy(false);
    }
  };

  const resumeJob = async (job: RetrievalEvalJob) => {
    const nextBudget = Math.max(job.budget_cny + 2, budget);
    setBusy(true);
    try {
      const updated = await resumeRetrievalEvalJob(job.id, nextBudget);
      upsert(updated);
    } catch (reason) {
      setError(String((reason as Error).message || reason));
    } finally {
      setBusy(false);
    }
  };

  const importReview = async (job: RetrievalEvalJob, event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    setBusy(true);
    try {
      const review = JSON.parse(await file.text()) as Record<string, unknown>;
      upsert(await importRetrievalEvalReview(job.id, review));
    } catch (reason) {
      setError(String((reason as Error).message || reason));
    } finally {
      setBusy(false);
    }
  };

  const upsert = (job: RetrievalEvalJob) => {
    setJobs((current) => [job, ...current.filter((item) => item.id !== job.id)]);
  };

  return (
    <section className="evaluation-workspace retrieval-jobs-workspace">
      <aside className="evaluation-query-rail retrieval-jobs-rail">
        <div className="evaluation-rail-heading"><strong>RAG 评估</strong></div>
        {viewSwitcher}
        <div className="retrieval-job-form">
          <label>作者
            <select value={author} onChange={(event) => setAuthor(event.target.value)}>
              {personas.map((persona) => <option key={persona.author} value={persona.author}>{persona.display_name}</option>)}
            </select>
          </label>
          <label>数据切分
            <select value={split} onChange={(event) => setSplit(event.target.value as RetrievalEvalJob['split'])}>
              <option value="dev">Dev 10</option>
              <option value="test">Test 20</option>
            </select>
          </label>
          <label>标注方式
            <select value={labeler} onChange={(event) => setLabeler(event.target.value as RetrievalEvalJob['labeler'])}>
              <option value="codex_handoff">Codex handoff</option>
              <option value="deepseek_api" disabled={user.role !== 'admin'}>DeepSeek API</option>
              <option value="manual_import">人工导入</option>
            </select>
          </label>
          {labeler === 'deepseek_api' ? (
            <label>预算上限（元）
              <input type="number" min="0.1" step="1" value={budget} onChange={(event) => setBudget(Number(event.target.value) || 5)} />
            </label>
          ) : null}
          <button className="retrieval-job-create" type="button" disabled={busy || !author} onClick={() => void createJob()}>
            {busy ? <LoaderCircle className="spin" size={17} /> : <Play size={17} />}
            创建评估任务
          </button>
          {selectedPersona ? <p>{selectedPersona.content_count ?? 0} 篇已入库内容</p> : null}
          {!authorScope ? <p className="evaluation-scope-note">当前是全部作者范围；任务仍会按左侧表单选择的具体作者创建。</p> : null}
        </div>
      </aside>

      <main className="retrieval-jobs-main">
        <header className="retrieval-jobs-header">
          <div><span>AUTHOR EVALUATION</span><h1>评估初始化任务</h1><p>冻结时间切分、构建候选池，并生成可恢复的双轴 Qrels。</p></div>
          <button type="button" title="刷新任务" onClick={() => void refresh()}><RefreshCw size={18} /></button>
        </header>
        <div className="retrieval-jobs-list">
          {error ? <div className="retrieval-job-error"><AlertCircle size={18} />{error}</div> : null}
          {notice ? <div className="retrieval-job-notice"><CheckCircle2 size={18} />{notice}</div> : null}
          {!visibleJobs.length ? (
            <div className="retrieval-jobs-empty"><strong>还没有评估任务</strong><p>从左侧选择作者和标注方式开始。</p></div>
          ) : visibleJobs.map((job) => (
            <RetrievalJobCard
              key={job.id}
              job={job}
              persona={personas.find((item) => item.author === job.author)}
              user={user}
              busy={busy}
              onDownload={() => void downloadRetrievalEvalHandoff(job.id).catch((reason) => setError(String(reason)))}
              onImport={(event) => void importReview(job, event)}
              onResume={() => void resumeJob(job)}
            />
          ))}
        </div>
      </main>
    </section>
  );
}

function RetrievalJobCard({ job, persona, user, busy, onDownload, onImport, onResume }: {
  job: RetrievalEvalJob;
  persona?: PersonaInfo;
  user: AuthUser;
  busy: boolean;
  onDownload: () => void;
  onImport: (event: ChangeEvent<HTMLInputElement>) => void;
  onResume: () => void;
}) {
  const progress = job.total_items > 0 ? Math.min(100, (job.completed_items / job.total_items) * 100) : 0;
  const cacheRate = job.usage?.cache_hit_rate;
  const canResume = ['paused_budget', 'failed', 'interrupted'].includes(job.status) && (job.labeler !== 'deepseek_api' || user.role === 'admin');
  return (
    <article className={`retrieval-job-card status-${job.status}`}>
      <div className="retrieval-job-card-head">
        <div className="retrieval-job-author">
          {persona?.avatar_url ? <img src={persona.avatar_url} alt="" /> : <span>{(persona?.display_name || job.author).slice(0, 1)}</span>}
          <div><strong>{persona?.display_name || job.author}</strong><p>{job.split === 'dev' ? 'Dev 10' : 'Test 20'} · {LABELER_LABELS[job.labeler]}</p></div>
        </div>
        <div className={`retrieval-job-status status-${job.status}`}>
          {job.status === 'running' ? <LoaderCircle className="spin" size={16} /> : job.status === 'completed' ? <CheckCircle2 size={16} /> : null}
          {STATUS_LABELS[job.status]}
        </div>
      </div>
      <div className="retrieval-job-stage"><strong>{job.label}</strong><span>{job.stage}</span></div>
      <div className="retrieval-job-progress"><i style={{ width: `${progress}%` }} /></div>
      <div className="retrieval-job-metrics">
        <span><b>{job.completed_items}</b> / {job.total_items || '待计算'} 项</span>
        <span><b>{cacheRate == null ? '—' : `${(cacheRate * 100).toFixed(1)}%`}</b> 缓存命中</span>
        <span><b>{job.usage?.completion_tokens?.toLocaleString() || '—'}</b> 输出 tokens</span>
        <span><CircleDollarSign size={15} /><b>¥{job.estimated_cost_cny.toFixed(3)}</b> / ¥{job.budget_cny.toFixed(0)}</span>
      </div>
      {job.error_message ? <p className="retrieval-job-error"><AlertCircle size={16} />{job.error_message}</p> : null}
      <footer className="retrieval-job-actions">
        {job.handoff_ready ? <button type="button" onClick={onDownload}><Download size={16} />下载 handoff</button> : null}
        {job.status === 'awaiting_codex' ? (
          <label className="retrieval-job-upload"><FileUp size={16} />导入 review JSON<input type="file" accept="application/json,.json" onChange={onImport} /></label>
        ) : null}
        {canResume ? <button type="button" disabled={busy} onClick={onResume}><RefreshCw size={16} />增加预算并继续</button> : null}
      </footer>
    </article>
  );
}
