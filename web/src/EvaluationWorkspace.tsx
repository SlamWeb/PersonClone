import { type ReactNode, useEffect, useMemo, useState } from 'react';
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Download,
  ExternalLink,
  Filter,
  LoaderCircle,
  PanelLeftClose,
  PanelLeftOpen
} from 'lucide-react';
import {
  fetchRetrievalPools,
  fetchRetrievalQuery,
  fetchRetrievalWorkspace,
  AuthUser,
  PersonaInfo,
  RetrievalCandidate,
  RetrievalPoolSummary,
  RetrievalQuery,
  RetrievalWorkspace,
  retrievalExportUrl,
  saveRetrievalLabel
} from './api';
import { GenerationEvaluationWorkspace } from './GenerationEvaluationWorkspace';
import { RetrievalLlmReport } from './RetrievalLlmReport';
import { RetrievalEvalJobs } from './RetrievalEvalJobs';

const SCORE_OPTIONS = [
  { score: 0 as const, key: '0', label: '无用', description: '不能帮助回答这道题' },
  { score: 1 as const, key: '1', label: '有一定帮助', description: '提供部分观点、例子或表达参考' },
  { score: 2 as const, key: '2', label: '明显有用', description: '直接支撑高质量回答' }
];

function candidateHeading(candidate: RetrievalCandidate): string {
  if (candidate.kind === 'pin') return '想法';
  if (candidate.title.trim()) return candidate.title.trim();
  if (candidate.kind === 'article') return '文章';
  if (candidate.kind === 'answer') return '回答';
  return '历史内容';
}

function HumanRetrievalEvaluationWorkspace({ viewSwitcher, authorScope }: { viewSwitcher: ReactNode; authorScope: string | null }) {
  const [pools, setPools] = useState<RetrievalPoolSummary[]>([]);
  const [poolId, setPoolId] = useState('');
  const [workspace, setWorkspace] = useState<RetrievalWorkspace | null>(null);
  const [itemId, setItemId] = useState('');
  const [query, setQuery] = useState<RetrievalQuery | null>(null);
  const [candidateIndex, setCandidateIndex] = useState(0);
  const [unfinishedOnly, setUnfinishedOnly] = useState(true);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [savedPulse, setSavedPulse] = useState(false);
  const [error, setError] = useState('');
  const [railCollapsed, setRailCollapsed] = useState(
    () => localStorage.getItem('pf-evaluation-sidebar-collapsed') === 'true'
  );

  useEffect(() => {
    let active = true;
    setLoading(true);
    fetchRetrievalPools(authorScope)
      .then((items) => {
        if (!active) return;
        setPools(items);
        setPoolId((current) => items.some((item) => item.pool_id === current) ? current : items[0]?.pool_id || '');
      })
      .catch((reason) => active && setError(String((reason as Error).message || reason)))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [authorScope]);

  useEffect(() => {
    if (!poolId) {
      setWorkspace(null);
      setItemId('');
      return;
    }
    let active = true;
    setLoading(true);
    setError('');
    fetchRetrievalWorkspace(poolId)
      .then((next) => {
        if (!active) return;
        setWorkspace(next);
        setItemId((current) => {
          if (next.queries.some((row) => row.item_id === current)) return current;
          return next.queries.find((row) => !row.completed)?.item_id || next.queries[0]?.item_id || '';
        });
      })
      .catch((reason) => active && setError(String((reason as Error).message || reason)))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [poolId]);

  useEffect(() => {
    if (!poolId || !itemId) {
      setQuery(null);
      return;
    }
    let active = true;
    setLoading(true);
    setError('');
    fetchRetrievalQuery(poolId, itemId)
      .then((next) => {
        if (!active) return;
        setQuery(next);
        const firstUnfinished = next.candidates.findIndex((candidate) => candidate.score === null);
        setCandidateIndex(firstUnfinished >= 0 ? firstUnfinished : 0);
      })
      .catch((reason) => active && setError(String((reason as Error).message || reason)))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [poolId, itemId]);

  const visibleQueries = useMemo(
    () => workspace?.queries.filter((row) => !unfinishedOnly || !row.completed) || [],
    [workspace, unfinishedOnly]
  );
  const currentCandidate = query?.candidates[candidateIndex] || null;

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.metaKey || event.ctrlKey || event.altKey || saving) return;
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
      if (event.key === '0' || event.key === '1' || event.key === '2') {
        event.preventDefault();
        void scoreCandidate(Number(event.key) as 0 | 1 | 2);
      }
      if (event.key === 'ArrowLeft') {
        event.preventDefault();
        moveCandidate(-1);
      }
      if (event.key === 'ArrowRight') {
        event.preventDefault();
        moveCandidate(1);
      }
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [currentCandidate, saving, query, candidateIndex, workspace]);

  function moveCandidate(delta: number) {
    if (!query) return;
    const next = candidateIndex + delta;
    if (next >= 0 && next < query.candidates.length) {
      setCandidateIndex(next);
      return;
    }
    moveQuery(delta > 0 ? 1 : -1);
  }

  function moveQuery(delta: number) {
    if (!workspace) return;
    const rows = unfinishedOnly ? workspace.queries.filter((row) => !row.completed) : workspace.queries;
    const currentIndex = rows.findIndex((row) => row.item_id === itemId);
    const next = rows[currentIndex + delta];
    if (next) setItemId(next.item_id);
  }

  async function scoreCandidate(score: 0 | 1 | 2) {
    if (!currentCandidate || !query || saving) return;
    setSaving(true);
    setError('');
    try {
      const saved = await saveRetrievalLabel(poolId, query.item_id, currentCandidate.parent_id, score);
      const nextCandidates = query.candidates.map((candidate, index) =>
        index === candidateIndex ? { ...candidate, score, retrieval_details: saved.retrieval_details } : candidate
      );
      setQuery({ ...query, candidates: nextCandidates, labeled_count: new Set(
        nextCandidates.filter((candidate) => candidate.score !== null).map((candidate) => candidate.parent_id)
      ).size });
      const nextWorkspace = await fetchRetrievalWorkspace(poolId);
      setWorkspace(nextWorkspace);
      setSavedPulse(true);
      window.setTimeout(() => setSavedPulse(false), 900);
      const nextUnfinished = nextCandidates.findIndex(
        (candidate, index) => index > candidateIndex && candidate.score === null
      );
      if (nextUnfinished >= 0) {
        setCandidateIndex(nextUnfinished);
      } else {
        const nextQuery = nextWorkspace.queries.find(
          (row) => row.ordinal > (nextWorkspace.queries.find((entry) => entry.item_id === itemId)?.ordinal || 0) && !row.completed
        );
        if (nextQuery) setItemId(nextQuery.item_id);
      }
    } catch (reason) {
      setError(String((reason as Error).message || reason));
    } finally {
      setSaving(false);
    }
  }

  if (loading && !workspace && !pools.length) {
    return <div className="evaluation-state"><LoaderCircle className="spin" size={20} />正在读取评估集</div>;
  }

  if (!pools.length) {
    return (
      <div className="evaluation-empty">
        <strong>{authorScope ? '这个作者还没有检索评估资产' : '还没有可汇总的检索候选池'}</strong>
        <p>{authorScope ? '请先为当前作者生成冻结候选池；其他作者的材料不会混入这里。' : '先为至少一个作者生成冻结候选池，完成后这里会自动出现。'}</p>
        <code>pf eval retrieval-pool wu-ren-jun-28 --dataset data/eval/wu-ren-jun-28-temporal-v0/dataset.jsonl --dataset-id temporal_dev10_v0 --split dev</code>
      </div>
    );
  }

  return (
    <section className={`evaluation-workspace ${railCollapsed ? 'rail-collapsed' : ''}`}>
      <aside className="evaluation-query-rail">
        <div className="evaluation-rail-heading">
          <strong>RAG 评估</strong>
          <button
            type="button"
            title="收起评估栏"
            onClick={() => {
              setRailCollapsed(true);
              localStorage.setItem('pf-evaluation-sidebar-collapsed', 'true');
            }}
          >
            <PanelLeftClose size={17} />
          </button>
        </div>
        {viewSwitcher}
        <div className="evaluation-dataset-controls">
          <label>
            评估集
            <select value={poolId} onChange={(event) => setPoolId(event.target.value)}>
              {pools.map((pool) => (
                <option key={pool.pool_id} value={pool.pool_id}>
                  {pool.author ? `${pool.author} · ` : '未归属/旧数据 · '}{pool.display_name || pool.dataset_id} · {pool.candidate_count} 条
                </option>
              ))}
            </select>
          </label>
          <div className="evaluation-total-progress">
            <span>{workspace?.progress.labeled || 0} / {workspace?.progress.total || 0}</span>
            <div><i style={{ width: `${workspace?.progress.total ? (workspace.progress.labeled / workspace.progress.total) * 100 : 0}%` }} /></div>
          </div>
          <button
            className={`unfinished-filter ${unfinishedOnly ? 'active' : ''}`}
            type="button"
            onClick={() => setUnfinishedOnly((value) => !value)}
            aria-pressed={unfinishedOnly}
          >
            <Filter size={14} />只看未完成
          </button>
        </div>
        <nav className="evaluation-query-list" aria-label="评估问题">
          {visibleQueries.map((row) => (
            <button
              key={row.item_id}
              className={row.item_id === itemId ? 'active' : ''}
              type="button"
              onClick={() => setItemId(row.item_id)}
            >
              <span className="query-ordinal">{row.ordinal}</span>
              <span className="query-list-copy">
                <strong>{row.query}</strong>
                <small>{row.labeled_count} / {row.candidate_count}</small>
              </span>
              {row.completed ? <Check size={15} /> : null}
            </button>
          ))}
          {!visibleQueries.length && workspace?.progress.completed ? (
            <div className="evaluation-finished-note"><Check size={16} />全部完成</div>
          ) : null}
        </nav>
        <div className="evaluation-export">
          <a href={retrievalExportUrl(poolId, 'jsonl')}><Download size={14} />JSONL</a>
          <a href={retrievalExportUrl(poolId, 'csv')}><Download size={14} />CSV</a>
        </div>
      </aside>

      <div className="evaluation-reader">
        {railCollapsed ? (
          <button
            className="evaluation-rail-reveal"
            type="button"
            title="展开评估栏"
            onClick={() => {
              setRailCollapsed(false);
              localStorage.setItem('pf-evaluation-sidebar-collapsed', 'false');
            }}
          >
            <PanelLeftOpen size={18} />
            <span>RAG 评估</span>
          </button>
        ) : null}
        <header className="evaluation-question-header">
          <span>
            评估问题 {workspace?.queries.find((row) => row.item_id === itemId)?.ordinal || '-'} / {workspace?.queries.length || 0}
          </span>
          <h1>{query?.query || '正在读取问题'}</h1>
          <div className="candidate-position">
            候选 {query ? candidateIndex + 1 : 0} / {query?.candidate_count || 0}
          </div>
        </header>

        <article className="evaluation-document">
          {currentCandidate ? (
            <div className="evaluation-document-surface">
              <div className="evaluation-document-title">
                <div>
                  <h2>{candidateHeading(currentCandidate)}</h2>
                </div>
                {currentCandidate.url ? (
                  <a href={currentCandidate.url} target="_blank" rel="noreferrer" title="打开知乎原文">
                    <ExternalLink size={17} />
                  </a>
                ) : null}
              </div>
              <div className="evaluation-document-body">{currentCandidate.text}</div>
              {currentCandidate.score !== null && currentCandidate.retrieval_details ? (
                <details className="retrieval-details">
                  <summary>查看检索来源</summary>
                  <div>
                    {Object.entries(currentCandidate.retrieval_details).map(([route, details]) => (
                      <span key={route}>{route} · #{details.rank}</span>
                    ))}
                  </div>
                </details>
              ) : null}
            </div>
          ) : <div className="evaluation-state">没有候选材料</div>}
        </article>

        <footer className="evaluation-scoring-dock">
          <div className="evaluation-score-header">
            <div className="score-prompt">
              <strong>这篇材料对回答当前问题有多大帮助？</strong>
              <span>按数字键 0、1、2 也可以快速评分</span>
            </div>
            <div className="candidate-navigation">
              <button type="button" onClick={() => moveCandidate(-1)} disabled={!query || (candidateIndex === 0 && workspace?.queries[0]?.item_id === itemId)}>
                <ChevronLeft size={17} />上一篇
              </button>
              <button type="button" onClick={() => moveCandidate(1)} disabled={!query}>暂时跳过</button>
              <button type="button" onClick={() => moveCandidate(1)} disabled={!query}>
                下一篇<ChevronRight size={17} />
              </button>
            </div>
          </div>
          <div className="score-options" aria-label="材料帮助程度">
            {SCORE_OPTIONS.map((option) => (
              <button
                key={option.score}
                className={currentCandidate?.score === option.score ? 'selected' : ''}
                type="button"
                disabled={!currentCandidate || saving}
                onClick={() => void scoreCandidate(option.score)}
              >
                <kbd>{option.key}</kbd>
                <span><strong>{option.label}</strong><small>{option.description}</small></span>
              </button>
            ))}
          </div>
          <div className={`evaluation-save-state ${savedPulse ? 'visible' : ''}`}><Check size={13} />已保存</div>
          {error ? <div className="evaluation-error">{error}</div> : null}
        </footer>
      </div>
    </section>
  );
}

export function EvaluationWorkspace({ user, personas, authorScope }: { user: AuthUser; personas: PersonaInfo[]; authorScope: string | null }) {
  const [evaluationKind, setEvaluationKind] = useState<'rag' | 'generate'>(() =>
    localStorage.getItem('pf-evaluation-kind') === 'generate' ? 'generate' : 'rag'
  );
  const [retrievalView, setRetrievalView] = useState<'human' | 'llm' | 'jobs'>(() => {
    const stored = localStorage.getItem('pf-retrieval-view');
    return stored === 'llm' || stored === 'jobs' ? stored : 'human';
  });

  const selectRetrievalView = (next: 'human' | 'llm' | 'jobs') => {
    setRetrievalView(next);
    localStorage.setItem('pf-retrieval-view', next);
  };

  const retrievalViewSwitcher = (
    <div className="retrieval-view-switch" role="tablist" aria-label="RAG 评估视图">
      <button
        type="button"
        className={retrievalView === 'human' ? 'active' : ''}
        role="tab"
        aria-selected={retrievalView === 'human'}
        onClick={() => selectRetrievalView('human')}
      >人工标注</button>
      <button
        type="button"
        className={retrievalView === 'llm' ? 'active' : ''}
        role="tab"
        aria-selected={retrievalView === 'llm'}
        onClick={() => selectRetrievalView('llm')}
      >LLM 报告</button>
      <button
        type="button"
        className={retrievalView === 'jobs' ? 'active' : ''}
        role="tab"
        aria-selected={retrievalView === 'jobs'}
        onClick={() => selectRetrievalView('jobs')}
      >评估任务</button>
    </div>
  );

  return (
    <section className="evaluation-hub">
      <div className={`evaluation-kind-switch mode-${evaluationKind}`} role="tablist" aria-label="评估对象">
        <span aria-hidden="true" />
        <button
          type="button"
          role="tab"
          aria-selected={evaluationKind === 'rag'}
          onClick={() => {
            setEvaluationKind('rag');
            localStorage.setItem('pf-evaluation-kind', 'rag');
          }}
        >RAG</button>
        <button
          type="button"
          role="tab"
          aria-selected={evaluationKind === 'generate'}
          onClick={() => {
            setEvaluationKind('generate');
            localStorage.setItem('pf-evaluation-kind', 'generate');
          }}
        >Generate</button>
      </div>
      <div className="evaluation-kind-stage">
        {evaluationKind === 'rag' ? (
          <>
            {retrievalView === 'human' ? (
              <HumanRetrievalEvaluationWorkspace viewSwitcher={retrievalViewSwitcher} authorScope={authorScope} />
            ) : retrievalView === 'llm' ? (
              <RetrievalLlmReport viewSwitcher={retrievalViewSwitcher} authorScope={authorScope} />
            ) : (
              <RetrievalEvalJobs viewSwitcher={retrievalViewSwitcher} user={user} personas={personas} authorScope={authorScope} />
            )}
          </>
        ) : <GenerationEvaluationWorkspace authorScope={authorScope} />}
      </div>
    </section>
  );
}
