import { useEffect, useMemo, useState } from 'react';
import {
  BarChart3,
  Bot,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleDot,
  LoaderCircle,
  PanelLeftClose,
  PanelLeftOpen,
  Scale,
  Sparkles
} from 'lucide-react';
import {
  createGenerationJudgeJob,
  fetchGenerationComparison,
  fetchGenerationComparisonItem,
  fetchGenerationItem,
  fetchGenerationJudgeJob,
  fetchGenerationSystems,
  fetchGenerationWorkspace,
  GenerationComparison,
  GenerationComparisonItem,
  GenerationItem,
  GenerationJudgeJob,
  GenerationSystem,
  GenerationWorkspace,
  saveGenerationPairwise,
  saveGenerationRubric
} from './api';

type GenerateView = 'overview' | 'rubric' | 'pairwise' | 'judge';

const GENERATE_VIEWS: Array<{ key: GenerateView; label: string; icon: typeof BarChart3 }> = [
  { key: 'overview', label: '总览', icon: BarChart3 },
  { key: 'rubric', label: '人工六维', icon: CircleDot },
  { key: 'pairwise', label: 'AB 对比', icon: Scale },
  { key: 'judge', label: 'LLM Judge', icon: Bot }
];

const GROUP_LABELS: Record<string, string> = {
  content: '内容与思路',
  style: '语言与表达',
  naturalness: '自然度'
};

function systemName(system: GenerationSystem): string {
  return system.display_name || system.run_name;
}

function methodName(system: GenerationSystem): string {
  const names: Record<string, string> = {
    rag_magic_if: '纯 RAG + Magic If',
    mrprompt: 'MRPrompt',
    persona_pack: 'Persona Pack',
    strong_identity: 'Strong Identity',
    current: 'Current',
    writer_replay: 'Writer Replay'
  };
  return names[system.writer_prompt] || names[system.method_id || ''] || systemName(system);
}

function formatScore(value: number | null | undefined): string {
  return typeof value === 'number' ? value.toFixed(2) : '—';
}

function dimensionMean(system: GenerationSystem, key: string): number | null {
  return system.judge?.result?.dimensions?.[key]?.mean ?? null;
}

function itemPosition(items: Array<{ item_id: string }>, itemId: string): number {
  return Math.max(0, items.findIndex((item) => item.item_id === itemId));
}

export function GenerationEvaluationWorkspace({ authorScope }: { authorScope: string | null }) {
  const [view, setView] = useState<GenerateView>(() => {
    const stored = localStorage.getItem('pf-generation-evaluation-view');
    return GENERATE_VIEWS.some((entry) => entry.key === stored) ? stored as GenerateView : 'overview';
  });
  const [systems, setSystems] = useState<GenerationSystem[]>([]);
  const [systemId, setSystemId] = useState('');
  const [rightSystemId, setRightSystemId] = useState('');
  const [workspace, setWorkspace] = useState<GenerationWorkspace | null>(null);
  const [itemId, setItemId] = useState('');
  const [item, setItem] = useState<GenerationItem | null>(null);
  const [scores, setScores] = useState<Record<string, number>>({});
  const [note, setNote] = useState('');
  const [comparison, setComparison] = useState<GenerationComparison | null>(null);
  const [comparisonItem, setComparisonItem] = useState<GenerationComparisonItem | null>(null);
  const [judgeJob, setJudgeJob] = useState<GenerationJudgeJob | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [railCollapsed, setRailCollapsed] = useState(
    () => localStorage.getItem('pf-generation-sidebar-collapsed') === 'true'
  );

  async function refreshSystems(preferredSystemId?: string) {
    const next = await fetchGenerationSystems(authorScope);
    setSystems(next);
    setSystemId((current) => {
      const preferred = preferredSystemId || current;
      return next.some((system) => system.system_id === preferred) ? preferred : next[0]?.system_id || '';
    });
    setRightSystemId((current) => {
      if (next.some((system) => system.system_id === current) && current !== (preferredSystemId || systemId)) {
        return current;
      }
      return next.find((system) => system.system_id !== (preferredSystemId || systemId || next[0]?.system_id))?.system_id || '';
    });
  }

  useEffect(() => {
    let active = true;
    setLoading(true);
    fetchGenerationSystems(authorScope)
      .then((next) => {
        if (!active) return;
        setSystems(next);
        const params = new URLSearchParams(window.location.search);
        const requestedSystem = params.get('generationSystem');
        const requestedItem = params.get('generationItem');
        const initialSystem = next.some((entry) => entry.system_id === requestedSystem)
          ? requestedSystem || ''
          : next[0]?.system_id || '';
        setSystemId(initialSystem);
        setRightSystemId(next.find((entry) => entry.system_id !== initialSystem)?.system_id || '');
        if (requestedItem && params.get('generationView') === 'judge') setItemId(requestedItem);
        const requestedView = params.get('generationView') as GenerateView | null;
        if (requestedView && GENERATE_VIEWS.some((entry) => entry.key === requestedView)) setView(requestedView);
      })
      .catch((reason) => active && setError(String((reason as Error).message || reason)))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [authorScope]);

  useEffect(() => {
    if (rightSystemId !== systemId && systems.some((system) => system.system_id === rightSystemId)) return;
    setRightSystemId(systems.find((system) => system.system_id !== systemId)?.system_id || '');
  }, [systemId, rightSystemId, systems]);

  useEffect(() => {
    if (!systemId || (view !== 'rubric' && view !== 'judge')) {
      if (view !== 'rubric' && view !== 'judge') setWorkspace(null);
      return;
    }
    let active = true;
    setLoading(true);
    setError('');
    fetchGenerationWorkspace(systemId)
      .then((next) => {
        if (!active) return;
        setWorkspace(next);
        setJudgeJob(next.judge || null);
        if (view === 'rubric') {
          setItemId((current) => {
            if (next.items.some((entry) => entry.item_id === current)) return current;
            return next.items.find((entry) => !entry.completed)?.item_id || next.items[0]?.item_id || '';
          });
        } else if (view === 'judge') {
          setItemId((current) => next.items.some((entry) => entry.item_id === current) ? current : '');
        }
      })
      .catch((reason) => active && setError(String((reason as Error).message || reason)))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [systemId, view]);

  useEffect(() => {
    if ((view !== 'rubric' && view !== 'judge') || !systemId || !itemId) {
      setItem(null);
      return;
    }
    let active = true;
    setLoading(true);
    setError('');
    fetchGenerationItem(systemId, itemId)
      .then((next) => {
        if (!active) return;
        setItem(next);
        setScores(next.human_scores || {});
        setNote(next.human_note || '');
      })
      .catch((reason) => active && setError(String((reason as Error).message || reason)))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [view, systemId, itemId]);

  useEffect(() => {
    if (view !== 'pairwise' || !systemId || !rightSystemId || systemId === rightSystemId) {
      setComparison(null);
      return;
    }
    let active = true;
    setLoading(true);
    setError('');
    fetchGenerationComparison(systemId, rightSystemId)
      .then((next) => {
        if (!active) return;
        setComparison(next);
        setItemId((current) => {
          if (next.items.some((entry) => entry.item_id === current)) return current;
          return next.items.find((entry) => !entry.completed)?.item_id || next.items[0]?.item_id || '';
        });
      })
      .catch((reason) => active && setError(String((reason as Error).message || reason)))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [view, systemId, rightSystemId]);

  useEffect(() => {
    if (view !== 'pairwise' || !systemId || !rightSystemId || !itemId || systemId === rightSystemId) {
      setComparisonItem(null);
      return;
    }
    let active = true;
    setLoading(true);
    fetchGenerationComparisonItem(systemId, rightSystemId, itemId)
      .then((next) => active && setComparisonItem(next))
      .catch((reason) => active && setError(String((reason as Error).message || reason)))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [view, systemId, rightSystemId, itemId]);

  useEffect(() => {
    if (!judgeJob || (judgeJob.status !== 'queued' && judgeJob.status !== 'running')) return;
    const timer = window.setInterval(() => {
      void fetchGenerationJudgeJob(judgeJob.id)
        .then((next) => {
          setJudgeJob(next);
          if (next.status === 'completed' || next.status === 'failed') {
            window.clearInterval(timer);
            void refreshSystems(next.system_id);
          }
        })
        .catch(() => undefined);
    }, 2000);
    return () => window.clearInterval(timer);
  }, [judgeJob?.id, judgeJob?.status]);

  const selectedSystem = useMemo(
    () => systems.find((system) => system.system_id === systemId) || null,
    [systems, systemId]
  );
  const authorLabel = authorScope || '全部作者';
  const currentItems = view === 'pairwise' ? comparison?.items || [] : workspace?.items || [];
  const currentOrdinal = itemPosition(currentItems, itemId);

  function selectView(next: GenerateView) {
    setView(next);
    setItemId('');
    setError('');
    localStorage.setItem('pf-generation-evaluation-view', next);
    const params = new URLSearchParams(window.location.search);
    params.set('generationView', next);
    params.delete('generationItem');
    window.history.replaceState(null, '', `${window.location.pathname}?${params.toString()}`);
  }

  function openJudgeItem(nextItemId: string) {
    setView('judge');
    setItemId(nextItemId);
    setError('');
    localStorage.setItem('pf-generation-evaluation-view', 'judge');
    const params = new URLSearchParams(window.location.search);
    params.set('generationView', 'judge');
    params.set('generationSystem', systemId);
    params.set('generationItem', nextItemId);
    window.history.pushState(null, '', `${window.location.pathname}?${params.toString()}`);
  }

  function openJudgeOverview() {
    setView('judge');
    setItemId('');
    setItem(null);
    setError('');
    localStorage.setItem('pf-generation-evaluation-view', 'judge');
    const params = new URLSearchParams(window.location.search);
    params.set('generationView', 'judge');
    params.set('generationSystem', systemId);
    params.delete('generationItem');
    window.history.pushState(null, '', `${window.location.pathname}?${params.toString()}`);
  }

  function moveItem(delta: number) {
    const next = currentItems[currentOrdinal + delta];
    if (next) {
      setItemId(next.item_id);
      if (view === 'judge') {
        const params = new URLSearchParams(window.location.search);
        params.set('generationView', 'judge');
        params.set('generationSystem', systemId);
        params.set('generationItem', next.item_id);
        window.history.replaceState(null, '', `${window.location.pathname}?${params.toString()}`);
      }
    }
  }

  async function setDimension(key: string, score: number) {
    if (!item || saving) return;
    const nextScores = { ...scores, [key]: score };
    setScores(nextScores);
    setSaving(true);
    setError('');
    try {
      await saveGenerationRubric(systemId, item.item_id, nextScores, note);
      const nextWorkspace = await fetchGenerationWorkspace(systemId);
      setWorkspace(nextWorkspace);
      setItem({ ...item, human_scores: nextScores, human_completed: Object.keys(nextScores).length === 6 });
    } catch (reason) {
      setError(String((reason as Error).message || reason));
    } finally {
      setSaving(false);
    }
  }

  async function saveNote() {
    if (!item || saving) return;
    setSaving(true);
    try {
      await saveGenerationRubric(systemId, item.item_id, scores, note);
    } catch (reason) {
      setError(String((reason as Error).message || reason));
    } finally {
      setSaving(false);
    }
  }

  async function vote(choice: 'A' | 'B') {
    if (!comparisonItem || saving) return;
    setSaving(true);
    setError('');
    try {
      const saved = await saveGenerationPairwise(systemId, rightSystemId, comparisonItem.item_id, choice);
      setComparisonItem({ ...comparisonItem, choice, revealed: saved.revealed });
      const next = await fetchGenerationComparison(systemId, rightSystemId);
      setComparison(next);
    } catch (reason) {
      setError(String((reason as Error).message || reason));
    } finally {
      setSaving(false);
    }
  }

  async function startJudge() {
    if (!systemId || judgeJob?.status === 'queued' || judgeJob?.status === 'running') return;
    setError('');
    try {
      const next = await createGenerationJudgeJob(systemId);
      setJudgeJob(next);
    } catch (reason) {
      setError(String((reason as Error).message || reason));
    }
  }

  if (loading && !systems.length) {
    return <div className="generation-empty"><LoaderCircle className="spin" size={20} />正在读取冻结系统</div>;
  }

  if (!systems.length) {
    return (
      <div className="generation-empty">
        <strong>{authorScope ? '这个作者还没有可比较的生成系统' : '还没有可汇总的生成系统'}</strong>
        <p>{authorScope ? '请先为当前作者生成同一套 dev10；其他作者的回答不会混入这里。' : '全部作者范围只用于汇总观察，不能跨作者进行逐题比较。'}</p>
      </div>
    );
  }

  return (
    <section className={`generation-evaluation ${railCollapsed ? 'rail-collapsed' : ''}`}>
      <aside className="generation-rail">
        <div className="generation-rail-heading">
          <div><Sparkles size={16} /><strong>生成评估</strong></div>
          <button
            type="button"
            title="收起评估栏"
            onClick={() => {
              setRailCollapsed(true);
              localStorage.setItem('pf-generation-sidebar-collapsed', 'true');
            }}
          >
            <PanelLeftClose size={17} />
          </button>
        </div>
        <nav className="generation-view-nav" aria-label="生成评估方式">
          {GENERATE_VIEWS.map((entry) => {
            const Icon = entry.icon;
            return (
              <button
                key={entry.key}
                className={view === entry.key ? 'active' : ''}
                type="button"
                onClick={() => selectView(entry.key)}
              >
                <Icon size={16} />{entry.label}
              </button>
            );
          })}
        </nav>
        {view !== 'overview' ? (
          <div className="generation-system-picker">
            <p className="evaluation-scope-note">作者范围：{authorLabel}</p>
            <label>
              生成系统
              <select value={systemId} onChange={(event) => { setSystemId(event.target.value); setItemId(''); setItem(null); }}>
                {systems.map((system) => (
                  <option key={system.system_id} value={system.system_id}>{systemName(system)}{!authorScope && system.author ? ` · ${system.author}` : ''}{!system.author ? ' · 未归属/旧数据' : ''}</option>
                ))}
              </select>
            </label>
            {view === 'pairwise' ? (
              <label>
                对比系统
                <select value={rightSystemId} onChange={(event) => { setRightSystemId(event.target.value); setItemId(''); setItem(null); }}>
                  {systems.filter((system) => system.system_id !== systemId && Boolean(system.author) && system.author === selectedSystem?.author).map((system) => (
                    <option key={system.system_id} value={system.system_id}>{systemName(system)}</option>
                  ))}
                </select>
              </label>
            ) : null}
          </div>
        ) : null}
        {(view === 'rubric' || view === 'pairwise') && currentItems.length ? (
          <nav className="generation-item-list" aria-label="dev10 问题">
            {currentItems.map((entry) => (
              <button
                key={entry.item_id}
                className={entry.item_id === itemId ? 'active' : ''}
                type="button"
                onClick={() => setItemId(entry.item_id)}
              >
                <span>{entry.ordinal}</span>
                <strong>{entry.question}</strong>
                {entry.completed ? <Check size={14} /> : null}
              </button>
            ))}
          </nav>
        ) : null}
      </aside>

      <main className="generation-stage">
        {railCollapsed ? (
          <button
            className="generation-rail-reveal"
            type="button"
            onClick={() => {
              setRailCollapsed(false);
              localStorage.setItem('pf-generation-sidebar-collapsed', 'false');
            }}
          >
            <PanelLeftOpen size={18} />生成评估
          </button>
        ) : null}

        {view === 'overview' ? (
          <Overview systems={systems} authorScope={authorScope} onOpen={(nextId, nextView) => { setSystemId(nextId); selectView(nextView); }} />
        ) : null}

        {view === 'rubric' && workspace && item ? (
          <RubricView
            workspace={workspace}
            item={item}
            scores={scores}
            note={note}
            saving={saving}
            ordinal={currentOrdinal}
            onScore={setDimension}
            onNote={setNote}
            onSaveNote={saveNote}
            onMove={moveItem}
          />
        ) : null}

        {view === 'pairwise' && comparison && comparisonItem ? (
          <PairwiseView
            comparison={comparison}
            item={comparisonItem}
            saving={saving}
            ordinal={currentOrdinal}
            onVote={vote}
            onMove={moveItem}
          />
        ) : null}

        {view === 'judge' && selectedSystem ? (
          item && workspace ? (
            <JudgeItemDetail
              system={selectedSystem}
              workspace={workspace}
              item={item}
              ordinal={currentOrdinal}
              onBack={openJudgeOverview}
              onMove={moveItem}
            />
          ) : (
            <JudgeView
              system={selectedSystem}
              workspace={workspace}
              job={judgeJob}
              onStart={startJudge}
              onOpenItem={openJudgeItem}
            />
          )
        ) : null}

        {loading && systems.length ? <div className="generation-loading"><LoaderCircle className="spin" size={18} />读取中</div> : null}
        {error ? <div className="generation-error">{error}</div> : null}
      </main>
    </section>
  );
}

function Overview({
  systems,
  authorScope,
  onOpen
}: {
  systems: GenerationSystem[];
  authorScope: string | null;
  onOpen: (systemId: string, view: GenerateView) => void;
}) {
  const datasetOptions = useMemo(() => {
    const seen = new Map<string, { id: string; sha: string; count: number }>();
    systems.forEach((system) => {
      const key = `${system.dataset_id}:${system.dataset_sha256}`;
      const current = seen.get(key);
      seen.set(key, current ? { ...current, count: current.count + 1 } : {
        id: system.dataset_id,
        sha: system.dataset_sha256,
        count: 1
      });
    });
    return Array.from(seen.values());
  }, [systems]);
  const judgeOptions = useMemo(() => {
    const values = new Set<string>(['gold-judge-v1.0']);
    systems.forEach((system) => {
      if (system.judge?.prompt_version) values.add(system.judge.prompt_version);
    });
    return Array.from(values);
  }, [systems]);
  const [datasetKey, setDatasetKey] = useState('');
  const [judgeVersion, setJudgeVersion] = useState('gold-judge-v1.0');

  useEffect(() => {
    if (!datasetKey && datasetOptions[0]) {
      setDatasetKey(`${datasetOptions[0].id}:${datasetOptions[0].sha}`);
    } else if (datasetKey && !datasetOptions.some((entry) => `${entry.id}:${entry.sha}` === datasetKey)) {
      setDatasetKey(datasetOptions[0] ? `${datasetOptions[0].id}:${datasetOptions[0].sha}` : '');
    }
    if (!judgeOptions.includes(judgeVersion)) setJudgeVersion(judgeOptions[0] || 'gold-judge-v1.0');
  }, [datasetKey, datasetOptions, judgeOptions, judgeVersion]);

  const selectedDataset = datasetOptions.find((entry) => `${entry.id}:${entry.sha}` === datasetKey);
  const filteredSystems = systems.filter((system) => `${system.dataset_id}:${system.dataset_sha256}` === datasetKey);
  const selectedSplit = filteredSystems[0]?.split || 'dev';
  const selectedSplitLabel = selectedSplit === 'test' ? 'Test20' : 'Dev10';
  const judgedSystems = filteredSystems.filter((system) => system.judge?.prompt_version === judgeVersion && Boolean(system.judge?.result));
  const aggregateRows = useMemo(() => {
    if (authorScope !== null) return [];
    const byMethod = new Map<string, Map<string, GenerationSystem[]>>();
    judgedSystems.forEach((system) => {
      if (!system.author || !system.judge?.result) return;
      const method = system.method_id || system.display_name || system.run_name;
      const byAuthor = byMethod.get(method) || new Map<string, GenerationSystem[]>();
      const runs = byAuthor.get(system.author) || [];
      runs.push(system);
      byAuthor.set(system.author, runs);
      byMethod.set(method, byAuthor);
    });
    return Array.from(byMethod.entries()).map(([method, byAuthor]) => {
      const groups: Record<string, number | null> = {};
      Object.keys(GROUP_LABELS).forEach((group) => {
        const authorMeans = Array.from(byAuthor.values()).map((runs) => {
          const values = runs
            .map((system) => system.judge?.result?.groups?.[group])
            .filter((value): value is number => typeof value === 'number');
          return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
        }).filter((value): value is number => value !== null);
        groups[group] = authorMeans.length
          ? authorMeans.reduce((sum, value) => sum + value, 0) / authorMeans.length
          : null;
      });
      const first = Array.from(byAuthor.values())[0]?.[0];
      return {
        method,
        name: first ? methodName(first) : method,
        authorCount: byAuthor.size,
        groups
      };
    });
  }, [authorScope, judgedSystems]);
  const matrixKeys = ['d1_stance_value', 'd2_argumentation', 'd3_lexicon_register', 'd4_tone_posture', 'd5_syntax_rhythm', 'd6_naturalness_artifacts'];
  const matrixLabels = ['D1 立场与世界观', 'D2 论证习惯', 'D3 词汇与语域', 'D4 语气与人格', 'D5 句法与节奏', 'D6 自然度与 AI 痕迹'];

  return (
    <div className="generation-overview">
      <header>
        <span>同一冻结评估集 · 可复现实验对比</span>
        <h1>生成质量总览</h1>
        <p>先选择冻结的评估集和评估版本，再横向比较不同生成方法。缺少结果的方法显示为“未运行”，不会被当成低分。</p>
      </header>
      <section className="generation-comparison-controls" aria-label="生成评估筛选">
        <label>
          <span>冻结评估集</span>
          <select value={datasetKey} onChange={(event) => setDatasetKey(event.target.value)}>
            {datasetOptions.map((entry) => (
              <option key={`${entry.id}:${entry.sha}`} value={`${entry.id}:${entry.sha}`}>
                {entry.id} · {entry.count} 版方法
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>评估体系</span>
          <select value={judgeVersion} onChange={(event) => setJudgeVersion(event.target.value)}>
            {judgeOptions.map((version) => <option key={version} value={version}>{version}</option>)}
          </select>
        </label>
        <div className="generation-comparison-context">
          <strong>{judgedSystems.length}/{filteredSystems.length}</strong>
          <span>已有该版本 Judge 结果</span>
        </div>
      </section>

      {authorScope === null ? (
        <section className="generation-aggregate-section" aria-label="跨作者汇总">
          <header>
            <div><strong>跨作者宏平均</strong><span>先在每位作者内部平均，再对作者平均，避免材料多的作者支配结果。</span></div>
            <small>{aggregateRows.length} 个方法 · 当前 {judgeVersion}</small>
          </header>
          <div className="generation-method-cards">
            {aggregateRows.length ? aggregateRows.map((row) => (
              <article className="generation-method-card generation-aggregate-card" key={row.method}>
                <header><div><span>跨作者汇总</span><strong>{row.name}</strong></div><em className="ready">{row.authorCount} 位作者</em></header>
                <div className="generation-method-meta">宏平均 · 不支持逐题打开或跨作者 AB 对比</div>
                <div className="generation-group-bars">
                  {Object.entries(GROUP_LABELS).map(([key, label]) => {
                    const value = row.groups[key] ?? null;
                    return <div key={key}><span>{label}</span><div className="generation-score-track"><i style={{ width: `${value === null ? 0 : Math.max(0, Math.min(100, value / 5 * 100))}%` }} /></div><b>{formatScore(value)}</b></div>;
                  })}
                </div>
              </article>
            )) : <p className="evaluation-scope-note">当前冻结数据集还没有能按同一 Judge 版本汇总的跨作者结果。</p>}
          </div>
        </section>
      ) : null}

      <section className="generation-method-cards" aria-label="方法总览">
        {filteredSystems.map((system) => {
          const result = system.judge?.prompt_version === judgeVersion ? system.judge?.result || null : null;
          const groups = result?.groups || {};
          return (
            <article className="generation-method-card" key={system.system_id}>
              <header>
                <div>
                  <span>{methodName(system)}</span>
                  <strong>{systemName(system)}</strong>
                </div>
                <em className={result ? 'ready' : 'missing'}>{result ? '已评估' : system.judge?.status === 'running' ? '评估中' : '未运行'}</em>
              </header>
              <div className="generation-method-meta">{system.model || '模型未记录'} · {system.item_count} 题 · {system.split || 'dev'}</div>
              <div className="generation-group-bars">
                {Object.entries(GROUP_LABELS).map(([key, label]) => {
                  const value = typeof groups[key] === 'number' ? groups[key] as number : null;
                  return (
                    <div key={key}>
                      <span>{label}</span>
                      <div className="generation-score-track"><i style={{ width: `${value === null ? 0 : Math.max(0, Math.min(100, value / 5 * 100))}%` }} /></div>
                      <b>{formatScore(value)}</b>
                    </div>
                  );
                })}
              </div>
              <footer>
                <button type="button" onClick={() => onOpen(system.system_id, 'rubric')}>人工 {system.human_progress?.completed || 0}/{system.human_progress?.total || 10}</button>
                <button type="button" className="generation-row-action" onClick={() => onOpen(system.system_id, 'judge')}>
                  {result ? '查看详情' : '打开 Judge'} <ChevronRight size={15} />
                </button>
              </footer>
            </article>
          );
        })}
      </section>

      <section className="generation-dimension-matrix">
        <header>
          <div><strong>D1–D6 维度对比</strong><span>同一评估版本下，各方法分别展示均值；满分 5。</span></div>
          <small>{selectedDataset?.id || `未选择评估集 · ${selectedSplitLabel}`}</small>
        </header>
        <div className="generation-matrix-scroll">
          <div className="generation-matrix" style={{ gridTemplateColumns: `minmax(190px, 1.4fr) repeat(${Math.max(filteredSystems.length, 1)}, minmax(145px, 1fr))` }}>
            <strong>维度</strong>
            {filteredSystems.map((system) => <strong key={system.system_id}>{methodName(system)}</strong>)}
            {matrixKeys.map((key, index) => (
              <div className="generation-matrix-row" key={key}>
                <span>{matrixLabels[index]}</span>
                {filteredSystems.map((system) => {
                  const value = system.judge?.prompt_version === judgeVersion ? dimensionMean(system, key) : null;
                  return (
                    <div className="generation-matrix-value" key={system.system_id}>
                      <b>{formatScore(value)}</b>
                      <i><span style={{ width: `${value === null ? 0 : Math.max(0, Math.min(100, value / 5 * 100))}%` }} /></i>
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}

function RubricView({
  workspace,
  item,
  scores,
  note,
  saving,
  ordinal,
  onScore,
  onNote,
  onSaveNote,
  onMove
}: {
  workspace: GenerationWorkspace;
  item: GenerationItem;
  scores: Record<string, number>;
  note: string;
  saving: boolean;
  ordinal: number;
  onScore: (key: string, score: number) => Promise<void>;
  onNote: (value: string) => void;
  onSaveNote: () => Promise<void>;
  onMove: (delta: number) => void;
}) {
  return (
    <div className="generation-review">
      <header className="generation-question">
        <span>人工六维 · {ordinal + 1}/{workspace.items.length}</span>
        <h1>{item.question}</h1>
      </header>
      <div className="generation-answer-pair">
        <AnswerPane label="作者原回答" text={item.gold_answer} />
        <AnswerPane label="系统回答" text={item.candidate_answer} />
      </div>
      <section className="generation-rubric-dock">
        <div className="generation-rubric-heading">
          <div><strong>这版回答在六个维度上有多接近作者？</strong><span>每个维度独立评分，点击后立即保存</span></div>
          <ItemNavigation ordinal={ordinal} total={workspace.items.length} onMove={onMove} />
        </div>
        <div className="generation-rubric-grid">
          {workspace.rubric.map((dimension) => (
            <div className="generation-rubric-row" key={dimension.key}>
              <details>
                <summary><b>{dimension.short}</b><span>{dimension.label}</span></summary>
                <p>{dimension.question}</p>
                <ol>{Object.entries(dimension.anchors).map(([score, anchor]) => <li key={score}><b>{score}</b>{anchor}</li>)}</ol>
              </details>
              <div className="generation-scale" aria-label={dimension.label}>
                {[1, 2, 3, 4, 5].map((score) => (
                  <button
                    key={score}
                    className={scores[dimension.key] === score ? 'selected' : ''}
                    type="button"
                    disabled={saving}
                    title={dimension.anchors[String(score)]}
                    onClick={() => void onScore(dimension.key, score)}
                  >{score}</button>
                ))}
              </div>
            </div>
          ))}
        </div>
        <details className="generation-note">
          <summary>补充判断依据（可选）</summary>
          <textarea value={note} onChange={(event) => onNote(event.target.value)} onBlur={() => void onSaveNote()} />
        </details>
      </section>
    </div>
  );
}

function PairwiseView({
  comparison,
  item,
  saving,
  ordinal,
  onVote,
  onMove
}: {
  comparison: GenerationComparison;
  item: GenerationComparisonItem;
  saving: boolean;
  ordinal: number;
  onVote: (choice: 'A' | 'B') => Promise<void>;
  onMove: (delta: number) => void;
}) {
  return (
    <div className="generation-review generation-pairwise">
      <header className="generation-question">
        <span>匿名 AB · {ordinal + 1}/{comparison.items.length}</span>
        <h1>{item.question}</h1>
      </header>
      <AnswerPane label="作者原回答" text={item.gold_answer} wide />
      <div className="generation-answer-pair candidates">
        <AnswerPane label={item.revealed ? `A · ${systemName(item.revealed.A)}` : '候选 A'} text={item.candidate_a} selected={item.choice === 'A'} />
        <AnswerPane label={item.revealed ? `B · ${systemName(item.revealed.B)}` : '候选 B'} text={item.candidate_b} selected={item.choice === 'B'} />
      </div>
      <footer className="generation-pair-dock">
        <div>
          <strong>哪一版更像作者本人会写出的回答？</strong>
          <span>{item.choice ? '已保存并揭示系统身份，可以修改选择' : '必须二选一；投票前不显示系统身份'}</span>
        </div>
        <div className="generation-vote-actions">
          <button className={item.choice === 'A' ? 'selected' : ''} type="button" disabled={saving} onClick={() => void onVote('A')}>选择 A</button>
          <button className={item.choice === 'B' ? 'selected' : ''} type="button" disabled={saving} onClick={() => void onVote('B')}>选择 B</button>
        </div>
        <ItemNavigation ordinal={ordinal} total={comparison.items.length} onMove={onMove} />
      </footer>
    </div>
  );
}

function JudgeView({
  system,
  workspace,
  job,
  onStart,
  onOpenItem
}: {
  system: GenerationSystem;
  workspace: GenerationWorkspace | null;
  job: GenerationJudgeJob | null;
  onStart: () => Promise<void>;
  onOpenItem: (itemId: string) => void;
}) {
  const active = job?.status === 'queued' || job?.status === 'running';
  const result = job?.result;
  const progress = job?.total_items ? (job.completed_items / job.total_items) * 100 : 0;
  return (
    <div className="generation-judge">
      <header>
        <span>Gold Judge · 固定 3 次独立评分</span>
        <h1>{systemName(system)}</h1>
        <p>{system.description || '逐题对照作者原回答，从六个维度独立打分。'} 这里不运行 LLM 两两选择，避免已观察到的位置偏差。当前评估集共 {workspace?.system.item_count || 0} 题，每题固定 3 次重复、3 组 rubric。</p>
      </header>
      <section className="judge-method-meta" aria-label="生成方法配置">
        <div><span>方法标识</span><strong>{system.method_id || system.writer_prompt}</strong></div>
        <div><span>Prompt 版本</span><strong>{system.prompt_version || '未记录'}</strong></div>
        <div><span>Writer 上下文</span><strong>{String(system.parameters?.writer_context_top_k ?? '—')} 篇</strong></div>
        <div><span>温度</span><strong>{String(system.parameters?.temperature ?? '—')}</strong></div>
        <div><span>模型</span><strong>{system.model || '未记录'}</strong></div>
        <div><span>代码版本</span><strong>{system.git_revision ? system.git_revision.slice(0, 10) : '未记录'}</strong></div>
      </section>
      <section className="judge-run-strip">
        <div>
          <strong>{job ? job.label : '尚未运行'}</strong>
          <span>{job?.status === 'completed' ? `完成于 ${job.completed_at || ''}` : job?.status === 'failed' ? job.error_message || '任务失败' : active ? `${job.completed_items}/${job.total_items} 题 · ${job.stage}` : '将创建可离开页面继续执行的后台任务'}</span>
        </div>
        {active ? <LoaderCircle className="spin" size={20} /> : (
          <button type="button" onClick={() => void onStart()}>{job?.status === 'completed' ? '重新使用同配置结果' : '开始评估'}</button>
        )}
        {active ? <div className="judge-progress"><i style={{ width: `${progress}%` }} /></div> : null}
      </section>
      {result ? (
        <>
          <section className="judge-group-summary">
            {Object.entries(result.groups).map(([key, value]) => (
              <div key={key}><span>{GROUP_LABELS[key] || key}</span><strong>{formatScore(value)}</strong><small>满分 5</small></div>
            ))}
          </section>
          <section className="judge-dimension-table">
            <div className="judge-table-head"><span>维度</span><span>均值</span><span>95% CI</span><span>完全一致</span><span>相差 ≤ 1</span><span>平均极差</span></div>
            {Object.entries(result.dimensions).map(([key, metric], index) => (
              <div className="judge-table-row" key={key}>
                <strong>D{index + 1}</strong>
                <b>{formatScore(metric.mean)}</b>
                <span>{formatScore(metric.ci95[0])}–{formatScore(metric.ci95[1])}</span>
                <span>{metric.exact_agreement === null ? '—' : `${Math.round(metric.exact_agreement * 100)}%`}</span>
                <span>{metric.within_one_agreement === null ? '—' : `${Math.round(metric.within_one_agreement * 100)}%`}</span>
                <span>{formatScore(metric.mean_range)}</span>
              </div>
            ))}
          </section>
          <section className="judge-question-links">
            <header>
              <div>
                <strong>逐题详细报告</strong>
                <span>选择一道题，直接打开该题的回答、六维分数和三次 Judge 记录。</span>
              </div>
              <small>{workspace?.items.length || result.item_count} 题</small>
            </header>
            <div className="judge-question-link-list">
              {(workspace?.items || []).map((entry) => (
                <a
                  key={entry.item_id}
                  href={generationItemHref(system.system_id, entry.item_id)}
                  onClick={(event) => {
                    event.preventDefault();
                    onOpenItem(entry.item_id);
                  }}
                >
                  <span>{entry.ordinal}</span>
                  <strong>{entry.question}</strong>
                  <ChevronRight size={17} />
                </a>
              ))}
            </div>
          </section>
        </>
      ) : (
        <div className="judge-empty-result"><Bot size={23} /><strong>还没有 Judge 结果</strong><span>任务启动后可以切回 Chat 或关闭当前页面，后端仍会继续。</span></div>
      )}
    </div>
  );
}

function generationItemHref(systemId: string, itemId: string): string {
  const params = new URLSearchParams(window.location.search);
  params.set('generationView', 'judge');
  params.set('generationSystem', systemId);
  params.set('generationItem', itemId);
  return `${window.location.pathname}?${params.toString()}`;
}

function JudgeItemDetail({
  system,
  workspace,
  item,
  ordinal,
  onBack,
  onMove
}: {
  system: GenerationSystem;
  workspace: GenerationWorkspace;
  item: GenerationItem;
  ordinal: number;
  onBack: () => void;
  onMove: (delta: number) => void;
}) {
  const dimensions = Object.entries(item.judge?.dimensions || {});
  return (
    <div className="generation-judge generation-judge-detail">
      <header className="judge-detail-header">
        <button type="button" onClick={onBack}><ChevronLeft size={17} />返回方法报告</button>
        <span>逐题 Judge · {ordinal + 1}/{workspace.items.length}</span>
        <h1>{item.question}</h1>
        <p>{methodName(system)} · {String(system.parameters?.writer_context_top_k ?? '—')} 篇上下文 · 固定 3 次评分</p>
      </header>
      <section className="judge-detail-answers">
        <AnswerPane label="作者原回答" text={item.gold_answer} />
        <AnswerPane label="系统回答" text={item.candidate_answer} />
      </section>
      <section className="judge-detail-dimensions">
        <header><strong>六维评分</strong><span>中位数为最终分；下方保留三次独立评分，便于检查稳定性。</span></header>
        {dimensions.map(([key, metric], index) => (
          <article key={key} className="judge-detail-dimension">
            <div className="judge-detail-dimension-title">
              <strong>D{index + 1}</strong>
              <b>{formatScore(metric.score)}</b>
              <span>完全一致 {metric.exact_agreement == null ? '—' : `${Math.round(metric.exact_agreement * 100)}%`} · 相差 ≤ 1 {metric.within_one_agreement == null ? '—' : `${Math.round(metric.within_one_agreement * 100)}%`} · 极差 {formatScore(metric.range)}</span>
            </div>
            <div className="judge-detail-repeats">
              {(metric.raw_ratings || []).map((rating) => <span key={rating.repeat}>第 {rating.repeat} 次：{rating.score == null ? '失败' : rating.score}</span>)}
            </div>
            <p>{metric.reason || '暂无理由'}</p>
            {metric.candidate_evidence?.length ? <small>系统证据：{metric.candidate_evidence.join('；')}</small> : null}
          </article>
        ))}
      </section>
      <ItemNavigation ordinal={ordinal} total={workspace.items.length} onMove={onMove} />
    </div>
  );
}

function AnswerPane({ label, text, wide = false, selected = false }: { label: string; text: string; wide?: boolean; selected?: boolean }) {
  return (
    <article className={`generation-answer-pane ${wide ? 'wide' : ''} ${selected ? 'selected' : ''}`}>
      <header><strong>{label}</strong>{selected ? <Check size={15} /> : null}</header>
      <div>{text}</div>
    </article>
  );
}

function ItemNavigation({ ordinal, total, onMove }: { ordinal: number; total: number; onMove: (delta: number) => void }) {
  return (
    <div className="generation-item-navigation">
      <button type="button" disabled={ordinal <= 0} onClick={() => onMove(-1)}><ChevronLeft size={16} />上一题</button>
      <span>{ordinal + 1}/{total}</span>
      <button type="button" disabled={ordinal >= total - 1} onClick={() => onMove(1)}>下一题<ChevronRight size={16} /></button>
    </div>
  );
}
