import { type ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import { BarChart3, ExternalLink, ListChecks, LoaderCircle, PanelLeftClose, PanelLeftOpen } from 'lucide-react';
import {
  fetchRetrievalLlmLabelSets,
  fetchRetrievalLlmQuery,
  fetchRetrievalLlmWorkspace,
  fetchRetrievalPools,
  RetrievalLlmCandidate,
  RetrievalLlmLabelSet,
  RetrievalLlmQuery,
  RetrievalLlmWorkspace,
  RetrievalPoolSummary
} from './api';

const ROUTE_ORDER = [
  'raw_dense',
  'raw_sparse',
  'raw_hybrid_rrf',
  'transformed_rrf',
  'raw_bm25',
  'transformed_dense_bm25_rrf'
];

const ROUTE_LABELS: Record<string, string> = {
  raw_dense: '原问题 Dense',
  raw_sparse: '原问题 Sparse',
  raw_hybrid_rrf: '原问题 Dense + Sparse RRF',
  transformed_rrf: '四路变换 Dense + Sparse RRF',
  raw_bm25: '原问题 BM25',
  transformed_dense_bm25_rrf: '四路变换 Dense + BM25 RRF'
};

type ReportSplit = 'all' | 'dev' | 'test';

const SPLIT_NAMES: Record<ReportSplit, string> = {
  all: '全部',
  dev: 'Dev',
  test: 'Test'
};

const AXIS_LABELS: Record<string, string> = {
  score: '问题相关性',
  content_support: '内容支撑',
  persona_expression_support: '作者表达支撑'
};

function candidateHeading(candidate: RetrievalLlmCandidate): string {
  if (candidate.kind === 'pin') return '想法';
  if (candidate.title.trim()) return candidate.title.trim();
  if (candidate.kind === 'article') return '文章';
  if (candidate.kind === 'answer') return '回答';
  return '历史内容';
}

function metricValue(value: number | null): string {
  return value === null ? '—' : value.toFixed(3);
}

function scoreLabel(score: RetrievalLlmCandidate['score']): string {
  if (score === 2) return '明显有用';
  if (score === 1) return '有一定帮助';
  if (score === 0) return '无用';
  return '未完成';
}

function scoreClass(score: RetrievalLlmCandidate['score']): string {
  return score === 2 ? 'score-2' : score === 1 ? 'score-1' : score === 0 ? 'score-0' : 'score-null';
}

export function RetrievalLlmReport({ viewSwitcher, authorScope }: { viewSwitcher: ReactNode; authorScope: string | null }) {
  const [pools, setPools] = useState<RetrievalPoolSummary[]>([]);
  const [poolId, setPoolId] = useState('');
  const [labelSets, setLabelSets] = useState<RetrievalLlmLabelSet[]>([]);
  const [labelSet, setLabelSet] = useState('');
  const [axis, setAxis] = useState('score');
  const [workspace, setWorkspace] = useState<RetrievalLlmWorkspace | null>(null);
  const [itemId, setItemId] = useState('');
  const [query, setQuery] = useState<RetrievalLlmQuery | null>(null);
  const [order, setOrder] = useState<'relevance' | 'retrieval'>('relevance');
  const [split, setSplit] = useState<ReportSplit>('all');
  const [cutoff, setCutoff] = useState(3);
  const [reportSection, setReportSection] = useState<'overview' | 'labels'>(() =>
    localStorage.getItem('pf-retrieval-report-section') === 'labels' ? 'labels' : 'overview'
  );
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem('pf-retrieval-llm-sidebar') === 'true');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let active = true;
    if (!poolId) {
      setLabelSets([]);
      setLabelSet('');
      return;
    }
    setLoading(true);
    fetchRetrievalLlmLabelSets(poolId)
      .then((items) => {
        if (!active) return;
        setLabelSets(items);
        setLabelSet((current) => items.some((item) => item.label_set === current) ? current : items[0]?.label_set || '');
      })
      .catch((reason) => active && setError(String((reason as Error).message || reason)))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [poolId]);

  useEffect(() => {
    let active = true;
    if (!poolId || !labelSet) {
      setWorkspace(null);
      setItemId('');
      return;
    }
    const selected = labelSets.find((item) => item.label_set === labelSet);
    const availableAxes = Object.keys(selected?.axes || { score: {} });
    const requestedAxis = availableAxes.includes(axis) ? axis : selected?.default_axis || availableAxes[0] || 'score';
    if (requestedAxis !== axis) setAxis(requestedAxis);
    setLoading(true);
    fetchRetrievalLlmWorkspace(poolId, labelSet, requestedAxis)
      .then((next) => {
        if (!active) return;
        setWorkspace(next);
        setItemId((current) => next.queries.some((row) => row.item_id === current) ? current : next.queries[0]?.item_id || '');
      })
      .catch((reason) => active && setError(String((reason as Error).message || reason)))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [axis, labelSet, labelSets, poolId]);

  useEffect(() => {
    let active = true;
    if (reportSection !== 'labels' || !poolId || !labelSet || !itemId) {
      setQuery(null);
      return;
    }
    setLoading(true);
    fetchRetrievalLlmQuery(poolId, labelSet, itemId, axis)
      .then((next) => active && setQuery(next))
      .catch((reason) => active && setError(String((reason as Error).message || reason)))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [axis, poolId, labelSet, itemId, reportSection]);

  useEffect(() => {
    let active = true;
    fetchRetrievalPools(authorScope)
      .then((items) => {
        if (!active) return;
        setPools(items);
        setPoolId((current) => items.some((item) => item.pool_id === current) ? current : items[0]?.pool_id || '');
      })
      .catch((reason) => active && setError(String((reason as Error).message || reason)));
    return () => {
      active = false;
    };
  }, [authorScope]);

  const selectedLabelSet = useMemo(
    () => labelSets.find((item) => item.label_set === labelSet) || labelSets[0],
    [labelSet, labelSets]
  );
  const availableSplits = useMemo<ReportSplit[]>(() => {
    const selected = (selectedLabelSet?.selected_splits || []).filter(
      (value): value is 'dev' | 'test' => value === 'dev' || value === 'test'
    );
    if (selected.length) return selected;
    const present = new Set((workspace?.queries || []).map((row) => row.split));
    const result: ReportSplit[] = ['all'];
    if (present.has('dev')) result.push('dev');
    if (present.has('test')) result.push('test');
    return result;
  }, [selectedLabelSet, workspace]);
  useEffect(() => {
    if (!availableSplits.includes(split)) setSplit(availableSplits[0] || 'all');
  }, [availableSplits, split]);
  const splitLabel = (value: ReportSplit) => {
    const rows = workspace?.queries || [];
    const count = value === 'all' ? rows.length : rows.filter((row) => row.split === value).length;
    return `${SPLIT_NAMES[value]} ${count} 题`;
  };
  const selectedMetrics = useMemo(() => {
    if (!workspace) return null;
    if (split === 'all') return workspace.metrics;
    return workspace.metrics.splits?.[split] || workspace.metrics;
  }, [split, workspace]);
  const currentMetrics = useMemo(() => selectedMetrics?.routes || {}, [selectedMetrics]);
  const availableCutoffs = selectedMetrics?.cutoffs?.length ? selectedMetrics.cutoffs : [selectedMetrics?.cutoff || 3];
  const visibleQueries = useMemo(() => {
    const rows = workspace?.queries || [];
    return split === 'all' ? rows : rows.filter((row) => row.split === split);
  }, [split, workspace]);
  useEffect(() => {
    if (!visibleQueries.length) return;
    if (!visibleQueries.some((row) => row.item_id === itemId)) setItemId(visibleQueries[0].item_id);
  }, [itemId, visibleQueries]);
  const orderedCandidates = useMemo(() => {
    if (!query) return [];
    if (order === 'relevance') return query.candidates;
    return [...query.candidates].sort((left, right) => {
      if (left.best_route_rank !== right.best_route_rank) return left.best_route_rank - right.best_route_rank;
      return left.relevance_order - right.relevance_order;
    });
  }, [order, query]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: 0 });
  }, [itemId, reportSection]);

  const selectReportSection = (next: 'overview' | 'labels') => {
    setReportSection(next);
    localStorage.setItem('pf-retrieval-report-section', next);
  };

  if (!poolId || !pools.length) {
    return <div className="evaluation-empty"><strong>还没有冻结的检索候选池</strong><p>先生成冻结候选池，再完成机器相关性标注。</p></div>;
  }

  const availableAxes = Object.keys(selectedLabelSet?.axes || { score: {} });
  const recallScope = selectedMetrics?.recall_scope || workspace?.pool.recall_scope || 'six_route_candidate_union';
  const eligibleParentCount = workspace?.pool.counts?.eligible_parents_per_query;
  const recallScopeLabel = recallScope === 'eligible_author_corpus_before_cutoff'
    ? `时间切分前全部${eligibleParentCount ? ` ${eligibleParentCount} 篇` : ''}作者语料`
    : '冻结六路候选并集';
  const stabilityProgress = selectedLabelSet?.progress;
  return (
    <section className={`evaluation-workspace retrieval-llm-workspace ${collapsed ? 'rail-collapsed' : ''}`}>
      {!collapsed ? (
        <aside className="evaluation-query-rail retrieval-llm-rail">
          <div className="evaluation-rail-heading">
            <strong>RAG 评估</strong>
            <button type="button" title="收起评估栏" onClick={() => { setCollapsed(true); localStorage.setItem('pf-retrieval-llm-sidebar', 'true'); }}>
              <PanelLeftClose size={17} />
            </button>
          </div>
          {viewSwitcher}
          <nav className="retrieval-report-navigation" aria-label="LLM 检索报告页面">
            <button type="button" className={reportSection === 'overview' ? 'active' : ''} onClick={() => selectReportSection('overview')}><BarChart3 size={17} />指标总览</button>
            <button type="button" className={reportSection === 'labels' ? 'active' : ''} onClick={() => selectReportSection('labels')}><ListChecks size={17} />逐题标注</button>
          </nav>
          <div className="evaluation-dataset-controls">
            <label>候选池<select value={poolId} onChange={(event) => setPoolId(event.target.value)}>
              {pools.map((pool) => {
                const labelSets = pool.llm_label_sets || [];
                const completedReportCount = labelSets.filter((item) => item.status === 'completed').length;
                const reportStatus = completedReportCount
                  ? `已有 ${completedReportCount} 份完整标注`
                  : labelSets.length
                    ? '部分标注'
                    : '未标注';
                return <option key={pool.pool_id} value={pool.pool_id}>
                  {pool.author ? `${pool.author} · ` : '未归属/旧数据 · '}{pool.display_name || pool.dataset_id} · {pool.candidate_count} 条 · {reportStatus}
                </option>;
              })}
            </select></label>
            {labelSets.length ? <label>标注版本<select value={labelSet} onChange={(event) => setLabelSet(event.target.value)}>
              {labelSets.map((item) => <option key={item.label_set} value={item.label_set}>{item.label_set} · {item.completed}/{item.total}</option>)}
            </select></label> : <div className="retrieval-overview-note">当前候选池暂无 LLM 标注<br /><small>候选池仍保留在这里，方便切换到其他已有报告。</small></div>}
            {labelSets.length && availableAxes.length > 1 ? <label>评估维度<select value={axis} onChange={(event) => setAxis(event.target.value)}>
              {availableAxes.map((value) => <option key={value} value={value}>{selectedLabelSet?.axes?.[value]?.label || AXIS_LABELS[value] || value}</option>)}
            </select></label> : null}
          </div>
          {reportSection === 'labels' && labelSets.length ? (
            <nav className="evaluation-query-list" aria-label="LLM 检索评估问题">
              {visibleQueries.map((row) => (
                <button key={row.item_id} className={row.item_id === itemId ? 'active' : ''} type="button" onClick={() => setItemId(row.item_id)}>
                  <span className="query-ordinal">{row.ordinal}</span>
                  <span className="query-list-copy"><strong>{row.query}</strong><small>{row.labeled_count} / {row.candidate_count}</small></span>
                </button>
              ))}
            </nav>
          ) : reportSection === 'labels' ? <div className="retrieval-overview-note">当前候选池暂无逐题标注。</div> : <div className="retrieval-overview-note">总览展示当前数据划分内所有问题的平均检索指标，不随单题变化。</div>}
        </aside>
      ) : null}
      <main className="evaluation-reader retrieval-llm-reader">
        {collapsed ? <button className="evaluation-rail-reveal" type="button" title="展开评估栏" onClick={() => { setCollapsed(false); localStorage.setItem('pf-retrieval-llm-sidebar', 'false'); }}><PanelLeftOpen size={18} /><span>RAG 评估</span></button> : null}
        <header className="evaluation-question-header retrieval-llm-header">
          <span>{reportSection === 'overview' ? '检索评估 · 聚合结果' : labelSets.length ? `${splitLabel(split)} · ${visibleQueries.findIndex((row) => row.item_id === itemId) + 1 || '-'} / ${visibleQueries.length || 0}` : '逐题标注'}</span>
          <h1>{!labelSets.length ? '当前候选池暂无 LLM 报告' : reportSection === 'overview' ? '六路检索指标总览' : query?.query || '正在读取问题'}</h1>
          <div className="retrieval-llm-meta">
            {!labelSets.length ? '请从左侧切换到已有标注的候选池，或先创建 LLM 检索标注任务。' : <>{reportSection === 'overview' ? `${splitLabel(split)}的逐题指标平均值` : `标注者：${selectedLabelSet?.model || '未知'}`} · 稳定完成 {selectedLabelSet?.completed}/{selectedLabelSet?.total}
            {selectedLabelSet?.status !== 'completed' && stabilityProgress ? (
              <span>
                {' '}· 已有首遍 {stabilityProgress.pass1_completed ?? 0}
                {' '}· 待首遍 {stabilityProgress.missing_pass1 ?? 0}
                {' '}· 待复评 {(stabilityProgress.pending_pass2 ?? 0) + (stabilityProgress.pending_pass3 ?? 0)}
              </span>
            ) : null}
            </>}
          </div>
        </header>
        <div className="retrieval-llm-scroll" ref={scrollRef}>
          {!labelSets.length ? <div className="evaluation-empty retrieval-report-unavailable">
            <strong>这个候选池已经建立，但还没有机器标注</strong>
            <p>候选池本身不是报告。请从左侧切换到带有“已有标注”的候选池，或者在“评估任务”中先创建标注。</p>
          </div> : reportSection === 'overview' ? <>
            <div className="retrieval-metric-controls">
              <div className="retrieval-segment" role="tablist" aria-label="数据划分">
                {availableSplits.map((value) => <button type="button" className={split === value ? 'active' : ''} key={value} onClick={() => setSplit(value)}>{splitLabel(value)}</button>)}
              </div>
              <label>观察深度 K
                <select value={cutoff} onChange={(event) => setCutoff(Number(event.target.value))}>
                  {availableCutoffs.map((value) => <option key={value} value={value}>Top {value}</option>)}
                </select>
              </label>
              <span>{selectedMetrics?.relevant_candidate_count ?? 0} 个有用 query-parent 对 · Recall 分母：{recallScopeLabel}</span>
            </div>
            <div className="retrieval-aggregate-explanation">下面六组数字是 <strong>{splitLabel(split)}</strong> 在当前评估维度、Top {cutoff} 下的逐题平均结果，不属于任何一道单题。</div>
            {workspace?.comparison ? <div className="retrieval-comparison-summary">
              <strong>Gold-aware 相比旧 Query-only Judge 改判 {workspace.comparison.changed_count} / {workspace.comparison.total}</strong>
              <span>其中旧版 0 分、Gold-aware 改为 1/2 分：{workspace.comparison.v1_zero_to_v2_positive} 对</span>
            </div> : null}
            <div className="retrieval-llm-metrics">
              {ROUTE_ORDER.filter((route) => currentMetrics[route]).map((route) => {
                const base = currentMetrics[route];
                const metric = base.by_cutoff?.[String(cutoff)] || base;
                return (
                <div className="retrieval-llm-metric" key={route}>
                  <strong>{ROUTE_LABELS[route] || route}</strong>
                  <span title="至少命中一篇有用材料的问题比例">Hit@{cutoff} <b>{metricValue(metric.hit_at_k)}</b></span>
                  <span title="第一篇有用材料排名倒数的平均值">MRR@{cutoff} <b>{metricValue(metric.mrr_at_k)}</b></span>
                  <span title="同时考虑 0/1/2 相关等级与排序位置">nDCG@{cutoff} <b>{metricValue(metric.ndcg_at_k)}</b></span>
                  <span title="前 K 篇中有用材料所占比例">Precision@{cutoff} <b>{metricValue(metric.precision_at_k)}</b></span>
                  <span title="前 K 篇覆盖完整相关材料集合的比例">Recall@{cutoff} <b>{metricValue(metric.recall_at_k)}</b></span>
                  <span title="前 K 位平均准确率，兼顾多个相关结果的排序位置">MAP@{cutoff} <b>{metricValue(metric.map_at_k ?? null)}</b></span>
                </div>
              );})}
            </div>
          </> : <>
            {query?.gold_answer ? <details className="retrieval-gold-panel">
              <summary>查看作者原回答与判断锚点</summary>
              <div className="retrieval-gold-answer">{query.gold_answer}</div>
              {query.gold_units ? <div className="retrieval-gold-units">
                {Object.entries(query.gold_units).map(([kind, units]) => (
                  <section key={kind}><strong>{kind}</strong>{units.map((unit) => <p key={unit.id || unit.text}><b>{unit.id}</b>{unit.text}</p>)}</section>
                ))}
              </div> : null}
            </details> : null}
            <div className="retrieval-llm-toolbar">
              <strong>{query?.candidate_count || 0} 个候选材料</strong>
              <span>相关性顺序按 Judge 分数展示；检索顺序用于核对六路实际排名。</span>
              <div className="retrieval-order-switch"><button className={order === 'relevance' ? 'active' : ''} type="button" onClick={() => setOrder('relevance')}>相关性顺序</button><button className={order === 'retrieval' ? 'active' : ''} type="button" onClick={() => setOrder('retrieval')}>检索顺序</button></div>
            </div>
            <div className="retrieval-llm-list">
              {orderedCandidates.map((candidate, index) => (
            <article className="retrieval-llm-card" key={candidate.parent_id}>
              <div className="retrieval-llm-card-heading">
                <span className="retrieval-llm-rank">{order === 'relevance' ? candidate.relevance_order : index + 1}</span>
                <div><strong>{candidateHeading(candidate)}</strong><small>{candidate.kind} · 最佳路由 #{candidate.best_route_rank === 1000000000 ? '—' : candidate.best_route_rank} · {candidate.route_count} 路命中</small></div>
                <span className={`retrieval-score ${scoreClass(candidate.score)}`}>{candidate.score === null ? '—' : `${candidate.score} · ${scoreLabel(candidate.score)}`}</span>
                {candidate.url ? <a href={candidate.url} target="_blank" rel="noreferrer" title="打开知乎原文"><ExternalLink size={16} /></a> : null}
              </div>
              <details open={index < 3}>
                <summary>查看完整材料与判断依据</summary>
                <div className="retrieval-llm-card-body">{candidate.text}</div>
                <div className="retrieval-llm-card-evidence">
                  {candidate.axis_scores && Object.keys(candidate.axis_scores).length > 1 ? <div className="retrieval-axis-scores">
                    {Object.entries(candidate.axis_scores).map(([key, value]) => <span key={key}>{AXIS_LABELS[key] || key}<b>{value}</b></span>)}
                  </div> : null}
                  <b>判断：</b>{candidate.reason || '无'}
                  {candidate.evidence ? <><br /><b>{AXIS_LABELS[axis] || '当前维度'}证据：</b>{candidate.evidence}</> : null}
                  {axis === 'content_support' && candidate.content_gold_unit_ids?.length ? <><br /><b>对应 Gold 单元：</b>{candidate.content_gold_unit_ids.join('、')}</> : null}
                  {axis === 'persona_expression_support' && candidate.persona_gold_unit_ids?.length ? <><br /><b>对应 Gold 单元：</b>{candidate.persona_gold_unit_ids.join('、')}</> : null}
                  {candidate.repeat_count ? <><br /><b>稳定性：</b>{candidate.repeat_count} 次判分{candidate.exact_agreement === true ? '，完全一致' : candidate.exact_agreement === false ? '，中位数聚合' : ''}</> : null}
                </div>
                <div className="retrieval-route-tags">{ROUTE_ORDER.filter((route) => candidate.route_ranks[route]).map((route) => <span key={route}>{ROUTE_LABELS[route] || route} · #{candidate.route_ranks[route].rank}</span>)}</div>
              </details>
            </article>
              ))}
            </div>
          </>}
        </div>
        {loading ? <div className="retrieval-llm-loading"><LoaderCircle className="spin" size={18} />正在读取</div> : null}
        {error ? <div className="evaluation-error">{error}</div> : null}
      </main>
    </section>
  );
}
