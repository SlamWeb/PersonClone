import { type ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import { BarChart3, ExternalLink, Globe2, ListChecks, LoaderCircle, PanelLeftClose, PanelLeftOpen } from 'lucide-react';
import {
  fetchRetrievalGlobalReport,
  fetchRetrievalLlmLabelSets,
  fetchRetrievalLlmQuery,
  fetchRetrievalLlmWorkspace,
  fetchRetrievalPools,
  RetrievalLlmCandidate,
  RetrievalLlmLabelSet,
  RetrievalLlmQuery,
  RetrievalLlmWorkspace,
  RetrievalGlobalReport,
  RetrievalPoolSummary,
  RetrievalRankingSummary
} from './api';

const ROUTE_ORDER = [
  'raw_dense',
  'raw_sparse',
  'raw_hybrid_rrf',
  'raw_hybrid_rrf_reranked',
  'transformed_dense_rrf',
  'transformed_rrf',
  'transformed_rrf_reranked',
  'raw_bm25',
  'transformed_dense_bm25_rrf',
  'transformed_dense_bm25_rrf_reranked'
];

const ROUTE_LABELS: Record<string, string> = {
  raw_dense: '原问题 Dense',
  raw_sparse: '原问题 Sparse',
  raw_hybrid_rrf: '原问题 Dense + Sparse RRF',
  raw_hybrid_rrf_reranked: '原问题 Hybrid + BGE Reranker',
  transformed_dense_rrf: '四路变换 Dense RRF',
  transformed_rrf: '四路变换 Dense + Sparse RRF',
  transformed_rrf_reranked: '四路变换 Hybrid + BGE Reranker',
  raw_bm25: '原问题 BM25',
  transformed_dense_bm25_rrf: '四路变换 Dense + BM25 RRF',
  transformed_dense_bm25_rrf_reranked: '四路变换 Dense + BM25 + BGE Reranker'
};

const RERANK_BASELINES: Record<string, string> = {
  raw_hybrid_rrf_reranked: 'raw_hybrid_rrf',
  transformed_rrf_reranked: 'transformed_rrf',
  transformed_dense_bm25_rrf_reranked: 'transformed_dense_bm25_rrf'
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

function metricDelta(value: number | null | undefined, baseline: number | null | undefined): string | null {
  if (typeof value !== 'number' || typeof baseline !== 'number') return null;
  const delta = value - baseline;
  return `${delta >= 0 ? '+' : ''}${delta.toFixed(3)}`;
}

function countValue(value: number | null | undefined): string {
  return typeof value === 'number' ? String(value) : '—';
}

function recallDistribution(metric: Record<string, unknown> | null | undefined, prefix: 'useful' | 'strong'): string {
  if (!metric) return '分布：—';
  const value = (key: string) => metricValue(typeof metric[key] === 'number' ? metric[key] as number : null);
  const count = (key: string) => countValue(typeof metric[key] === 'number' ? metric[key] as number : null);
  return `${prefix === 'useful' ? 'Useful' : 'Strong'} Recall 分布 ${value(`${prefix}_recall_min_at_k`)} / ${value(`${prefix}_recall_median_at_k`)} / ${value(`${prefix}_recall_max_at_k`)} · 0分题 ${count(`${prefix}_recall_zero_query_count`)} · 1分题 ${count(`${prefix}_recall_one_query_count`)}`;
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

function commonRankingDepth(snapshot: RetrievalRankingSummary | null | undefined): number {
  const depths = Object.values(snapshot?.actual_depth_by_route || {}).filter((value) => value > 0);
  return depths.length ? Math.min(...depths) : snapshot?.expected_depth || snapshot?.requested_depth || 0;
}

export function RetrievalLlmReport({ viewSwitcher, authorScope }: { viewSwitcher: ReactNode; authorScope: string | null }) {
  const [pools, setPools] = useState<RetrievalPoolSummary[]>([]);
  const [poolId, setPoolId] = useState('');
  const [labelSets, setLabelSets] = useState<RetrievalLlmLabelSet[]>([]);
  const [labelSet, setLabelSet] = useState('');
  const [rankingId, setRankingId] = useState('');
  const [axis, setAxis] = useState('score');
  const [workspace, setWorkspace] = useState<RetrievalLlmWorkspace | null>(null);
  const [itemId, setItemId] = useState('');
  const [query, setQuery] = useState<RetrievalLlmQuery | null>(null);
  const [order, setOrder] = useState<'relevance' | 'retrieval' | 'route'>('relevance');
  const [queryRoute, setQueryRoute] = useState('');
  const [queryCutoff, setQueryCutoff] = useState(10);
  const [split, setSplit] = useState<ReportSplit>('all');
  const [ndcgCutoff, setNdcgCutoff] = useState(10);
  const [precisionCutoff, setPrecisionCutoff] = useState(20);
  const [recallCutoff, setRecallCutoff] = useState(50);
  const [reportSection, setReportSection] = useState<'overview' | 'global' | 'labels'>(() => {
    const stored = localStorage.getItem('pf-retrieval-report-section');
    return stored === 'labels' || stored === 'global' ? stored : 'overview';
  });
  const [globalReport, setGlobalReport] = useState<RetrievalGlobalReport | null>(null);
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem('pf-retrieval-llm-sidebar') === 'true');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let active = true;
    if (reportSection === 'global') {
      setLabelSets([]);
      setLabelSet('');
      setWorkspace(null);
      setQuery(null);
      return () => {
        active = false;
      };
    }
    if (!poolId) {
      setLabelSets([]);
      setLabelSet('');
      return;
    }
    setError('');
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
  }, [poolId, reportSection]);

  const selectedPool = useMemo(
    () => pools.find((item) => item.pool_id === poolId) || null,
    [poolId, pools]
  );
  const rankingSnapshots = selectedPool?.ranking_snapshots || [];
  const completedRankings = rankingSnapshots.filter((item) => item.status === 'completed');

  useEffect(() => {
    setRankingId((current) => completedRankings.some((item) => item.ranking_id === current)
      ? current
      : completedRankings[0]?.ranking_id || '');
  }, [poolId, completedRankings.map((item) => item.ranking_id).join(',')]);

  useEffect(() => {
    let active = true;
    if (reportSection === 'global') {
      setWorkspace(null);
      setItemId('');
      return () => {
        active = false;
      };
    }
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
    fetchRetrievalLlmWorkspace(poolId, labelSet, requestedAxis, rankingId || undefined)
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
  }, [axis, labelSet, labelSets, poolId, rankingId, reportSection]);

  useEffect(() => {
    let active = true;
    if (reportSection !== 'labels' || !poolId || !labelSet || !itemId) {
      setQuery(null);
      return;
    }
    setLoading(true);
    fetchRetrievalLlmQuery(poolId, labelSet, itemId, axis, rankingId || undefined)
      .then((next) => active && setQuery(next))
      .catch((reason) => active && setError(String((reason as Error).message || reason)))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [axis, poolId, labelSet, itemId, rankingId, reportSection]);

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

  useEffect(() => {
    let active = true;
    if (reportSection !== 'global') return () => { active = false; };
    const requestedAxis = axis === 'score' ? 'content_support' : axis;
    if (requestedAxis !== axis) setAxis(requestedAxis);
    setGlobalReport(null);
    setError('');
    setLoading(true);
    fetchRetrievalGlobalReport(requestedAxis, split)
      .then((next) => {
        if (!active) return;
        setGlobalReport(next);
        if (next.active_axis && next.active_axis !== requestedAxis) setAxis(next.active_axis);
        if (next.available_splits.length && !next.available_splits.includes(split)) {
          setSplit(next.available_splits[0]);
        }
      })
      .catch((reason) => active && setError(String((reason as Error).message || reason)))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [axis, reportSection, split]);

  const selectedLabelSet = useMemo(
    () => labelSets.find((item) => item.label_set === labelSet) || labelSets[0],
    [labelSet, labelSets]
  );
  const localAvailableSplits = useMemo<ReportSplit[]>(() => {
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
  const availableSplits = reportSection === 'global'
    ? (globalReport?.available_splits || ['all'])
    : localAvailableSplits;
  useEffect(() => {
    if (!availableSplits.includes(split)) setSplit(availableSplits[0] || 'all');
  }, [availableSplits, split]);
  const splitLabel = (value: ReportSplit) => {
    if (reportSection === 'global') return value === 'all' ? '全部作者' : SPLIT_NAMES[value];
    const rows = workspace?.queries || [];
    const count = value === 'all' ? rows.length : rows.filter((row) => row.split === value).length;
    return `${SPLIT_NAMES[value]} ${count} 题`;
  };
  const selectedMetrics = useMemo(() => {
    if (reportSection === 'global') return globalReport?.metrics || null;
    if (!workspace) return null;
    if (split === 'all') return workspace.metrics;
    return workspace.metrics.splits?.[split] || workspace.metrics;
  }, [globalReport, reportSection, split, workspace]);
  const currentMetrics = useMemo(() => selectedMetrics?.routes || {}, [selectedMetrics]);
  const availableCutoffs = selectedMetrics?.cutoffs?.length ? selectedMetrics.cutoffs : [selectedMetrics?.cutoff || 3];
  const metricGroups = selectedMetrics?.cutoff_groups;
  const supportedDepth = reportSection === 'global'
    ? Number.POSITIVE_INFINITY
    : workspace?.ranking ? commonRankingDepth(workspace.ranking) : Number.POSITIVE_INFINITY;
  const rankingCutoffs = (metricGroups?.ndcg?.length ? metricGroups.ndcg : availableCutoffs.filter((value) => value <= 30))
    .filter((value) => value <= supportedDepth);
  const precisionMetricCutoffs = (metricGroups?.precision?.length ? metricGroups.precision : rankingCutoffs)
    .filter((value) => value <= supportedDepth);
  const deepRecallCutoffs = (metricGroups?.recall?.length ? metricGroups.recall : availableCutoffs.filter((value) => value >= 10))
    .filter((value) => value <= supportedDepth);
  const recallCutoffs = deepRecallCutoffs.length
    ? deepRecallCutoffs
    : availableCutoffs.filter((value) => value <= supportedDepth);
  const cutoffSignature = availableCutoffs.join(',');
  useEffect(() => {
    const keepOrNearest = (current: number, values: number[], fallback: number) => {
      if (values.includes(current)) return current;
      const belowFallback = values.filter((value) => value <= fallback);
      return belowFallback[belowFallback.length - 1] || values[values.length - 1] || fallback;
    };
    setNdcgCutoff((current) => keepOrNearest(current, rankingCutoffs, 10));
    setPrecisionCutoff((current) => keepOrNearest(current, precisionMetricCutoffs, 20));
    setRecallCutoff((current) => keepOrNearest(current, recallCutoffs, 50));
  }, [cutoffSignature, metricGroups, precisionMetricCutoffs.join(','), rankingCutoffs.join(','), recallCutoffs.join(',')]);
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
    const activeRoute = queryRoute || ROUTE_ORDER.find((route) => query.route_metrics?.[route]) || '';
    if (order === 'route' && activeRoute) {
      const rankOf = (candidate: RetrievalLlmCandidate) => (
        candidate.ranking_routes?.[activeRoute]?.rank
        ?? candidate.route_ranks?.[activeRoute]?.rank
        ?? Number.MAX_SAFE_INTEGER
      );
      return [...query.candidates].sort((left, right) => {
        const rankDelta = rankOf(left) - rankOf(right);
        return rankDelta || left.relevance_order - right.relevance_order;
      });
    }
    if (order === 'relevance') return query.candidates;
    return [...query.candidates].sort((left, right) => {
      if (left.best_route_rank !== right.best_route_rank) return left.best_route_rank - right.best_route_rank;
      return left.relevance_order - right.relevance_order;
    });
  }, [order, query, queryRoute]);

  const queryRoutes = useMemo(
    () => ROUTE_ORDER.filter((route) => query?.route_metrics?.[route]),
    [query]
  );
  const selectedQueryRoute = queryRoute && queryRoutes.includes(queryRoute)
    ? queryRoute
    : queryRoutes[0] || '';
  const queryCutoffs = query?.route_metrics?.[selectedQueryRoute]?.available_cutoffs || [];
  const selectedQueryCutoff = queryCutoffs.includes(queryCutoff)
    ? queryCutoff
    : queryCutoffs[queryCutoffs.length - 1] || 1;
  const selectedQueryRouteMetric = query?.route_metrics?.[selectedQueryRoute]?.by_cutoff?.[String(selectedQueryCutoff)] || null;

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: 0 });
  }, [itemId, reportSection]);

  const selectReportSection = (next: 'overview' | 'global' | 'labels') => {
    setReportSection(next);
    localStorage.setItem('pf-retrieval-report-section', next);
  };

  if (reportSection !== 'global' && (!poolId || !pools.length)) {
    return <div className="evaluation-empty"><strong>还没有冻结的检索候选池</strong><p>先生成冻结候选池，再完成机器相关性标注。</p></div>;
  }

  const availableAxes = Object.keys(selectedLabelSet?.axes || { score: {} });
  const recallScope = selectedMetrics?.recall_scope || workspace?.pool.recall_scope || 'six_route_candidate_union';
  const eligibleParentCount = workspace?.pool.counts?.eligible_parents_per_query;
  const recallScopeLabel = recallScope === 'eligible_author_corpus_before_cutoff'
    ? `时间切分前全部${eligibleParentCount ? ` ${eligibleParentCount} 篇` : ''}作者语料`
    : '冻结六路候选并集';
  const selectedRanking = rankingSnapshots.find((item) => item.ranking_id === rankingId) || null;
  const selectedRankingDepth = commonRankingDepth(selectedRanking);
  const stabilityProgress = selectedLabelSet?.progress;
  const qrels = selectedMetrics ? {
    labeled: selectedMetrics.qrels_label_count ?? selectedMetrics.judged_candidate_count ?? 0,
    unlabelled: selectedMetrics.qrels_unlabelled_count ?? 0,
    zero: selectedMetrics.qrels_zero_count ?? 0,
    useful: selectedMetrics.qrels_useful_count ?? selectedMetrics.relevant_candidate_count ?? 0,
    strong: selectedMetrics.qrels_strong_count ?? 0,
  } : null;
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
              <button type="button" className={reportSection === 'global' ? 'active' : ''} onClick={() => selectReportSection('global')}><Globe2 size={17} />所有作者</button>
              <button type="button" className={reportSection === 'labels' ? 'active' : ''} onClick={() => selectReportSection('labels')}><ListChecks size={17} />逐题标注</button>
            </nav>
          {reportSection === 'global' ? <div className="retrieval-global-sidebar-note">
            <strong>跨作者总览</strong>
            <span>按作者等权平均，不受左侧当前作者选择影响。</span>
            <small>{globalReport
              ? `已纳入 ${globalReport.included_authors} / ${globalReport.total_authors} 位作者`
              : loading
                ? '正在读取作者报告…'
                : error
                  ? `读取失败：${error}`
                  : '暂无可用报告'}</small>
            {globalReport && Object.keys(globalReport.axes).length > 1 ? <label>评估维度<select value={axis} onChange={(event) => setAxis(event.target.value)}>
              {Object.keys(globalReport.axes).map((value) => <option key={value} value={value}>{globalReport.axes[value]?.label || AXIS_LABELS[value] || value}</option>)}
            </select></label> : null}
          </div> : <div className="evaluation-dataset-controls">
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
            {rankingSnapshots.length ? <label>检索排名快照<select value={rankingId} onChange={(event) => setRankingId(event.target.value)}>
              {rankingSnapshots.map((item: RetrievalRankingSummary) => <option key={item.ranking_id} value={item.ranking_id} disabled={item.status !== 'completed'}>
                {item.ranking_id} · {item.status === 'completed' ? `请求 Top ${item.requested_depth} · 实际 ${commonRankingDepth(item)}` : item.status}
              </option>)}
            </select></label> : <div className="retrieval-overview-note">还没有独立排名快照<br /><small>完成 Parent Top100 快照后，才能显示正式 Recall@50/100。</small></div>}
            {labelSets.length ? <label>标注版本<select value={labelSet} onChange={(event) => setLabelSet(event.target.value)}>
              {labelSets.map((item) => <option key={item.label_set} value={item.label_set}>{item.label_set} · {item.provisional ? '离线代理' : item.model || '未知来源'} · {item.completed}/{item.total}</option>)}
            </select></label> : <div className="retrieval-overview-note">当前候选池暂无 LLM 标注<br /><small>候选池仍保留在这里，方便切换到其他已有报告。</small></div>}
            {labelSets.length && availableAxes.length > 1 ? <label>评估维度<select value={axis} onChange={(event) => setAxis(event.target.value)}>
              {availableAxes.map((value) => <option key={value} value={value}>{selectedLabelSet?.axes?.[value]?.label || AXIS_LABELS[value] || value}</option>)}
            </select></label> : null}
          </div>}
          {reportSection === 'labels' && labelSets.length ? (
            <nav className="evaluation-query-list" aria-label="LLM 检索评估问题">
              {visibleQueries.map((row) => (
                <button key={row.item_id} className={row.item_id === itemId ? 'active' : ''} type="button" onClick={() => setItemId(row.item_id)}>
                  <span className="query-ordinal">{row.ordinal}</span>
                  <span className="query-list-copy"><strong>{row.query}</strong><small>已标 {row.labeled_count} / {row.candidate_count}{row.qrels ? ` · 0/1/2：${row.qrels.zero_count}/${Math.max(row.qrels.useful_count - row.qrels.strong_count, 0)}/${row.qrels.strong_count}` : ''}</small></span>
                </button>
              ))}
            </nav>
          ) : reportSection === 'labels' ? <div className="retrieval-overview-note">当前候选池暂无逐题标注。</div> : reportSection === 'global' ? <div className="retrieval-overview-note">跨作者页面展示已完成同口径报告的宏平均；单作者诊断请切回“指标总览”。</div> : <div className="retrieval-overview-note">总览展示当前数据划分内所有问题的平均检索指标，不随单题变化。</div>}
        </aside>
      ) : null}
      <main className="evaluation-reader retrieval-llm-reader">
        {collapsed ? <button className="evaluation-rail-reveal" type="button" title="展开评估栏" onClick={() => { setCollapsed(false); localStorage.setItem('pf-retrieval-llm-sidebar', 'false'); }}><PanelLeftOpen size={18} /><span>RAG 评估</span></button> : null}
        <header className="evaluation-question-header retrieval-llm-header">
          <span>{reportSection === 'global' ? '检索评估 · 跨作者总览' : reportSection === 'overview' ? '检索评估 · 聚合结果' : labelSets.length ? `${splitLabel(split)} · ${visibleQueries.findIndex((row) => row.item_id === itemId) + 1 || '-'} / ${visibleQueries.length || 0}` : '逐题标注'}</span>
          <h1>{reportSection === 'global' ? '所有作者 RAG 指标总览' : !labelSets.length ? '当前候选池暂无 LLM 报告' : reportSection === 'overview' ? '多路检索指标总览' : query?.query || '正在读取问题'}</h1>
          <div className="retrieval-llm-meta">
             {reportSection === 'global' ? `${splitLabel(split)} · ${globalReport?.active_axis ? (AXIS_LABELS[globalReport.active_axis] || globalReport.active_axis) : '检索维度'} · 作者宏平均` : !labelSets.length ? '请从左侧切换到已有标注的候选池，或先创建 LLM 检索标注任务。' : <>{reportSection === 'overview' ? `${splitLabel(split)}的逐题指标平均值` : `标注者：${selectedLabelSet?.model || '未知'}`} · 稳定完成 {selectedLabelSet?.completed}/{selectedLabelSet?.total}
             {selectedLabelSet?.provisional ? <span className="retrieval-provisional-note"> · 离线代理标签，仅用于工程验证</span> : null}
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
          {reportSection !== 'global' && !labelSets.length ? <div className="evaluation-empty retrieval-report-unavailable">
            <strong>这个候选池已经建立，但还没有机器标注</strong>
            <p>候选池本身不是报告。请从左侧切换到带有“已有标注”的候选池，或者在“评估任务”中先创建标注。</p>
          </div> : reportSection === 'overview' || reportSection === 'global' ? <>
            <div className="retrieval-metric-controls">
              <div className="retrieval-segment" role="tablist" aria-label="数据划分">
                {availableSplits.map((value) => <button type="button" className={split === value ? 'active' : ''} key={value} onClick={() => setSplit(value)}>{splitLabel(value)}</button>)}
              </div>
              <label>排序质量 K
                <select value={ndcgCutoff} onChange={(event) => setNdcgCutoff(Number(event.target.value))}>
                  {rankingCutoffs.map((value) => <option key={value} value={value}>Top {value}</option>)}
                </select>
              </label>
              <label>精确率 K
                <select value={precisionCutoff} onChange={(event) => setPrecisionCutoff(Number(event.target.value))}>
                  {precisionMetricCutoffs.map((value) => <option key={value} value={value}>Top {value}</option>)}
                </select>
              </label>
              <label>召回率 K
                <select value={recallCutoff} onChange={(event) => setRecallCutoff(Number(event.target.value))}>
                  {recallCutoffs.map((value) => <option key={value} value={value}>Top {value}</option>)}
                </select>
              </label>
                  <span>{reportSection === 'global' ? `已纳入 ${globalReport?.included_authors || 0} 位作者 · 作者等权平均` : `${qrels?.useful ?? 0} 个有用 query-parent 对 · Recall 分母：${recallScopeLabel}`}</span>
            </div>
             <div className="retrieval-aggregate-explanation">{reportSection === 'global' ? <><strong>作者宏平均：</strong>每位作者先独立计算各路线指标，再让作者等权参与平均；材料更多的作者不会自动占更大权重。当前只汇总已有兼容完整标注的作者，K 选择只展示所有纳入作者和可用路线共同支持的深度。跨作者页面显示汇总数量和微平均；每道题的 Recall 分布请进入“逐题标注”查看。</> : <>下面各组数字是 <strong>{splitLabel(split)}</strong> 的逐题平均结果。指数 nDCG 使用 0/1/2 的指数增益，线性 nDCG 直接使用 0/1/2 作为增益；Useful 表示得分至少为 1，Strong 表示得分为 2。{rankingId ? `当前使用 ${workspace?.ranking?.ranking_id || rankingId} 的独立 Parent 排名快照，请求深度 ${workspace?.ranking?.requested_depth || 100}，实际可用深度 ${selectedRankingDepth}；Recall 分母来自冻结 Qrels。` : ''}</>}</div>
            {qrels ? <div className="retrieval-qrels-summary" aria-label="Qrels 标注分布">
              <div><span>已标注</span><strong>{countValue(qrels.labeled)}</strong></div>
              <div className="qrels-zero"><span>0 分 无用</span><strong>{countValue(qrels.zero)}</strong></div>
              <div className="qrels-one"><span>1 分 有帮助</span><strong>{countValue(Math.max(qrels.useful - qrels.strong, 0))}</strong></div>
              <div className="qrels-two"><span>2 分 明显有用</span><strong>{countValue(qrels.strong)}</strong></div>
              <div className="qrels-unknown"><span>未标注</span><strong>{countValue(qrels.unlabelled)}</strong></div>
              <small>Useful = 1+2；未标注不按 0 分计入指标。跨作者页面的数量是纳入作者合计，指标主口径仍是作者宏平均。</small>
            </div> : null}
            {reportSection === 'overview' && workspace?.comparison ? <div className="retrieval-comparison-summary">
              <strong>Gold-aware 相比旧 Query-only Judge 改判 {workspace.comparison.changed_count} / {workspace.comparison.total}</strong>
              <span>其中旧版 0 分、Gold-aware 改为 1/2 分：{workspace.comparison.v1_zero_to_v2_positive} 对</span>
            </div> : null}
            <div className="retrieval-llm-metrics">
              {ROUTE_ORDER.filter((route) => currentMetrics[route]).map((route) => {
                const base = currentMetrics[route];
                const ndcgMetric = base.by_cutoff?.[String(ndcgCutoff)] || base;
                const precisionMetric = base.by_cutoff?.[String(precisionCutoff)] || base;
                const recallMetric = base.by_cutoff?.[String(recallCutoff)] || base;
                const baselineRoute = RERANK_BASELINES[route];
                const baseline = baselineRoute ? currentMetrics[baselineRoute] : null;
                const baselineNdcg = baseline?.by_cutoff?.[String(ndcgCutoff)] || baseline;
                const baselinePrecision = baseline?.by_cutoff?.[String(precisionCutoff)] || baseline;
                const ndcgDelta = metricDelta(ndcgMetric.ndcg_at_k, baselineNdcg?.ndcg_at_k);
                const precisionDelta = metricDelta(precisionMetric.useful_precision_at_k, baselinePrecision?.useful_precision_at_k);
                return (
                <div className={`retrieval-llm-metric ${baselineRoute ? 'reranked' : ''}`} key={route}>
                  <strong>{ROUTE_LABELS[route] || route}</strong>
                  {baselineRoute ? <small className="retrieval-rerank-baseline">对比 {ROUTE_LABELS[baselineRoute] || baselineRoute}</small> : null}
                   <span title="使用 0/1/2 的指数增益，2 分材料的增益为 3">指数 nDCG@{ndcgCutoff} <b>{metricValue(ndcgMetric.ndcg_at_k)}</b>{ndcgDelta ? <em className={ndcgDelta.startsWith('-') ? 'negative' : 'positive'}>{ndcgDelta}</em> : null}</span>
                   <span title="直接使用 0/1/2 作为增益，不额外放大 2 分材料">线性 nDCG@{ndcgCutoff} <b>{metricValue(ndcgMetric.linear_ndcg_at_k ?? null)}</b></span>
                  <span title="前 K 篇中得分至少为 1 的材料比例">Useful Precision@{precisionCutoff} <b>{metricValue(precisionMetric.useful_precision_at_k ?? null)}</b>{precisionDelta ? <em className={precisionDelta.startsWith('-') ? 'negative' : 'positive'}>{precisionDelta}</em> : null}</span>
                  <span title="前 K 篇中得分为 2 的材料比例">Strong Precision@{precisionCutoff} <b>{metricValue(precisionMetric.strong_precision_at_k ?? null)}</b></span>
                  <span title="Top K 覆盖候选池内全部 1/2 分材料的比例">Useful Recall@{recallCutoff} <b>{metricValue(recallMetric.useful_recall_at_k ?? null)}</b></span>
                  <span title="把所有题目的 Useful 命中数和 Useful Qrels 数分别相加后计算">Useful Recall@{recallCutoff} · 材料微平均 <b>{metricValue(recallMetric.useful_recall_micro_at_k ?? null)}</b></span>
                  <span title="Top K 覆盖候选池内全部 2 分材料的比例">Strong Recall@{recallCutoff} <b>{metricValue(recallMetric.strong_recall_at_k ?? null)}</b></span>
                  <span title="把所有题目的 Strong 命中数和 Strong Qrels 数分别相加后计算">Strong Recall@{recallCutoff} · 材料微平均 <b>{metricValue(recallMetric.strong_recall_micro_at_k ?? null)}</b></span>
                  {reportSection !== 'global' ? <>
                    <small className="retrieval-recall-diagnostic">{recallDistribution(recallMetric, 'useful')}</small>
                    <small className="retrieval-recall-diagnostic">{recallDistribution(recallMetric, 'strong')}</small>
                  </> : null}
                </div>
              );})}
            </div>
            {reportSection === 'global' ? <div className="retrieval-global-authors">
              <div className="retrieval-global-authors-heading">
                <div><strong>作者纳入情况</strong><span>每行代表一位作者；上方指标卡是这些作者的等权平均。</span></div>
                <b>{globalReport?.included_authors || 0} / {globalReport?.total_authors || 0}</b>
              </div>
              <div className="retrieval-global-author-list">
                {(globalReport?.authors || []).map((author) => <div className="retrieval-global-author-row" key={author.author}>
                  <strong>{author.author}</strong>
                  <span>{author.effective_split === 'all' ? '全部数据' : author.effective_split.toUpperCase()}</span>
                  <span>{author.query_count} 题</span>
                  <small>{author.label_sets.join(' + ')}</small>
                </div>)}
                {(globalReport?.skipped_authors || []).map((author) => <div className="retrieval-global-author-row skipped" key={`skipped-${author.author}`}>
                  <strong>{author.author}</strong><span>未纳入</span><small>{author.reason}</small>
                </div>)}
              </div>
            </div> : null}
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
            {query?.qrels ? <div className="retrieval-query-qrels" aria-label="当前问题的 Qrels 分布">
              <strong>这道题的标注范围</strong>
              <span>已标注 {query.qrels.labeled_count} / {query.qrels.candidate_count}</span>
              <span className="qrels-zero">0 分 {query.qrels.zero_count}</span>
              <span className="qrels-one">1 分 {Math.max(query.qrels.useful_count - query.qrels.strong_count, 0)}</span>
              <span className="qrels-two">2 分 {query.qrels.strong_count}</span>
              <span className="qrels-unknown">未标注 {query.qrels.unlabelled_count}</span>
              <small>Recall 分母：Useful {query.qrels.useful_recall_denominator} · Strong {query.qrels.strong_recall_denominator}</small>
            </div> : null}
            {queryRoutes.length ? <div className="retrieval-query-route-report">
               <div className="retrieval-query-route-heading"><strong>路线对比</strong><span>指标只在当前题目内计算；遇到未标注材料会显示为未知，不会当作 0。</span></div>
              <div className="retrieval-query-route-table">
                {queryRoutes.map((route) => {
                  const routeSummary = query?.route_metrics?.[route];
                  const metric = routeSummary?.by_cutoff?.[String(selectedQueryCutoff)] || routeSummary?.by_cutoff?.[String(routeSummary.available_cutoffs?.[routeSummary.available_cutoffs.length - 1])];
                  return <button type="button" key={route} className={selectedQueryRoute === route ? 'active' : ''} onClick={() => { setQueryRoute(route); setOrder('route'); }}>
                    <strong>{ROUTE_LABELS[route] || route}</strong>
                    <span>nDCG <b>{metricValue(metric?.ndcg_at_k ?? null)}</b></span>
                    <span>Useful R <b>{metricValue(metric?.useful_recall_at_k ?? null)}</b></span>
                    <span>微平均 <b>{metricValue(metric?.useful_recall_micro_at_k ?? null)}</b></span>
                  </button>;
                })}
              </div>
            </div> : null}
            <div className="retrieval-llm-toolbar">
              <strong>{query?.candidate_count || 0} 个候选材料</strong>
               <span>相关性顺序按 Judge 分数展示；检索顺序用于核对各路线实际排名。</span>
              {selectedQueryRoute ? <label>查看路线<select value={selectedQueryRoute} onChange={(event) => { setQueryRoute(event.target.value); setOrder('route'); }}>
                {queryRoutes.map((route) => <option key={route} value={route}>{ROUTE_LABELS[route] || route}</option>)}
              </select></label> : null}
              {queryCutoffs.length ? <label>路线 K<select value={selectedQueryCutoff} onChange={(event) => { setQueryCutoff(Number(event.target.value)); setOrder('route'); }}>
                {queryCutoffs.map((value) => <option key={value} value={value}>Top {value}</option>)}
              </select></label> : null}
              <div className="retrieval-order-switch"><button className={order === 'relevance' ? 'active' : ''} type="button" onClick={() => setOrder('relevance')}>相关性顺序</button><button className={order === 'retrieval' ? 'active' : ''} type="button" onClick={() => setOrder('retrieval')}>最佳检索顺序</button><button className={order === 'route' ? 'active' : ''} type="button" onClick={() => setOrder('route')}>当前路线</button></div>
            </div>
             {selectedQueryRouteMetric ? <div className="retrieval-selected-query-metric"><strong>{ROUTE_LABELS[selectedQueryRoute] || selectedQueryRoute} · Top {selectedQueryCutoff}</strong><span>指数 nDCG {metricValue(selectedQueryRouteMetric.ndcg_at_k ?? null)}</span><span>线性 nDCG {metricValue(selectedQueryRouteMetric.linear_ndcg_at_k ?? null)}</span><span>Useful Precision {metricValue(selectedQueryRouteMetric.useful_precision_at_k ?? null)}</span><span>Useful Recall（题目平均） {metricValue(selectedQueryRouteMetric.useful_recall_at_k ?? null)}</span><span>Useful Recall（材料微平均） {metricValue(selectedQueryRouteMetric.useful_recall_micro_at_k ?? null)}</span><span>严格计入 {selectedQueryRouteMetric.strict_metric_query_count ?? 0} 题</span><small>{recallDistribution(selectedQueryRouteMetric, 'useful')}</small><small>{recallDistribution(selectedQueryRouteMetric, 'strong')}</small></div> : null}
            <div className="retrieval-llm-list">
              {orderedCandidates.map((candidate, index) => {
                const actualRouteRanks = Object.keys(candidate.ranking_routes || {}).length
                  ? candidate.ranking_routes
                  : candidate.route_ranks;
                const activeRouteDetails = selectedQueryRoute ? candidate.ranking_routes?.[selectedQueryRoute] : null;
                return (
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
                {activeRouteDetails?.reranked ? <div className="retrieval-rerank-evidence">
                  <div><strong>Reranker 审计</strong><span>原排名 #{activeRouteDetails.base_rank ?? '—'} → 新排名 #{activeRouteDetails.rank}</span><span>Cross-encoder {typeof activeRouteDetails.rerank_score === 'number' ? activeRouteDetails.rerank_score.toFixed(4) : '—'}</span></div>
                  <small>{activeRouteDetails.evidence?.node_type || 'passage'} · 第 {(activeRouteDetails.evidence?.index ?? 0) + 1} 段 · {activeRouteDetails.input_tokens ?? '—'} tokens{activeRouteDetails.input_truncated ? ' · 已截断' : ''}</small>
                  {activeRouteDetails.evidence?.text ? <p>{activeRouteDetails.evidence.text}</p> : null}
                </div> : null}
                <div className="retrieval-route-tags">{ROUTE_ORDER.filter((route) => actualRouteRanks?.[route]).map((route) => <span key={route}>{ROUTE_LABELS[route] || route} · #{actualRouteRanks?.[route].rank}</span>)}</div>
              </details>
            </article>
                );
              })}
            </div>
          </>}
        </div>
        {loading ? <div className="retrieval-llm-loading"><LoaderCircle className="spin" size={18} />正在读取</div> : null}
        {error ? <div className="evaluation-error">{error}</div> : null}
      </main>
    </section>
  );
}
