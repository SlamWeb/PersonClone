export type AuthUser = {
  id: string;
  username: string;
  display_name: string;
  role: 'admin' | 'member';
};

export type AuthState = {
  configured: boolean;
  authenticated: boolean;
  user?: AuthUser | null;
};

export type AdminUser = AuthUser & {
  created_at: string;
};

export type UserMemory = {
  id: string;
  kind: 'semantic' | 'episodic' | 'procedural';
  memory_key: string;
  content: string;
  status: string;
  pinned: boolean;
  sensitivity: 'normal' | 'private' | 'restricted';
  importance: number;
  confidence: number;
  event_status: 'ongoing' | 'historical' | 'stable';
  source_author?: string | null;
  source_conversation_id?: string | null;
  updated_at: string;
};

export type UserMemorySettings = {
  enabled: boolean;
  auto_write: boolean;
};

export type PersonaInfo = {
  author: string;
  source: string;
  index_dir: string;
  display_name: string;
  avatar_url?: string | null;
  headline: string;
  content_count?: number | null;
  persona_pack_available: boolean;
  narrative_schema_available?: boolean;
  profile_url?: string | null;
  last_synced_at?: string | null;
};

export type AuthorPreview = {
  author: string;
  display_name: string;
  avatar_url?: string | null;
  headline: string;
  profile_url: string;
  exists: boolean;
  ready: boolean;
};

export type AuthorJob = {
  id: string;
  source: string;
  author_input: string;
  author: string;
  operation: 'create' | 'sync';
  status: 'queued' | 'running' | 'ready' | 'failed' | 'cancelled' | 'interrupted';
  stage: string;
  label: string;
  kinds: string[];
  max_items?: number | null;
  display_name: string;
  avatar_url?: string | null;
  headline: string;
  profile_url: string;
  item_count?: number | null;
  parent_count?: number | null;
  node_count?: number | null;
  error_message?: string | null;
  cancel_requested: boolean;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  completed_at?: string | null;
};

export type SourceHit = {
  rank: number;
  score: number;
  node_id: string;
  node_type: string;
  route: string;
};

export type Source = {
  rank: number;
  parent_id: string;
  score: number;
  title: string;
  path: string;
  url?: string | null;
  first_hits: SourceHit[];
};

export type ChatStreamRequest = {
  author: string;
  session_id?: string | null;
  query: string;
  query_mode: 'raw' | 'grounded';
  writer_prompt: 'current' | 'strong_identity' | 'persona_pack' | 'mrprompt';
  parent_top_k: number;
  trace_capture: 'summary' | 'full';
};

export type ChatMessage = {
  id?: string | null;
  role: 'user' | 'assistant' | 'error';
  text: string;
  status?: 'queued' | 'running' | 'completed' | 'failed' | 'interrupted';
  sources?: Source[] | null;
  trace_id?: string | null;
  turn_id?: string | null;
};

export type ChatSessionSummary = {
  id: string;
  author: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
};

export type ChatSession = {
  id: string;
  author: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: ChatMessage[];
};

export type ChatCallbacks = {
  onAccepted?: (payload: {
    session_id: string;
    turn_id: string;
    status: string;
    stage: string;
    label: string;
  }) => void;
  onMeta?: (payload: Record<string, unknown>) => void;
  onStatus?: (payload: { stage: string; label: string }) => void;
  onToken?: (text: string) => void;
  onDone?: (payload: {
    session_id: string;
    turn_id?: string;
    trace_id?: string;
    answer: string;
    sources: Source[];
  }) => void;
  onError?: (message: string) => void;
};

export type TurnRun = {
  id: string;
  conversation_id: string;
  author: string;
  query: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'interrupted';
  stage: string;
  label: string;
  partial_answer: string;
  error?: { message?: string } | null;
  planner?: Record<string, unknown> | null;
  response_depth?: string | null;
  trace_id?: string | null;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  completed_at?: string | null;
};

export type TraceChildHit = {
  rank: number;
  score: number;
  node_id: string;
  parent_id: string;
  node_type: string;
  title: string;
  path: string;
  route: string;
};

export type TraceParent = {
  rank: number;
  score: number;
  parent_id: string;
  title: string;
  path: string;
  url?: string | null;
  first_hits: TraceChildHit[];
};

export type TracePayload = {
  schema_version?: string;
  trace_id: string;
  status: 'prepared' | 'completed' | 'failed';
  created_at: string;
  updated_at: string;
  capture?: { mode: 'summary' | 'full'; retention: number };
  stages?: TraceStage[];
  conversation_context?: {
    summary_version?: number;
    summary_through_sequence?: number;
    recent_turn_ids?: string[];
    relevant_turn_ids?: string[];
    selected_turn_ids?: string[];
    history_matches?: Array<{ turn_id: string; score: number }>;
    used_full_short_history?: boolean;
  } | null;
  turn_planner?: {
    turn_type?: 'new_topic' | 'follow_up' | 'explain_previous' | 'casual' | 'unclear';
    resolved_question?: string;
    retrieval_policy?: 'new' | 'reuse' | 'none';
    evidence_source_turn_id?: string | null;
    needs_web?: boolean;
    search_queries?: string[];
    response_depth?: 'brief' | 'normal' | 'deep';
    clarification_focus?: string;
    memory_ids?: string[];
  } | null;
  user_memory_recall?: {
    candidate_ids?: string[];
    selected_ids?: string[];
  } | null;
  memory_update?: {
    status?: 'completed' | 'skipped' | 'failed';
    duration_ms?: number;
    conversation_summary?: {
      status?: 'completed' | 'skipped' | 'failed';
      summary_version?: number | null;
      through_sequence?: number | null;
    };
    user_memory?: {
      status?: 'completed' | 'skipped' | 'failed';
      operations?: Array<{ operation?: string; memory_id?: string; memory_key?: string }>;
      rejections?: Array<{ reason?: string }>;
    };
  } | null;
  input: {
    author: string;
    session_id: string;
    query: string;
    query_mode: string;
    writer_prompt: string;
    retrieval_parameters: Record<string, number>;
  };
  query_understanding: {
    duration_ms: number;
    trace: {
      search_plan?: { needs_web?: boolean; search_queries?: string[] };
      search_results?: Array<{ query: string; title: string; url: string }>;
      retrieval_queries?: Array<{ route: string; query: string }>;
    } | null;
    objective_background: string;
  } | null;
  retrieval: {
    duration_ms: number;
    timing?: Record<string, number>;
    collection_name: string;
    retrieval_queries: Array<{ route: string; query: string }>;
    routes: Record<string, TraceChildHit[]>;
    parents: TraceParent[];
  } | null;
  writer: {
    variant: string;
    persona_pack_id?: string | null;
    persona_pack_sha256?: string | null;
    persona_pack_claim_count?: number;
    duration_ms: number;
    context_parents: Array<{ rank: number; parent_id: string; title: string }>;
    messages: Array<{ role: string; characters: number }>;
    total_characters: number;
  } | null;
  generation: {
    provider: string;
    model: string;
    temperature: number;
    max_tokens: number;
    duration_ms: number;
    time_to_first_token_ms?: number | null;
    usage?: TraceUsage | null;
    answer_characters: number;
  } | null;
  timing?: { total_duration_ms: number };
  error?: { type: string; message: string };
};

export type TraceUsage = {
  source: 'provider' | 'estimated';
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
  prompt_cache_hit_tokens?: number | null;
  prompt_cache_miss_tokens?: number | null;
  estimated_tokens?: number;
  characters?: number;
  note?: string;
};

export type TraceStage = {
  id: string;
  label: string;
  status: 'completed' | 'fallback' | 'failed' | 'running';
  order: number;
  started_offset_ms: number;
  duration_ms: number;
  details?: Record<string, unknown>;
  usage?: TraceUsage | null;
};

export type RetrievalPoolSummary = {
  pool_id: string;
  dataset_id: string;
  display_name?: string;
  author: string;
  author_status?: 'assigned' | 'unassigned';
  split: string;
  query_count: number;
  candidate_count: number;
  labeled_count: number;
  completed: boolean;
  created_at: string;
  llm_label_sets?: Array<{
    label_set: string;
    status: string;
    completed: number;
    total: number;
  }>;
  recall_scope?: string;
  counts?: {
    queries?: number;
    candidate_pairs?: number;
    eligible_parents_per_query?: number;
    unique_parents?: number;
  };
};

export type RetrievalQuerySummary = {
  item_id: string;
  ordinal: number;
  query: string;
  split?: string;
  candidate_count: number;
  labeled_count: number;
  completed: boolean;
};

export type RetrievalWorkspace = {
  pool: Pick<RetrievalPoolSummary, 'pool_id' | 'dataset_id' | 'display_name' | 'author' | 'split' | 'created_at' | 'recall_scope'>;
  progress: { labeled: number; total: number; completed: boolean };
  queries: RetrievalQuerySummary[];
};

export type RetrievalCandidate = {
  parent_id: string;
  ordinal: number;
  title: string;
  text: string;
  url: string;
  kind: string;
  score: 0 | 1 | 2 | null;
  retrieval_details?: Record<string, { rank: number; score: number }> | null;
};

export type RetrievalQuery = {
  pool_id: string;
  item_id: string;
  query: string;
  candidate_count: number;
  labeled_count: number;
  candidates: RetrievalCandidate[];
};

export type RetrievalRouteMetrics = {
  route: string;
  cutoff: number;
  query_count: number;
  candidate_count: number;
  judged_candidate_count: number;
  judged_query_count: number;
  fully_judged_query_count?: number;
  unjudged_query_count: number;
  coverage: number;
  hit_at_k: number | null;
  mrr_at_k: number | null;
  ndcg_at_k: number | null;
  precision_at_k: number | null;
  recall_at_k: number | null;
  map_at_k?: number | null;
  relevant_query_count?: number;
  no_relevant_query_count?: number;
  by_cutoff?: Record<string, Omit<RetrievalRouteMetrics, 'by_cutoff'>>;
};

export type RetrievalMetrics = {
  schema_version?: string;
  cutoff: number;
  cutoffs?: number[];
  recall_scope?: string;
  relevance_threshold: number;
  query_count: number;
  candidate_count?: number;
  judged_candidate_count?: number;
  relevant_candidate_count?: number;
  coverage?: number;
  routes: Record<string, RetrievalRouteMetrics>;
  splits?: Record<string, RetrievalMetrics>;
};

export type RetrievalLlmLabelSet = {
  label_set: string;
  status: string;
  model: string;
  prompt_version: string;
  completed: number;
  total: number;
  progress?: {
    pass1_completed?: number;
    judge_pass1_completed?: number;
    missing_pass1?: number;
    pass2_required?: number;
    pass2_completed?: number;
    pending_pass2?: number;
    pass3_required?: number;
    pass3_completed?: number;
    pending_pass3?: number;
    stability_completed?: number;
  };
  updated_at?: string;
  axes?: Record<string, { label?: string; values?: number[] }>;
  default_axis?: string;
  selected_splits?: string[];
  metrics: RetrievalMetrics;
};

export type RetrievalLlmWorkspace = {
  pool: Pick<RetrievalPoolSummary, 'pool_id' | 'dataset_id' | 'display_name' | 'author' | 'split' | 'created_at' | 'recall_scope' | 'counts'>;
  label_set: Omit<RetrievalLlmLabelSet, 'metrics' | 'updated_at'>;
  active_axis?: string;
  metrics: RetrievalMetrics;
  comparison?: {
    v1_label_set: string;
    v2_label_set: string;
    comparison_axis: string;
    total: number;
    changed_count: number;
    v1_zero_to_v2_positive: number;
    transition_counts: Record<string, Record<string, number>>;
  } | null;
  queries: RetrievalQuerySummary[];
};

export type RetrievalLlmCandidate = {
  parent_id: string;
  relevance_order: number;
  best_route_rank: number;
  route_count: number;
  title: string;
  text: string;
  url: string;
  kind: string;
  score: 0 | 1 | 2 | null;
  axis_scores?: Record<string, 0 | 1 | 2>;
  confidence?: string | null;
  evidence?: string;
  reason?: string;
  content_candidate_evidence?: string;
  content_gold_unit_ids?: string[];
  persona_candidate_evidence?: string;
  persona_gold_unit_ids?: string[];
  repeat_count?: number;
  exact_agreement?: boolean | null;
  status: string;
  route_ranks: Record<string, { rank: number; score: number }>;
};

export type RetrievalLlmQuery = {
  pool_id: string;
  label_set: string;
  axes?: Record<string, { label?: string; values?: number[] }>;
  active_axis?: string;
  item_id: string;
  query: string;
  gold_answer?: string;
  gold_units?: Record<string, Array<{ id?: string; text?: string }>>;
  candidate_count: number;
  labeled_count: number;
  candidates: RetrievalLlmCandidate[];
};

export type RetrievalEvalJob = {
  id: string;
  author: string;
  owner_id: string;
  labeler: 'deepseek_api' | 'codex_handoff' | 'manual_import';
  split: 'dev' | 'test';
  status: 'queued' | 'running' | 'awaiting_codex' | 'paused_budget' | 'completed' | 'failed' | 'interrupted';
  stage: string;
  label: string;
  dataset_id: string;
  label_set?: string | null;
  budget_cny: number;
  completed_items: number;
  total_items: number;
  estimated_cost_cny: number;
  usage: {
    request_count?: number;
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
    prompt_cache_hit_tokens?: number;
    prompt_cache_miss_tokens?: number;
    cache_hit_rate?: number | null;
    estimated_cost_cny?: number;
  };
  handoff_ready: boolean;
  error_message?: string | null;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  completed_at?: string | null;
};

export type GenerationSystem = {
  system_id: string;
  run_name: string;
  method_id?: string;
  display_name: string;
  description: string;
  parent_method?: string | null;
  author: string;
  author_status?: 'assigned' | 'unassigned';
  dataset_id: string;
  dataset_sha256: string;
  split: string;
  item_count: number;
  writer_prompt: string;
  prompt_version?: string | null;
  prompt_sha256?: string;
  parameters?: Record<string, unknown>;
  git_revision?: string;
  model: string;
  created_at: string;
  human_progress?: { completed: number; total: number };
  judge?: GenerationJudgeJob | null;
};

export type GenerationDimension = {
  key: string;
  short: string;
  label: string;
  question: string;
  anchors: Record<string, string>;
};

export type GenerationItemSummary = {
  item_id: string;
  ordinal: number;
  question: string;
  completed: boolean;
};

export type GenerationJudgeSummary = {
  item_count: number;
  dimensions: Record<string, {
    count: number;
    mean: number | null;
    median: number | null;
    ci95: [number | null, number | null];
    exact_agreement: number | null;
    within_one_agreement: number | null;
    mean_range: number | null;
  }>;
  groups: Record<string, number | null>;
};

export type GenerationJudgeJob = {
  id: string;
  system_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  stage: string;
  label: string;
  model: string;
  repeats: number;
  completed_items: number;
  total_items: number;
  prompt_version?: string | null;
  error_message?: string | null;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  result?: GenerationJudgeSummary | null;
};

export type GenerationWorkspace = {
  system: GenerationSystem;
  rubric: GenerationDimension[];
  groups: Record<string, string>;
  progress: { completed: number; total: number };
  items: GenerationItemSummary[];
  judge?: GenerationJudgeJob | null;
};

export type GenerationItem = {
  system: GenerationSystem;
  item_id: string;
  question: string;
  gold_answer: string;
  candidate_answer: string;
  human_scores: Record<string, number>;
  human_note: string;
  human_completed: boolean;
  judge?: {
    dimensions: Record<string, {
      score: number | null;
      status: string;
      gold_evidence: string[];
      candidate_evidence: string[];
      reason: string;
      exact_agreement?: number | null;
      within_one_agreement?: number | null;
      range?: number | null;
      raw_ratings?: Array<{
        repeat: number;
        score: number | null;
        status?: string;
        reason?: string;
        gold_evidence?: string[];
        candidate_evidence?: string[];
      }>;
    }>;
    groups: Record<string, number | null>;
  } | null;
};

export type GenerationComparison = {
  comparison_id: string;
  systems: GenerationSystem[];
  progress: { completed: number; total: number };
  items: GenerationItemSummary[];
  result: { votes: number; wins: Array<{ system: GenerationSystem; count: number }> };
};

export type GenerationComparisonItem = {
  comparison_id: string;
  item_id: string;
  question: string;
  gold_answer: string;
  candidate_a: string;
  candidate_b: string;
  choice: 'A' | 'B' | null;
  revealed: { A: GenerationSystem; B: GenerationSystem } | null;
};

export async function fetchAuthState(): Promise<AuthState> {
  const response = await apiFetch('/api/auth/state');
  if (!response.ok) throw new Error(await apiError(response, '无法读取登录状态'));
  return response.json();
}

export async function bootstrapAuth(request: {
  username: string;
  password: string;
  display_name?: string;
}): Promise<AuthState> {
  const response = await apiFetch('/api/auth/bootstrap', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  });
  if (!response.ok) throw new Error(await apiError(response, '无法创建管理员'));
  return response.json();
}

export async function loginAuth(username: string, password: string): Promise<AuthState> {
  const response = await apiFetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password })
  });
  if (!response.ok) throw new Error(await apiError(response, '无法登录'));
  return response.json();
}

export async function logoutAuth(): Promise<void> {
  const response = await apiFetch('/api/auth/logout', { method: 'POST' });
  if (!response.ok) throw new Error(await apiError(response, '无法退出登录'));
}

export async function fetchAdminUsers(): Promise<AdminUser[]> {
  const response = await apiFetch('/api/admin/users');
  if (!response.ok) throw new Error(await apiError(response, '无法读取成员'));
  const payload = await response.json();
  return payload.users || [];
}

export async function createAdminUser(request: {
  username: string;
  password: string;
  display_name?: string;
  role: 'admin' | 'member';
}): Promise<AdminUser> {
  const response = await apiFetch('/api/admin/users', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  });
  if (!response.ok) throw new Error(await apiError(response, '无法创建成员'));
  return response.json();
}

export async function fetchMemories(): Promise<UserMemory[]> {
  const response = await apiFetch('/api/memories');
  if (!response.ok) throw new Error(await apiError(response, '无法读取记忆'));
  const payload = await response.json();
  return payload.memories || [];
}

export async function fetchMemorySettings(): Promise<UserMemorySettings> {
  const response = await apiFetch('/api/memory-settings');
  if (!response.ok) throw new Error(await apiError(response, '无法读取记忆设置'));
  return response.json();
}

export async function updateMemorySettings(
  patch: Partial<UserMemorySettings>
): Promise<UserMemorySettings> {
  const response = await apiFetch('/api/memory-settings', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch)
  });
  if (!response.ok) throw new Error(await apiError(response, '无法更新记忆设置'));
  return response.json();
}

export async function updateMemory(
  memoryId: string,
  patch: { content?: string; pinned?: boolean }
): Promise<UserMemory> {
  const response = await apiFetch(`/api/memories/${encodeURIComponent(memoryId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch)
  });
  if (!response.ok) throw new Error(await apiError(response, '无法更新记忆'));
  return response.json();
}

export async function forgetMemory(memoryId: string): Promise<void> {
  const response = await apiFetch(`/api/memories/${encodeURIComponent(memoryId)}`, {
    method: 'DELETE'
  });
  if (!response.ok) throw new Error(await apiError(response, '无法遗忘记忆'));
}

export async function clearMemories(): Promise<void> {
  const response = await apiFetch('/api/memories', { method: 'DELETE' });
  if (!response.ok) throw new Error(await apiError(response, '无法清空记忆'));
}

export async function fetchPersonas(): Promise<{ personas: PersonaInfo[]; default_author?: string }> {
  const response = await apiFetch('/api/personas');
  if (!response.ok) {
    throw new Error(`Failed to load personas: ${response.status}`);
  }
  return response.json();
}

export async function fetchRetrievalPools(author?: string | null): Promise<RetrievalPoolSummary[]> {
  const query = author ? `?author=${encodeURIComponent(author)}` : '';
  const response = await apiFetch(`/api/evaluations/retrieval/pools${query}`);
  if (!response.ok) throw new Error(await apiError(response, '无法读取检索评估集'));
  const payload = await response.json();
  return payload.pools || [];
}

export async function fetchRetrievalWorkspace(poolId: string): Promise<RetrievalWorkspace> {
  const response = await apiFetch(`/api/evaluations/retrieval/pools/${encodeURIComponent(poolId)}`);
  if (!response.ok) throw new Error(await apiError(response, '无法读取评估进度'));
  return response.json();
}

export async function fetchRetrievalQuery(poolId: string, itemId: string): Promise<RetrievalQuery> {
  const response = await apiFetch(
    `/api/evaluations/retrieval/pools/${encodeURIComponent(poolId)}/queries/${encodeURIComponent(itemId)}`
  );
  if (!response.ok) throw new Error(await apiError(response, '无法读取候选材料'));
  return response.json();
}

export async function fetchRetrievalLlmLabelSets(poolId: string): Promise<RetrievalLlmLabelSet[]> {
  const response = await apiFetch(
    `/api/evaluations/retrieval/pools/${encodeURIComponent(poolId)}/llm-labels`
  );
  if (!response.ok) throw new Error(await apiError(response, '无法读取 LLM 检索标注'));
  const payload = await response.json();
  return payload.label_sets || [];
}

export async function fetchRetrievalLlmWorkspace(
  poolId: string,
  labelSet: string,
  axis?: string
): Promise<RetrievalLlmWorkspace> {
  const query = axis ? `?axis=${encodeURIComponent(axis)}` : '';
  const response = await apiFetch(
    `/api/evaluations/retrieval/pools/${encodeURIComponent(poolId)}/llm-labels/${encodeURIComponent(labelSet)}${query}`
  );
  if (!response.ok) throw new Error(await apiError(response, '无法读取 LLM 检索报告'));
  return response.json();
}

export async function fetchRetrievalLlmQuery(
  poolId: string,
  labelSet: string,
  itemId: string,
  axis?: string
): Promise<RetrievalLlmQuery> {
  const query = axis ? `?axis=${encodeURIComponent(axis)}` : '';
  const response = await apiFetch(
    `/api/evaluations/retrieval/pools/${encodeURIComponent(poolId)}/llm-labels/${encodeURIComponent(labelSet)}` +
      `/queries/${encodeURIComponent(itemId)}${query}`
  );
  if (!response.ok) throw new Error(await apiError(response, '无法读取 LLM 检索材料'));
  return response.json();
}

export async function fetchRetrievalEvalJobs(): Promise<RetrievalEvalJob[]> {
  const response = await apiFetch('/api/evaluations/retrieval/jobs');
  if (!response.ok) throw new Error(await apiError(response, '无法读取检索评估任务'));
  const payload = await response.json();
  return payload.jobs || [];
}

export async function createRetrievalEvalJob(request: {
  author: string;
  labeler: RetrievalEvalJob['labeler'];
  split: RetrievalEvalJob['split'];
  budget_cny: number;
}): Promise<RetrievalEvalJob> {
  const response = await apiFetch('/api/evaluations/retrieval/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  });
  if (!response.ok) throw new Error(await apiError(response, '无法创建检索评估任务'));
  return response.json();
}

export async function resumeRetrievalEvalJob(jobId: string, budgetCny: number): Promise<RetrievalEvalJob> {
  const response = await apiFetch(`/api/evaluations/retrieval/jobs/${encodeURIComponent(jobId)}/resume`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ budget_cny: budgetCny })
  });
  if (!response.ok) throw new Error(await apiError(response, '无法继续检索评估任务'));
  return response.json();
}

export async function downloadRetrievalEvalHandoff(jobId: string): Promise<void> {
  const response = await apiFetch(`/api/evaluations/retrieval/jobs/${encodeURIComponent(jobId)}/handoff`);
  if (!response.ok) throw new Error(await apiError(response, '无法下载 Codex handoff'));
  const blob = await response.blob();
  const disposition = response.headers.get('content-disposition') || '';
  const matchedName = disposition.match(/filename="?([^";]+)"?/i)?.[1];
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = matchedName || `${jobId}-codex-handoff.zip`;
  anchor.click();
  URL.revokeObjectURL(url);
}

export async function importRetrievalEvalReview(
  jobId: string,
  review: Record<string, unknown>
): Promise<RetrievalEvalJob> {
  const response = await apiFetch(`/api/evaluations/retrieval/jobs/${encodeURIComponent(jobId)}/codex-review`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(review)
  });
  if (!response.ok) throw new Error(await apiError(response, '无法导入双轴标注'));
  return response.json();
}

export async function saveRetrievalLabel(
  poolId: string,
  itemId: string,
  parentId: string,
  score: 0 | 1 | 2
): Promise<{ retrieval_details: Record<string, { rank: number; score: number }> }> {
  const response = await apiFetch(
    `/api/evaluations/retrieval/pools/${encodeURIComponent(poolId)}/queries/${encodeURIComponent(itemId)}` +
      `/candidates/${encodeURIComponent(parentId)}`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ score })
    }
  );
  if (!response.ok) throw new Error(await apiError(response, '无法保存评分'));
  return response.json();
}

export function retrievalExportUrl(poolId: string, format: 'jsonl' | 'csv'): string {
  return `/api/evaluations/retrieval/pools/${encodeURIComponent(poolId)}/export?format=${format}`;
}

export async function fetchGenerationSystems(author?: string | null): Promise<GenerationSystem[]> {
  const query = author ? `?author=${encodeURIComponent(author)}` : '';
  const response = await apiFetch(`/api/evaluations/generation/systems${query}`);
  if (!response.ok) throw new Error(await apiError(response, '无法读取生成系统'));
  const payload = await response.json();
  return payload.systems || [];
}

export async function fetchGenerationWorkspace(systemId: string): Promise<GenerationWorkspace> {
  const response = await apiFetch(`/api/evaluations/generation/systems/${encodeURIComponent(systemId)}`);
  if (!response.ok) throw new Error(await apiError(response, '无法读取生成评估'));
  return response.json();
}

export async function fetchGenerationItem(systemId: string, itemId: string): Promise<GenerationItem> {
  const response = await apiFetch(
    `/api/evaluations/generation/systems/${encodeURIComponent(systemId)}/items/${encodeURIComponent(itemId)}`
  );
  if (!response.ok) throw new Error(await apiError(response, '无法读取生成回答'));
  return response.json();
}

export async function saveGenerationRubric(
  systemId: string,
  itemId: string,
  scores: Record<string, number>,
  note = ''
): Promise<{ completed: boolean }> {
  const response = await apiFetch(
    `/api/evaluations/generation/systems/${encodeURIComponent(systemId)}/items/${encodeURIComponent(itemId)}/rubric`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scores, note })
    }
  );
  if (!response.ok) throw new Error(await apiError(response, '无法保存六维评分'));
  return response.json();
}

export async function fetchGenerationComparison(leftId: string, rightId: string): Promise<GenerationComparison> {
  const response = await apiFetch(
    `/api/evaluations/generation/comparisons/${encodeURIComponent(leftId)}/${encodeURIComponent(rightId)}`
  );
  if (!response.ok) throw new Error(await apiError(response, '无法读取 AB 对比'));
  return response.json();
}

export async function fetchGenerationComparisonItem(
  leftId: string,
  rightId: string,
  itemId: string
): Promise<GenerationComparisonItem> {
  const response = await apiFetch(
    `/api/evaluations/generation/comparisons/${encodeURIComponent(leftId)}/${encodeURIComponent(rightId)}` +
      `/items/${encodeURIComponent(itemId)}`
  );
  if (!response.ok) throw new Error(await apiError(response, '无法读取 AB 材料'));
  return response.json();
}

export async function saveGenerationPairwise(
  leftId: string,
  rightId: string,
  itemId: string,
  choice: 'A' | 'B'
): Promise<{ choice: 'A' | 'B'; revealed: { A: GenerationSystem; B: GenerationSystem } }> {
  const response = await apiFetch(
    `/api/evaluations/generation/comparisons/${encodeURIComponent(leftId)}/${encodeURIComponent(rightId)}` +
      `/items/${encodeURIComponent(itemId)}`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ choice })
    }
  );
  if (!response.ok) throw new Error(await apiError(response, '无法保存 AB 选择'));
  return response.json();
}

export async function createGenerationJudgeJob(systemId: string): Promise<GenerationJudgeJob> {
  const response = await apiFetch('/api/evaluations/generation/judge-jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ system_id: systemId, repeats: 3 })
  });
  if (!response.ok) throw new Error(await apiError(response, '无法创建 Judge 任务'));
  return response.json();
}

export async function fetchGenerationJudgeJob(jobId: string): Promise<GenerationJudgeJob> {
  const response = await apiFetch(`/api/evaluations/generation/judge-jobs/${encodeURIComponent(jobId)}`);
  if (!response.ok) throw new Error(await apiError(response, '无法读取 Judge 任务'));
  return response.json();
}

export async function previewAuthor(value: string): Promise<AuthorPreview> {
  const response = await apiFetch('/api/personas/preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ value })
  });
  if (!response.ok) {
    throw new Error(await apiError(response, '无法读取作者资料'));
  }
  return response.json();
}

export async function fetchAuthorJobs(): Promise<AuthorJob[]> {
  const response = await apiFetch('/api/author-jobs');
  if (!response.ok) {
    throw new Error(await apiError(response, '无法读取作者任务'));
  }
  const payload = await response.json();
  return payload.jobs || [];
}

export async function createAuthorJob(request: {
  author: string;
  kinds: Array<'answer' | 'article' | 'pin'>;
  max_items?: number | null;
}): Promise<AuthorJob> {
  const response = await apiFetch('/api/author-jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  });
  if (!response.ok) {
    throw new Error(await apiError(response, '无法创建作者任务'));
  }
  return response.json();
}

export async function cancelAuthorJob(jobId: string): Promise<AuthorJob> {
  const response = await apiFetch(`/api/author-jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
  if (!response.ok) {
    throw new Error(await apiError(response, '无法取消任务'));
  }
  return response.json();
}

export async function retryAuthorJob(jobId: string): Promise<AuthorJob> {
  const response = await apiFetch(`/api/author-jobs/${encodeURIComponent(jobId)}/retry`, { method: 'POST' });
  if (!response.ok) {
    throw new Error(await apiError(response, '无法重试任务'));
  }
  return response.json();
}

export async function fetchSessions(author: string): Promise<ChatSessionSummary[]> {
  const response = await apiFetch(`/api/personas/${encodeURIComponent(author)}/sessions`);
  if (!response.ok) {
    throw new Error(`Failed to load sessions: ${response.status}`);
  }
  const payload = await response.json();
  return payload.sessions || [];
}

export async function fetchSuggestions(author: string): Promise<string[]> {
  const response = await apiFetch(`/api/personas/${encodeURIComponent(author)}/suggestions`);
  if (!response.ok) {
    throw new Error(`Failed to load suggestions: ${response.status}`);
  }
  const payload = await response.json();
  return payload.suggestions || [];
}

export async function fetchSession(author: string, sessionId: string): Promise<ChatSession> {
  const response = await apiFetch(
    `/api/personas/${encodeURIComponent(author)}/sessions/${encodeURIComponent(sessionId)}`
  );
  if (!response.ok) {
    throw new Error(`Failed to load session: ${response.status}`);
  }
  return response.json();
}

export async function deleteSession(author: string, sessionId: string): Promise<void> {
  const response = await apiFetch(
    `/api/personas/${encodeURIComponent(author)}/sessions/${encodeURIComponent(sessionId)}`,
    { method: 'DELETE' }
  );
  if (!response.ok) {
    throw new Error(`Failed to delete session: ${response.status}`);
  }
}

export async function fetchTrace(author: string, traceId: string): Promise<TracePayload> {
  const response = await apiFetch(
    `/api/personas/${encodeURIComponent(author)}/traces/${encodeURIComponent(traceId)}`
  );
  if (!response.ok) {
    throw new Error(`Failed to load trace: ${response.status}`);
  }
  return response.json();
}

export async function fetchTurn(turnId: string): Promise<TurnRun> {
  const response = await apiFetch(`/api/chat/turns/${encodeURIComponent(turnId)}`);
  if (!response.ok) {
    throw new Error(await apiError(response, '无法读取生成任务'));
  }
  return response.json();
}

export async function retryTurn(turnId: string): Promise<TurnRun> {
  const response = await apiFetch(`/api/chat/turns/${encodeURIComponent(turnId)}/retry`, {
    method: 'POST'
  });
  if (!response.ok) {
    throw new Error(await apiError(response, '无法重试生成任务'));
  }
  return response.json();
}

export async function streamChat(request: ChatStreamRequest, callbacks: ChatCallbacks): Promise<void> {
  const response = await apiFetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  });
  if (!response.ok || !response.body) {
    throw new Error(await apiError(response, '无法创建回答任务'));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop() || '';
    for (const part of parts) {
      dispatchSse(part, callbacks);
    }
  }
  if (buffer.trim()) {
    dispatchSse(buffer, callbacks);
  }
}

function dispatchSse(raw: string, callbacks: ChatCallbacks): void {
  const lines = raw.split('\n');
  const event = lines
    .find((line) => line.startsWith('event:'))
    ?.replace(/^event:\s*/, '')
    .trim();
  const data = lines
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.replace(/^data:\s*/, ''))
    .join('\n');
  if (!event || !data) return;
  const payload = JSON.parse(data);
  if (event === 'accepted') callbacks.onAccepted?.(payload);
  if (event === 'meta') callbacks.onMeta?.(payload);
  if (event === 'status') callbacks.onStatus?.({ stage: String(payload.stage || ''), label: String(payload.label || '') });
  if (event === 'token') callbacks.onToken?.(String(payload.text || ''));
  if (event === 'done') callbacks.onDone?.(payload);
  if (event === 'error') callbacks.onError?.(String(payload.error || 'Unknown error'));
}

async function apiError(response: Response, fallback: string): Promise<string> {
  try {
    const payload = await response.json();
    return String(payload.detail || payload.error || `${fallback}（${response.status}）`);
  } catch {
    return `${fallback}（${response.status}）`;
  }
}

async function apiFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  const response = await fetch(input, { ...init, credentials: 'include' });
  const url = typeof input === 'string' ? input : input.toString();
  if (response.status === 401 && !url.includes('/api/auth/')) {
    window.dispatchEvent(new CustomEvent('pf-auth-expired'));
  }
  return response;
}
