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
  writer_prompt: 'current' | 'strong_identity' | 'persona_pack';
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
