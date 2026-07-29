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
  role: 'user' | 'assistant' | 'error';
  text: string;
  sources?: Source[] | null;
  trace_id?: string | null;
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
  onMeta?: (payload: Record<string, unknown>) => void;
  onStatus?: (payload: { stage: string; label: string }) => void;
  onToken?: (text: string) => void;
  onDone?: (payload: { session_id: string; trace_id?: string; answer: string; sources: Source[] }) => void;
  onError?: (message: string) => void;
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
  trace_id: string;
  status: 'prepared' | 'completed' | 'failed';
  created_at: string;
  updated_at: string;
  capture?: { mode: 'summary' | 'full'; retention: number };
  stages?: TraceStage[];
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

export async function fetchPersonas(): Promise<{ personas: PersonaInfo[]; default_author?: string }> {
  const response = await fetch('/api/personas');
  if (!response.ok) {
    throw new Error(`Failed to load personas: ${response.status}`);
  }
  return response.json();
}

export async function previewAuthor(value: string): Promise<AuthorPreview> {
  const response = await fetch('/api/personas/preview', {
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
  const response = await fetch('/api/author-jobs');
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
  const response = await fetch('/api/author-jobs', {
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
  const response = await fetch(`/api/author-jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
  if (!response.ok) {
    throw new Error(await apiError(response, '无法取消任务'));
  }
  return response.json();
}

export async function retryAuthorJob(jobId: string): Promise<AuthorJob> {
  const response = await fetch(`/api/author-jobs/${encodeURIComponent(jobId)}/retry`, { method: 'POST' });
  if (!response.ok) {
    throw new Error(await apiError(response, '无法重试任务'));
  }
  return response.json();
}

export async function fetchSessions(author: string): Promise<ChatSessionSummary[]> {
  const response = await fetch(`/api/personas/${encodeURIComponent(author)}/sessions`);
  if (!response.ok) {
    throw new Error(`Failed to load sessions: ${response.status}`);
  }
  const payload = await response.json();
  return payload.sessions || [];
}

export async function fetchSuggestions(author: string): Promise<string[]> {
  const response = await fetch(`/api/personas/${encodeURIComponent(author)}/suggestions`);
  if (!response.ok) {
    throw new Error(`Failed to load suggestions: ${response.status}`);
  }
  const payload = await response.json();
  return payload.suggestions || [];
}

export async function fetchSession(author: string, sessionId: string): Promise<ChatSession> {
  const response = await fetch(
    `/api/personas/${encodeURIComponent(author)}/sessions/${encodeURIComponent(sessionId)}`
  );
  if (!response.ok) {
    throw new Error(`Failed to load session: ${response.status}`);
  }
  return response.json();
}

export async function deleteSession(author: string, sessionId: string): Promise<void> {
  const response = await fetch(
    `/api/personas/${encodeURIComponent(author)}/sessions/${encodeURIComponent(sessionId)}`,
    { method: 'DELETE' }
  );
  if (!response.ok) {
    throw new Error(`Failed to delete session: ${response.status}`);
  }
}

export async function fetchTrace(author: string, traceId: string): Promise<TracePayload> {
  const response = await fetch(
    `/api/personas/${encodeURIComponent(author)}/traces/${encodeURIComponent(traceId)}`
  );
  if (!response.ok) {
    throw new Error(`Failed to load trace: ${response.status}`);
  }
  return response.json();
}

export async function streamChat(request: ChatStreamRequest, callbacks: ChatCallbacks): Promise<void> {
  const response = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  });
  if (!response.ok || !response.body) {
    throw new Error(`Chat request failed: ${response.status}`);
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
