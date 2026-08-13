import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowLeft, ArrowRight, Check, CheckCircle2, ClipboardCopy, Copy, Download,
  FlaskConical, House, LogIn, Plus, RotateCcw, Save, Users, X
} from 'lucide-react';

import type { AuthState } from './api';

type Highlight = {
  annotation_id: string;
  start: number;
  end: number;
  selected_text: string;
  impact: -2 | -1 | 0 | 1 | 2;
  reason: string;
};

type StudyMeta = {
  available: boolean;
  study_id?: string;
  title: string;
  author?: string;
  author_label?: string;
  avatar_url?: string | null;
  pointwise_count?: number;
  pairwise_count?: number;
  total?: number;
  participant_path?: string;
  protocol_version?: string;
  recruitable?: boolean;
};

type StudyCatalogEntry = {
  study_id: string;
  author: string;
  author_label: string;
  item_count: number;
  available: boolean;
  protocol_version?: string;
  recruitable?: boolean;
  error?: string | null;
  participant_path?: string;
};

type StudyState = {
  session_id: string;
  phase: 'pointwise' | 'transition' | 'pairwise' | 'exposure' | 'completed';
  progress: { completed: number; total: number };
  phase_progress?: { completed: number; total: number };
  can_previous?: boolean;
  final_trial?: boolean;
  author: string;
  author_label: string;
  trial?: {
    kind: 'pointwise' | 'pairwise';
    trial_id: string;
    question: string;
    answer?: string;
    left_answer?: string;
    right_answer?: string;
  };
  draft?: Record<string, any>;
  exploratory_feedback?: string;
  demo_turn_count?: number;
  demo_turn_limit?: number;
};

type StudySessionPointer = {
  session_id: string;
  resume_token: string;
};

type StudyStartResponse = StudyState & {
  resume_token: string;
};

type SelectionRange = {
  start: number;
  end: number;
  text: string;
  left: number;
  top: number;
  placement: 'above' | 'below';
};

async function studyFetch(path: string, init?: RequestInit) {
  const response = await fetch(path, { credentials: 'include', ...init });
  if (!response.ok) {
    let message = '请求失败';
    try {
      const payload = await response.json();
      message = payload.detail || payload.error || message;
    } catch {
      message = `${message} (${response.status})`;
    }
    throw new Error(message);
  }
  return response;
}

function studySessionFetch(path: string, resumeToken: string, init?: RequestInit) {
  const headers = new Headers(init?.headers);
  headers.set('X-Study-Session-Token', resumeToken);
  return studyFetch(path, { ...init, headers });
}

async function copyText(value: string) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }
  const input = document.createElement('textarea');
  input.value = value;
  input.setAttribute('readonly', '');
  input.style.position = 'fixed';
  input.style.opacity = '0';
  document.body.appendChild(input);
  input.select();
  const copied = document.execCommand('copy');
  input.remove();
  if (!copied) throw new Error('浏览器未允许复制，请手动选择参与码');
}

function readSessionPointer(value: string | null): StudySessionPointer | null {
  if (!value) return null;
  try {
    const parsed = JSON.parse(value) as Partial<StudySessionPointer>;
    if (typeof parsed.session_id === 'string' && typeof parsed.resume_token === 'string') {
      return { session_id: parsed.session_id, resume_token: parsed.resume_token };
    }
  } catch {
    // Legacy values stored only a session id. Re-entering the participant code
    // safely claims a fresh browser credential without losing the session.
  }
  return null;
}

export function StudyWorkspace({ authState }: { authState: AuthState | null }) {
  const admin = window.location.pathname === '/experiment/admin';
  if (admin) return <StudyAdmin authState={authState} />;
  const match = window.location.pathname.match(/^\/experiment\/([^/]+)\/?$/);
  const studyId = match ? decodeURIComponent(match[1]) : null;
  return <ParticipantStudy studyId={studyId} />;
}

function ParticipantStudy({ studyId }: { studyId: string | null }) {
  const [meta, setMeta] = useState<StudyMeta | null>(null);
  const [state, setState] = useState<StudyState | null>(null);
  const [resumeToken, setResumeToken] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const endpoint = studyId
      ? `/api/studies/study1/studies/${encodeURIComponent(studyId)}`
      : '/api/studies/study1';
    studyFetch(endpoint)
      .then((response) => response.json())
      .then((payload: StudyMeta) => {
        setMeta(payload);
        const storageKey = payload.study_id ? `pf-study1-session:${payload.study_id}` : '';
        const url = new URL(window.location.href);
        const forceFresh = url.searchParams.get('new') === '1';
        if (forceFresh && storageKey) {
          localStorage.removeItem(storageKey);
          url.searchParams.delete('new');
          window.history.replaceState({}, '', `${url.pathname}${url.search}${url.hash}`);
        }
        const saved = !forceFresh && storageKey ? readSessionPointer(localStorage.getItem(storageKey)) : null;
        if (!saved) {
          if (storageKey) localStorage.removeItem(storageKey);
          return null;
        }
        return studySessionFetch(`/api/studies/study1/sessions/${encodeURIComponent(saved.session_id)}`, saved.resume_token)
          .then((response) => response.json())
          .then((next) => { setResumeToken(saved.resume_token); setState(next); })
          .catch(() => localStorage.removeItem(`pf-study1-session:${payload.study_id}`));
      })
      .catch((reason) => setError(String((reason as Error).message || reason)))
      .finally(() => setLoading(false));
  }, [studyId]);

  async function start(profile: Record<string, unknown>) {
    setError('');
    setLoading(true);
    try {
      const endpoint = studyId
        ? `/api/studies/study1/studies/${encodeURIComponent(studyId)}/sessions`
        : '/api/studies/study1/sessions';
      const response = await studyFetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(profile)
      });
      const next = await response.json() as StudyStartResponse;
      if (!next.resume_token) throw new Error('未收到实验恢复凭据，请重试');
      setState(next);
      setResumeToken(next.resume_token);
      if (meta?.study_id) localStorage.setItem(
        `pf-study1-session:${meta.study_id}`,
        JSON.stringify({ session_id: next.session_id, resume_token: next.resume_token }),
      );
    } catch (reason) {
      setError(String((reason as Error).message || reason));
    } finally {
      setLoading(false);
    }
  }

  function useAnotherCode() {
    if (meta?.study_id) localStorage.removeItem(`pf-study1-session:${meta.study_id}`);
    setState(null);
    setResumeToken('');
    setError('');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  if (loading && !meta) return <StudyLoading />;
  if (!meta?.available) return <StudyUnavailable error={error} />;
  if (!state || !resumeToken) return <StudyIntro meta={meta} error={error} busy={loading} onStart={start} />;

  const progress = state.progress.completed / state.progress.total;
  return (
    <div className="study-page">
      <StudyHeader title={meta.title} progress={progress} label={`${state.progress.completed}/${state.progress.total}`} />
      {state.phase === 'completed' ? (
        <StudyComplete state={state} resumeToken={resumeToken} meta={meta} onUseAnotherCode={useAnotherCode} />
      ) : state.phase === 'exposure' ? (
        <FinalExposureCheck state={state} resumeToken={resumeToken} onAdvance={setState} />
      ) : state.phase === 'transition' ? (
        <PhaseTransition state={state} resumeToken={resumeToken} onAdvance={setState} />
      ) : state.phase === 'pointwise' && state.trial ? (
        <PointwiseTrial key={state.trial.trial_id} state={state} resumeToken={resumeToken} onAdvance={setState} />
      ) : state.trial ? (
        <PairwiseTrial key={state.trial.trial_id} state={state} resumeToken={resumeToken} onAdvance={setState} />
      ) : null}
    </div>
  );
}

function StudyHeader({ title, progress, label }: { title: string; progress: number; label: string }) {
  return (
    <header className="study-topbar">
      <div className="study-topbar-brand"><FlaskConical size={18} />{title}</div>
      <div className="study-progress"><div><span style={{ width: `${progress * 100}%` }} /></div><strong>{label}</strong></div>
    </header>
  );
}

function StudyIntro({ meta, error, busy, onStart }: {
  meta: StudyMeta;
  error: string;
  busy: boolean;
  onStart: (profile: Record<string, unknown>) => Promise<void>;
}) {
  const [profile, setProfile] = useState({
    participant_code: '', follow_duration: '1_to_3_years', reading_frequency: 'weekly',
    familiarity: 'familiar', ai_frequency: 'almost_daily', consent: false
  });
  function change(key: string, value: string | boolean) { setProfile((item) => ({ ...item, [key]: value })); }
  return (
    <div className="study-page">
      <StudyHeader title={meta.title} progress={0} label={`0/${meta.total || 4}`} />
      <main className="study-intro-layout">
        <section className="study-intro-copy">
          <p className="study-kicker">FORMATIVE STUDY</p>
          <h1>哪些文字，让一篇回答听起来像这个作者？</h1>
          <p>本研究邀请熟悉「{meta.author_label}」的读者，凭第一感觉判断匿名回答。全程约 20–30 分钟。</p>
          <div className="study-outline">
            <div><span>01</span><div><strong>2 篇独立判断</strong><p>判断整体作者感，并标出最影响你的文字。</p></div></div>
            <div><span>02</span><div><strong>2 组同题比较</strong><p>比较两篇匿名回答，选择相对更像的一篇。</p></div></div>
          </div>
        </section>
        <form className="study-entry-form" onSubmit={(event) => { event.preventDefault(); void onStart(profile); }}>
          <h2>开始实验</h2>
          <label>参与码<input required value={profile.participant_code} onChange={(e) => change('participant_code', e.target.value)} placeholder="PF-XXXX-XXXX" /></label>
          <label>关注作者时长<select value={profile.follow_duration} onChange={(e) => change('follow_duration', e.target.value)}><option value="less_than_3_months">不足 3 个月</option><option value="3_to_12_months">3–12 个月</option><option value="1_to_3_years">1–3 年</option><option value="more_than_3_years">3 年以上</option></select></label>
          <label>阅读作者内容的频率<select value={profile.reading_frequency} onChange={(e) => change('reading_frequency', e.target.value)}><option value="rarely">偶尔</option><option value="monthly">每月几次</option><option value="weekly">每周几次</option><option value="almost_daily">几乎每天</option></select></label>
          <label>对作者表达的熟悉程度<select value={profile.familiarity} onChange={(e) => change('familiarity', e.target.value)}><option value="somewhat">有些熟悉</option><option value="familiar">比较熟悉</option><option value="very_familiar">非常熟悉</option></select></label>
          <label>使用生成式 AI 的频率<select value={profile.ai_frequency} onChange={(e) => change('ai_frequency', e.target.value)}><option value="rarely">很少使用</option><option value="monthly">每月几次</option><option value="weekly">每周几次</option><option value="almost_daily">几乎每天</option></select></label>
          <label className="study-consent"><input type="checkbox" checked={profile.consent} onChange={(e) => change('consent', e.target.checked)} /><CheckCircle2 size={18} /><span>我已了解任务内容，同意匿名保存本次判断；如感到不适，可以随时退出。</span></label>
          {error ? <p className="study-error">{error}</p> : null}
          <button className="study-primary" disabled={busy || !profile.consent} type="submit">{busy ? '正在进入' : '进入实验'}<ArrowRight size={18} /></button>
        </form>
      </main>
    </div>
  );
}

function PhaseTransition({ state, resumeToken, onAdvance }: {
  state: StudyState;
  resumeToken: string;
  onAdvance: (next: StudyState) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  async function advance() {
    setBusy(true); setError('');
    try {
      const response = await studySessionFetch(`/api/studies/study1/sessions/${state.session_id}/transition`, resumeToken, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ acknowledge: true })
      });
      onAdvance(await response.json());
    } catch (reason) { setError(String((reason as Error).message || reason)); setBusy(false); }
  }
  async function previous() {
    setBusy(true); setError('');
    try {
      const response = await studySessionFetch(`/api/studies/study1/sessions/${state.session_id}/navigate`, resumeToken, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ direction: 'previous' })
      });
      onAdvance(await response.json());
    } catch (reason) { setError(String((reason as Error).message || reason)); setBusy(false); }
  }
  return <main className="study-phase-transition">
    <p className="study-kicker">第一部分已完成</p>
    <h1>接下来比较两篇回答</h1>
    <p>第二部分共有 2 组。每组回答同一个问题，请选择相对更像作者的一篇，并写下选择与不选择的最关键原因。</p>
    <div className="study-transition-steps"><span>选择 A 或 B</span><ArrowRight size={17} /><span>判断信心</span><ArrowRight size={17} /><span>填写两条关键理由</span></div>
    {error ? <p className="study-error">{error}</p> : null}
    <div className="study-exposure-actions">
      <button type="button" className="study-secondary" disabled={busy} onClick={() => void previous()}><ArrowLeft size={17} />上一题</button>
      <button type="button" className="study-primary" disabled={busy} onClick={() => void advance()}>{busy ? '正在进入' : '进入第二部分'}<ArrowRight size={17} /></button>
    </div>
  </main>;
}

function PointwiseTrial({ state, resumeToken, onAdvance }: { state: StudyState; resumeToken: string; onAdvance: (next: StudyState) => void }) {
  const trial = state.trial!;
  const initial = state.draft || {};
  const [overallScore, setOverallScore] = useState<number | null>(initial.overall_score ?? null);
  const [highlights, setHighlights] = useState<Highlight[]>(initial.highlights || []);
  const [primaryReason, setPrimaryReason] = useState(initial.primary_reason || '');
  const [saveState, setSaveState] = useState('修改会自动保存');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const started = useRef(Date.now());
  const baseElapsed = useRef(initial.elapsed_ms || 0);
  const initialized = useRef(false);
  const payload = (submit: boolean) => ({ overall_score: overallScore, highlights, primary_reason: primaryReason, elapsed_ms: baseElapsed.current + Date.now() - started.current, submit });

  useEffect(() => {
    if (!initialized.current) { initialized.current = true; return; }
    const timer = window.setTimeout(async () => {
      setSaveState('正在保存');
      try {
        await studySessionFetch(`/api/studies/study1/sessions/${state.session_id}/pointwise/${trial.trial_id}`, resumeToken, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload(false)) });
        setSaveState('草稿已保存');
      } catch { setSaveState('保存失败，将继续保留当前内容'); }
    }, 800);
    return () => window.clearTimeout(timer);
  }, [overallScore, highlights, primaryReason]);

  async function submit() {
    setSubmitting(true); setError('');
    try {
      const response = await studySessionFetch(`/api/studies/study1/sessions/${state.session_id}/pointwise/${trial.trial_id}`, resumeToken, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload(true)) });
      onAdvance(await response.json());
    } catch (reason) { setError(String((reason as Error).message || reason)); setSubmitting(false); }
  }
  async function previous() {
    setSubmitting(true); setError('');
    try {
      await studySessionFetch(`/api/studies/study1/sessions/${state.session_id}/pointwise/${trial.trial_id}`, resumeToken, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload(false)) });
      const response = await studySessionFetch(`/api/studies/study1/sessions/${state.session_id}/navigate`, resumeToken, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ direction: 'previous' }) });
      onAdvance(await response.json());
    } catch (reason) { setError(String((reason as Error).message || reason)); setSubmitting(false); }
  }
  const ready = overallScore !== null && highlights.length >= 1 && Boolean(primaryReason.trim()) && highlights.every((item) => item.reason.trim());
  const scoreOptions = [
    [-2, '明显不像'], [-1, '比较不像'], [0, '拿不准'], [1, '比较像'], [2, '明显像']
  ] as const;
  return (
    <main className="study-trial">
      <TrialHeading eyebrow={`第一部分 · 单篇 ${state.phase_progress!.completed + 1}/2`} question={trial.question} instruction="可以边读边标记让你觉得像或不像的文字；读完后，再给整篇回答评分并写下整体理由。" />
      <section className="study-answer-card"><div className="study-answer-label">回答正文</div><HighlightableText text={trial.answer!} highlights={highlights} onChange={setHighlights} maximumTotal={6} enabled /></section>
      <section className="study-judgment">
        <h2>这篇回答整体上有多像这位作者？</h2>
        <div className="study-score-grid">
          {scoreOptions.map(([value,label]) => <button type="button" key={value} className={overallScore === value ? 'selected' : ''} onClick={() => setOverallScore(value)}><strong>{value > 0 ? `+${value}` : value}</strong><span>{label}</span></button>)}
        </div>
        <p className="study-score-help">阅读过程中可划出 1～6 处具体证据。正负证据可以同时存在。</p>
        <label className="study-global-note">整篇最关键的判断理由<textarea required rows={3} value={primaryReason} onChange={(e) => setPrimaryReason(e.target.value)} placeholder="用一句话写下最影响整体判断的原因。" /></label>
      </section>
      <StudyFooter status={saveState} error={error} ready={ready} submitting={submitting} finalTrial={Boolean(state.final_trial)} canPrevious={Boolean(state.can_previous)} onPrevious={previous} onSubmit={submit} />
    </main>
  );
}

function PairwiseTrial({ state, resumeToken, onAdvance }: { state: StudyState; resumeToken: string; onAdvance: (next: StudyState) => void }) {
  const trial = state.trial!;
  const initial = state.draft || {};
  const [choice, setChoice] = useState<string | null>(initial.choice || null);
  const [confidence, setConfidence] = useState<string | null>(initial.confidence || null);
  const [selectedReason, setSelectedReason] = useState(initial.selected_reason || '');
  const [rejectedReason, setRejectedReason] = useState(initial.rejected_reason || '');
  const [saveState, setSaveState] = useState('修改会自动保存');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const started = useRef(Date.now());
  const baseElapsed = useRef(initial.elapsed_ms || 0);
  const initialized = useRef(false);
  const payload = (submit: boolean) => ({ choice, confidence, selected_reason: selectedReason, rejected_reason: rejectedReason, elapsed_ms: baseElapsed.current + Date.now() - started.current, submit });
  useEffect(() => {
    if (!initialized.current) { initialized.current = true; return; }
    const timer = window.setTimeout(async () => {
      setSaveState('正在保存');
      try { await studySessionFetch(`/api/studies/study1/sessions/${state.session_id}/pairwise/${trial.trial_id}`, resumeToken, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload(false)) }); setSaveState('草稿已保存'); }
      catch { setSaveState('保存失败，将继续保留当前内容'); }
    }, 800);
    return () => window.clearTimeout(timer);
  }, [choice, confidence, selectedReason, rejectedReason]);
  async function submit() {
    setSubmitting(true); setError('');
    try { const response = await studySessionFetch(`/api/studies/study1/sessions/${state.session_id}/pairwise/${trial.trial_id}`, resumeToken, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload(true)) }); onAdvance(await response.json()); }
    catch (reason) { setError(String((reason as Error).message || reason)); setSubmitting(false); }
  }
  async function previous() {
    setSubmitting(true); setError('');
    try {
      await studySessionFetch(`/api/studies/study1/sessions/${state.session_id}/pairwise/${trial.trial_id}`, resumeToken, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload(false)) });
      const response = await studySessionFetch(`/api/studies/study1/sessions/${state.session_id}/navigate`, resumeToken, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ direction: 'previous' }) });
      onAdvance(await response.json());
    } catch (reason) { setError(String((reason as Error).message || reason)); setSubmitting(false); }
  }
  const ready = Boolean(choice && confidence && selectedReason.trim() && rejectedReason.trim());
  return (
    <main className="study-trial study-pair-trial">
      <TrialHeading eyebrow={`第二部分 · 配对 ${state.phase_progress!.completed + 1}/2`} question={trial.question} instruction="比较同一问题下的两篇匿名回答，选择相对更像作者的一篇。" />
      <section className="study-pair-grid">
        <article className={choice === 'left' ? 'selected-answer' : ''}><div className="study-pair-label">回答 A</div><div className="study-answer-text">{trial.left_answer}</div></article>
        <article className={choice === 'right' ? 'selected-answer' : ''}><div className="study-pair-label">回答 B</div><div className="study-answer-text">{trial.right_answer}</div></article>
      </section>
      <section className="study-judgment pair">
        <h2>哪一篇整体上更像作者？</h2>
        <div className="study-choice-grid two"><button type="button" className={choice === 'left' ? 'selected' : ''} onClick={() => setChoice('left')}><strong>A 更像</strong></button><button type="button" className={choice === 'right' ? 'selected' : ''} onClick={() => setChoice('right')}><strong>B 更像</strong></button></div>
        <h3>你对这次选择有多确定？</h3>
        <div className="study-confidence">{[['close','两篇比较接近'],['fairly_sure','有一定把握'],['very_sure','非常确定']].map(([value,label]) => <button type="button" key={value} className={confidence === value ? 'selected' : ''} onClick={() => setConfidence(value)}>{label}</button>)}</div>
        <div className="study-pair-reasons">
          <label>你选择这篇最关键的理由是什么？<textarea rows={3} value={selectedReason} onChange={(e) => setSelectedReason(e.target.value)} placeholder="只写一个最关键因素，一句话即可。" /></label>
          <label>另一篇让你没有选择它的最关键原因是什么？<textarea rows={3} value={rejectedReason} onChange={(e) => setRejectedReason(e.target.value)} placeholder="只写一个最关键因素，一句话即可。" /></label>
        </div>
      </section>
      <StudyFooter status={saveState} error={error} ready={ready} submitting={submitting} finalTrial={Boolean(state.final_trial)} canPrevious={Boolean(state.can_previous)} onPrevious={previous} onSubmit={submit} />
    </main>
  );
}

function TrialHeading({ eyebrow, question, instruction }: { eyebrow: string; question: string; instruction: string }) {
  return <section className="study-trial-heading"><p className="study-kicker">{eyebrow}</p><h1>{question}</h1><p>{instruction}</p></section>;
}

function StudyFooter({ status, error, ready, submitting, finalTrial, canPrevious, onPrevious, onSubmit }: { status: string; error: string; ready: boolean; submitting: boolean; finalTrial: boolean; canPrevious: boolean; onPrevious: () => void; onSubmit: () => void }) {
  return <footer className="study-trial-footer"><span aria-live="polite"><Save size={14} />{status}</span>{error ? <p className="study-error">{error}</p> : null}{canPrevious ? <button type="button" className="study-previous" disabled={submitting} onClick={onPrevious}><ArrowLeft size={17} />上一题</button> : null}<button type="button" className="study-primary" title={ready ? '' : '请先完成整体判断，并提供至少一条判断依据'} disabled={!ready || submitting} onClick={onSubmit}>{submitting ? '正在保存' : finalTrial ? '提交' : '下一题'}<ArrowRight size={18} /></button></footer>;
}

function FinalExposureCheck({ state, resumeToken, onAdvance }: { state: StudyState; resumeToken: string; onAdvance: (next: StudyState) => void }) {
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [value, setValue] = useState('');
  const options = [['no','都没看过'],['yes','看过其中一些'],['unsure','不确定']];
  async function confirm() {
    if (!value) return;
    setBusy(true); setError('');
    try { const response = await studySessionFetch(`/api/studies/study1/sessions/${state.session_id}/exposure`, resumeToken, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ value }) }); onAdvance(await response.json()); }
    catch (reason) { setError(String((reason as Error).message || reason)); setBusy(false); }
  }
  async function previous() {
    setBusy(true); setError('');
    try { const response = await studySessionFetch(`/api/studies/study1/sessions/${state.session_id}/navigate`, resumeToken, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ direction: 'previous' }) }); onAdvance(await response.json()); }
    catch (reason) { setError(String((reason as Error).message || reason)); setBusy(false); }
  }
  return <main className="study-exposure"><p className="study-kicker">提交前确认</p><h1>在参加实验之前，你是否看过本次实验中的相同回答或基本相同的版本？</h1><p>这里只做一次整体确认。选择后提交，前面的四道判断将不能再修改。</p><div className="study-exposure-options">{options.map(([option,label]) => <button className={value === option ? 'selected' : ''} disabled={busy} type="button" key={option} onClick={() => setValue(option)}>{label}{value === option ? <Check size={17} /> : null}</button>)}</div>{error ? <p className="study-error">{error}</p> : null}<div className="study-exposure-actions"><button type="button" className="study-secondary" disabled={busy} onClick={() => void previous()}><ArrowLeft size={17} />上一题</button><button type="button" className="study-primary" disabled={busy || !value} onClick={() => void confirm()}>{busy ? '正在提交' : '确认提交'}<ArrowRight size={17} /></button></div></main>;
}

function StudyComplete({ state, resumeToken, meta, onUseAnotherCode }: {
  state: StudyState;
  resumeToken: string;
  meta: StudyMeta;
  onUseAnotherCode: () => void;
}) {
  const [feedback, setFeedback] = useState(state.exploratory_feedback || '');
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState('');
  const [demoInput, setDemoInput] = useState('');
  const [demoTurns, setDemoTurns] = useState<Array<{ query: string; answer: string }>>([]);
  const [demoCount, setDemoCount] = useState(state.demo_turn_count || 0);
  const [demoStatus, setDemoStatus] = useState('');
  async function save() {
    try { await studySessionFetch(`/api/studies/study1/sessions/${state.session_id}/feedback`, resumeToken, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: feedback }) }); setSaved(true); }
    catch (reason) { setError(String((reason as Error).message || reason)); }
  }
  async function demo(event: FormEvent) {
    event.preventDefault();
    const query = demoInput.trim();
    if (!query || demoStatus || demoCount >= (state.demo_turn_limit || 3)) return;
    setDemoInput(''); setError(''); setDemoStatus('正在理解问题');
    const index = demoTurns.length;
    setDemoTurns((items) => [...items, { query, answer: '' }]);
    try {
      await streamStudyDemo(state.session_id, resumeToken, query, {
        status: setDemoStatus,
        token: (text) => setDemoTurns((items) => items.map((item, itemIndex) => itemIndex === index ? { ...item, answer: item.answer + text } : item)),
      });
      setDemoCount((count) => count + 1);
      setDemoStatus('');
    } catch (reason) {
      setError(String((reason as Error).message || reason));
      setDemoTurns((items) => items.slice(0, -1));
      setDemoStatus('');
    }
  }
  const limit = state.demo_turn_limit || 3;
  return <main className="study-complete">
    {meta.avatar_url ? <img className="study-complete-avatar" src={meta.avatar_url} alt={meta.author_label || '作者头像'} /> : <CheckCircle2 size={38} />}
    <p className="study-kicker">正式任务已完成</p><h1>感谢参加实验</h1><p>你的判断和文字标注已经保存。</p>
    <div className="study-complete-actions"><a className="study-secondary" href="/"><House size={17} />返回 PersonaForge</a><button type="button" className="study-secondary" onClick={onUseAnotherCode}><RotateCcw size={17} />使用其他参与码</button></div>
    <section className="study-demo"><div className="study-demo-heading"><div><h2>可选：自由体验分身</h2><p>最多交流三轮。这部分不会改变前面的正式判断。</p></div><span>{demoCount}/{limit}</span></div>
      {demoTurns.length ? <div className="study-demo-messages">{demoTurns.map((turn,index) => <div key={`${turn.query}-${index}`}><p className="user">{turn.query}</p><p className="assistant">{turn.answer || demoStatus}</p></div>)}</div> : null}
      <form className="study-demo-composer" onSubmit={demo}><input value={demoInput} onChange={(e) => setDemoInput(e.target.value)} disabled={demoCount >= limit || Boolean(demoStatus)} placeholder={demoCount >= limit ? '自由体验已完成' : `问问${state.author_label}`} /><button type="submit" disabled={!demoInput.trim() || Boolean(demoStatus) || demoCount >= limit}><ArrowRight size={17} /></button></form>
      {demoStatus ? <span className="study-demo-status">{demoStatus}</span> : null}
    </section>
    <section><h2>可选体验反馈</h2><p>体验后，你觉得它哪里像、哪里仍然不像？</p><textarea rows={4} value={feedback} onChange={(e) => { setFeedback(e.target.value); setSaved(false); }} placeholder="写下最明显的一点即可。" /><button type="button" className="study-primary" onClick={() => void save()}>保存反馈</button>{saved ? <span className="study-saved">已保存</span> : null}{error ? <p className="study-error">{error}</p> : null}</section>
  </main>;
}

async function streamStudyDemo(
  sessionId: string,
  resumeToken: string,
  query: string,
  callbacks: { status: (label: string) => void; token: (text: string) => void }
) {
  const response = await fetch(`/api/studies/study1/sessions/${encodeURIComponent(sessionId)}/demo/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Study-Session-Token': resumeToken },
    body: JSON.stringify({ query }),
    credentials: 'include'
  });
  if (!response.ok || !response.body) {
    let message = '无法开始自由体验';
    try { message = (await response.json()).detail || message; } catch { /* ignore */ }
    throw new Error(message);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split('\n\n');
    buffer = blocks.pop() || '';
    for (const block of blocks) {
      const event = block.split('\n').find((line) => line.startsWith('event:'))?.slice(6).trim();
      const dataText = block.split('\n').filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trim()).join('\n');
      if (!dataText) continue;
      const data = JSON.parse(dataText);
      if (event === 'status') callbacks.status(data.label || '正在处理');
      if (event === 'token') callbacks.token(data.text || '');
      if (event === 'error') throw new Error(data.error || '自由体验生成失败');
    }
  }
}

function HighlightableText({ text, highlights, onChange, maximumTotal, enabled }: { text: string; highlights: Highlight[]; onChange: (items: Highlight[]) => void; maximumTotal: number; enabled: boolean }) {
  const container = useRef<HTMLDivElement>(null);
  const popover = useRef<HTMLDivElement>(null);
  const [selection, setSelection] = useState<SelectionRange | null>(null);
  const [managed, setManaged] = useState<(Highlight & { left: number; top: number; placement: 'above' | 'below' }) | null>(null);
  const [impact, setImpact] = useState<-2 | -1 | 0 | 1 | 2 | null>(null);
  const [reason, setReason] = useState('');
  const [error, setError] = useState('');
  const segments = useMemo(() => segmentText(text, highlights), [text, highlights]);
  useEffect(() => {
    if (!selection && !managed) return;
    function dismiss(event: PointerEvent) {
      if (popover.current?.contains(event.target as Node)) return;
      cancel();
    }
    function escape(event: KeyboardEvent) { if (event.key === 'Escape') cancel(); }
    const timer = window.setTimeout(() => document.addEventListener('pointerdown', dismiss), 0);
    document.addEventListener('keydown', escape);
    return () => { window.clearTimeout(timer); document.removeEventListener('pointerdown', dismiss); document.removeEventListener('keydown', escape); };
  }, [selection, managed]);
  function capture() {
    if (!enabled) { setError('请先完成整篇作者相似度评分，再开始划线。'); window.getSelection()?.removeAllRanges(); return; }
    const root = container.current;
    const selected = window.getSelection();
    if (!root || !selected || selected.rangeCount === 0 || selected.isCollapsed) return;
    const range = selected.getRangeAt(0);
    if (!root.contains(range.commonAncestorContainer)) return;
    const before = document.createRange(); before.selectNodeContents(root); before.setEnd(range.startContainer, range.startOffset);
    let start = before.toString().length; let value = range.toString();
    const leading = value.match(/^\s*/)?.[0].length || 0; const trailing = value.match(/\s*$/)?.[0].length || 0;
    start += leading; value = value.slice(leading, trailing ? value.length - trailing : undefined);
    if (!value) return;
    const end = start + value.length;
    if (highlights.length >= maximumTotal) { setError(`这篇材料最多保留 ${maximumTotal} 处关键文字。`); selected.removeAllRanges(); return; }
    if (highlights.some((item) => Math.max(item.start, start) < Math.min(item.end, end))) { setError('这段文字与已有标注重叠，请重新选择。'); selected.removeAllRanges(); return; }
    const rect = range.getBoundingClientRect(); const placement = rect.top > 190 ? 'above' : 'below';
    setError(''); setManaged(null); setImpact(null); setReason('');
    setSelection({ start, end, text: value, left: Math.min(Math.max(rect.left + rect.width / 2, 220), window.innerWidth - 220), top: placement === 'above' ? rect.top - 10 : rect.bottom + 10, placement });
  }
  function cancel() { window.getSelection()?.removeAllRanges(); setSelection(null); setManaged(null); setImpact(null); setReason(''); }
  function save(event: FormEvent) {
    event.preventDefault();
    if (impact === null || !reason.trim() || (!selection && !managed)) return;
    if (managed) {
      onChange(highlights.map((item) => item.annotation_id === managed.annotation_id ? { ...item, impact, reason: reason.trim() } : item));
    } else if (selection) {
      const annotationId = typeof crypto.randomUUID === 'function' ? crypto.randomUUID() : `annotation-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      onChange([...highlights, { annotation_id: annotationId, start: selection.start, end: selection.end, selected_text: selection.text, impact, reason: reason.trim() }]);
    }
    cancel();
  }
  function manage(event: React.MouseEvent<HTMLElement>, item: Highlight) {
    event.preventDefault(); event.stopPropagation(); window.getSelection()?.removeAllRanges();
    const rect = event.currentTarget.getBoundingClientRect(); const placement = rect.top > 190 ? 'above' : 'below';
    setSelection(null); setImpact(item.impact); setReason(item.reason); setManaged({ ...item, left: Math.min(Math.max(rect.left + rect.width / 2, 220), window.innerWidth - 220), top: placement === 'above' ? rect.top - 10 : rect.bottom + 10, placement });
  }
  function removeManaged() {
    if (!managed) return;
    onChange(highlights.filter((item) => item.annotation_id !== managed.annotation_id));
    cancel();
  }
  const impactOptions = [
    [-2, '明显不像'], [-1, '有点不像'], [0, '方向不明'], [1, '有点像'], [2, '明显像']
  ] as const;
  const impactClass = (value: number) => value > 0 ? 'impact-positive' : value < 0 ? 'impact-negative' : 'impact-neutral';
  const activeText = selection?.text || managed?.selected_text || '';
  return <div className="study-highlight-workspace">
    <div className="study-highlight-counts"><strong>{enabled ? `已标注 ${highlights.length}/${maximumTotal}` : '完成整体评分后可划线'}</strong><small>点击已有标记可以修改或删除</small></div>
    <div ref={container} className={`study-answer-text${enabled ? '' : ' annotation-disabled'}`} onMouseUp={capture} onTouchEnd={() => window.setTimeout(capture, 80)}>{segments.map((segment) => segment.highlight ? <mark key={segment.highlight.annotation_id} className={impactClass(segment.highlight.impact)} title="点击管理这处标注" onClick={(event) => manage(event, segment.highlight!)}>{segment.text}</mark> : <span key={`${segment.start}-${segment.end}`}>{segment.text}</span>)}</div>
    {selection || managed ? <div ref={popover} className={`study-selection-popover ${(selection || managed)!.placement}`} style={{ left: (selection || managed)!.left, top: (selection || managed)!.top }}>
      <form onSubmit={save}><p>“{activeText}”</p><strong>这处文字怎样影响你的判断？</strong><div className="study-impact-options">{impactOptions.map(([value,label]) => <button type="button" key={value} className={impact === value ? `selected ${impactClass(value)}` : ''} onClick={() => setImpact(value)}><b>{value > 0 ? `+${value}` : value}</b><span>{label}</span></button>)}</div><input autoFocus={impact !== null} value={reason} onChange={(e) => setReason(e.target.value)} placeholder="为什么？" maxLength={500} /><div>{managed ? <button type="button" className="danger" onClick={removeManaged}><X size={15} />删除</button> : <span />}<button type="submit" disabled={impact === null || !reason.trim()}><Check size={15} />保存</button></div></form>
    </div> : null}
    {error ? <p className="study-inline-error">{error}</p> : null}
    {highlights.length ? <div className="study-annotation-list">{highlights.map((item,index) => <div key={item.annotation_id} className={impactClass(item.impact)}><span>{item.impact > 0 ? `+${item.impact}` : item.impact}</span><div><p>“{item.selected_text}”</p><input aria-label="修改标注理由" value={item.reason} onChange={(e) => onChange(highlights.map((entry,i) => i === index ? { ...entry, reason: e.target.value } : entry))} /></div><button type="button" aria-label="删除这处标注" title="删除这处标注" onClick={() => onChange(highlights.filter((_,i) => i !== index))}><X size={15} /></button></div>)}</div> : null}
  </div>;
}

function segmentText(text: string, highlights: Highlight[]) {
  const sorted = [...highlights].sort((a,b) => a.start - b.start);
  const output: Array<{ start: number; end: number; text: string; highlight?: Highlight }> = [];
  let cursor = 0;
  for (const item of sorted) { if (cursor < item.start) output.push({ start: cursor, end: item.start, text: text.slice(cursor, item.start) }); output.push({ start: item.start, end: item.end, text: text.slice(item.start,item.end), highlight: item }); cursor = item.end; }
  if (cursor < text.length) output.push({ start: cursor, end: text.length, text: text.slice(cursor) });
  return output;
}

function StudyAdmin({ authState }: { authState: AuthState | null }) {
  const [studies, setStudies] = useState<StudyCatalogEntry[]>([]);
  const [studyId, setStudyId] = useState('');
  const [overview, setOverview] = useState<any>(null);
  const [detail, setDetail] = useState<any>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [copiedCode, setCopiedCode] = useState('');
  async function refresh(selectedStudyId = studyId) {
    if (!selectedStudyId) return;
    try {
      setError('');
      const response = await studyFetch(`/api/studies/study1/admin/overview?study_id=${encodeURIComponent(selectedStudyId)}`);
      setOverview(await response.json());
    } catch (reason) {
      setError(String((reason as Error).message || reason));
    }
  }
  useEffect(() => {
    if (authState?.user?.role !== 'admin') return;
    studyFetch('/api/studies/study1/admin/studies')
      .then((response) => response.json())
      .then((payload) => {
        const catalog = (payload.studies || []) as StudyCatalogEntry[];
        setStudies(catalog);
        const requested = new URLSearchParams(window.location.search).get('study');
        const requestedEntry = catalog.find((item) => item.study_id === requested && item.available);
        const initial = (requestedEntry?.recruitable ? requestedEntry : undefined)
          || catalog.find((item) => item.available && item.recruitable)
          || requestedEntry
          || catalog.find((item) => item.available);
        setStudyId(initial?.study_id || '');
      })
      .catch((reason) => setError(String((reason as Error).message || reason)));
  }, [authState?.user?.role]);
  useEffect(() => {
    if (!studyId) return;
    const url = new URL(window.location.href);
    url.searchParams.set('study', studyId);
    window.history.replaceState({}, '', url);
    setOverview(null);
    setDetail(null);
    void refresh(studyId);
  }, [studyId]);
  async function createCodes() {
    setError('');
    setNotice('');
    try {
      const response = await studyFetch('/api/studies/study1/admin/codes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ count: 10, study_id: studyId })
      });
      const payload = await response.json();
      if (!Array.isArray(payload.codes) || payload.codes.length === 0) throw new Error('没有生成新的参与码');
      await copyText(payload.codes.join('\n'));
      setNotice(`已生成 ${payload.codes.length} 个参与码，并复制到剪贴板`);
      await refresh();
    } catch (reason) {
      setError(String((reason as Error).message || reason));
    }
  }
  async function copyParticipantLink() {
    setError('');
    try {
      const path = `/experiment/${encodeURIComponent(studyId)}`;
      await copyText(`${window.location.origin}${path}`);
      setNotice('实验链接已复制');
    } catch (reason) { setError(String((reason as Error).message || reason)); }
  }
  async function copyCode(code: string) {
    setError('');
    try {
      await copyText(code);
      setCopiedCode(code);
      window.setTimeout(() => setCopiedCode((current) => current === code ? '' : current), 1400);
    } catch (reason) { setError(String((reason as Error).message || reason)); }
  }
  function codeStatusLabel(status: string) {
    return ({ available: '未使用', started: '进行中', completed: '已完成' } as Record<string, string>)[status] || status;
  }
  async function openSession(id: string) { const response = await studyFetch(`/api/studies/study1/admin/sessions/${id}`); setDetail(await response.json()); }
  if (authState === null) return <StudyLoading />;
  if (!authState.authenticated || authState.user?.role !== 'admin') return <main className="study-admin-gate"><LogIn size={28} /><h1>管理员登录后才能进入</h1><p>参与者无需产品账号，研究者后台需要管理员权限。</p><a href="/">返回登录</a></main>;
  const selectedStudy = studies.find((item) => item.study_id === studyId);
  const canRecruit = Boolean(selectedStudy?.recruitable);
  const exportBase = `/api/studies/study1/admin/export?study_id=${encodeURIComponent(studyId)}`;
  const analysisBundle = `/api/studies/study1/admin/analysis-bundle?study_id=${encodeURIComponent(studyId)}`;
  return <div className="study-admin">
    <header><div><p className="study-kicker">STUDY 1</p><h1>实验管理</h1></div><nav><a target="_blank" rel="noreferrer" href={studyId ? `/experiment/${encodeURIComponent(studyId)}?new=1` : '/experiment'}>打开新参与者入口</a><a href="/">返回产品</a></nav></header>
     <section className="study-admin-studybar">
       <label>实验作者<select value={studyId} onChange={(event) => setStudyId(event.target.value)}>{studies.map((study) => <option key={study.study_id} value={study.study_id} disabled={!study.available}>{study.author_label} · {study.study_id}{!study.available ? '（材料不可用）' : study.recruitable ? '' : '（旧协议，仅回放）'}</option>)}</select></label>
      {selectedStudy ? <div><strong>{selectedStudy.author_label}</strong><span>{selectedStudy.item_count} 道冻结问题 · {selectedStudy.study_id}</span></div> : <span>尚未发现可用材料库</span>}
    </section>
    {error ? <p className="study-error">{error}</p> : null}
    <section className="study-admin-actions"><button type="button" disabled={!canRecruit} onClick={() => void createCodes()}><Plus size={17} />生成 10 个参与码并复制</button><button type="button" disabled={!canRecruit} onClick={() => void copyParticipantLink()}><ClipboardCopy size={17} />复制实验链接</button><a className={!studyId ? 'disabled' : ''} href={studyId ? analysisBundle : undefined}><Download size={17} />分析包 ZIP</a><a className={!studyId ? 'disabled' : ''} href={studyId ? `${exportBase}&format=jsonl` : undefined}><Download size={17} />原始 JSONL</a></section>
    {notice ? <p className="study-success">{notice}</p> : null}
    {overview ? <div className="study-admin-grid"><section><h2>参与码 <span>{overview.codes.length}</span></h2><div className="study-code-list">{overview.codes.map((code: any) => <div key={code.code}><code>{code.code}</code><span>{codeStatusLabel(code.status)}</span><button type="button" aria-label={`复制参与码 ${code.code}`} title={copiedCode === code.code ? '已复制' : '复制参与码'} onClick={() => void copyCode(code.code)}>{copiedCode === code.code ? <Check size={14} /> : <Copy size={14} />}</button></div>)}</div></section><section><h2>参与进度 <span>{overview.sessions.length}</span></h2><div className="study-session-list">{overview.sessions.map((session: any) => <button type="button" key={session.session_id} onClick={() => void openSession(session.session_id)}><Users size={16} /><span><strong>{session.participant_code}</strong><small>{session.completed}/{session.total} · {session.status === 'completed' ? '已完成' : '进行中'}</small></span><ArrowRight size={16} /></button>)}</div></section></div> : studyId ? <p>正在读取实验数据</p> : <p>请先准备至少一份可用材料库</p>}
    {detail ? <AdminDetail detail={detail} onClose={() => setDetail(null)} /> : null}
  </div>;
}

function parseReplayHighlights(value: unknown): Highlight[] {
  let rows: any[] = [];
  if (Array.isArray(value)) rows = value;
  else if (typeof value !== 'string') return [];
  try {
    if (typeof value === 'string') {
      const parsed = JSON.parse(value);
      rows = Array.isArray(parsed) ? parsed : [];
    }
  } catch {
    return [];
  }
  return rows.map((item, index) => ({
    annotation_id: item.annotation_id || `legacy-${item.start}-${item.end}-${index}`,
    start: item.start,
    end: item.end,
    selected_text: item.selected_text,
    impact: typeof item.impact === 'number' ? item.impact : item.polarity === 'like' ? 1 : -1,
    reason: item.reason || item.note || '',
  })) as Highlight[];
}

function ReplayAnswer({ label, text, highlights, selected = false }: {
  label: string;
  text: string;
  highlights: Highlight[];
  selected?: boolean;
}) {
  const segments = segmentText(text, highlights);
  return <section className={`study-replay-answer${selected ? ' selected' : ''}`}>
    <header><strong>{label}</strong>{selected ? <span>参与者选择</span> : null}</header>
    <div className="study-replay-text">{segments.map((segment) => segment.highlight
      ? <mark key={`${segment.start}-${segment.end}`} className={segment.highlight.impact > 0 ? 'impact-positive' : segment.highlight.impact < 0 ? 'impact-negative' : 'impact-neutral'}>{segment.text}</mark>
      : <span key={`${segment.start}-${segment.end}`}>{segment.text}</span>)}</div>
    {highlights.length ? <div className="study-replay-evidence">{highlights.map((item) => <div key={item.annotation_id} className={item.impact > 0 ? 'impact-positive' : item.impact < 0 ? 'impact-negative' : 'impact-neutral'}>
      <strong>{item.impact > 0 ? `+${item.impact} 提高相似感` : item.impact < 0 ? `${item.impact} 降低相似感` : '0 方向不明确'}</strong><span>“{item.selected_text}”</span>{item.reason ? <p>{item.reason}</p> : null}
    </div>)}</div> : <p className="study-replay-empty">未划线标注</p>}
  </section>;
}

function replayVerdict(value: string | null | undefined) {
  return ({ like: '像作者', unsure: '拿不准', unlike: '不像作者' } as Record<string, string>)[value || ''] || '尚未判断';
}

function replayScore(value: number | null | undefined) {
  if (value === null || value === undefined) return '尚未评分';
  const label = ({ '-2': '明显不像', '-1': '比较不像', '0': '拿不准', '1': '比较像', '2': '明显像' } as Record<string,string>)[String(value)];
  return `${value > 0 ? '+' : ''}${value} · ${label}`;
}

function replayConfidence(value: string | null | undefined) {
  return ({ close: '两篇很接近', fairly_sure: '比较确定', very_sure: '非常确定' } as Record<string, string>)[value || ''] || '尚未填写';
}

function AdminDetail({ detail, onClose }: { detail: any; onClose: () => void }) {
  const pointwise = detail.pointwise || [];
  const pairwise = detail.pairwise || [];
  return <div className="study-admin-overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}><aside>
    <header><div><p className="study-kicker">参与者回放</p><h2>{detail.participant_code}</h2><small>既往接触：{detail.prior_exposure || '尚未提交'}</small></div><button type="button" aria-label="关闭参与者回放" onClick={onClose}><X size={18} /></button></header>
    <section className="study-replay-section"><h3>单篇判断 <span>{pointwise.length} 篇</span></h3>{pointwise.map((item: any, index: number) => {
      const response = item.response;
      const highlights = parseReplayHighlights(item.annotations?.length ? item.annotations : response?.highlights_json);
      return <article className="study-replay-trial" key={item.trial.trial_id}><div className="study-replay-trial-heading"><span>{index + 1}</span><div><p>问题</p><h4>{item.trial.question}</h4></div></div>
        {!response ? <p className="study-replay-unanswered">尚未作答</p> : <><div className="study-replay-summary"><div><span>整体判断</span><strong>{response.overall_score === null || response.overall_score === undefined ? replayVerdict(response.verdict) : replayScore(response.overall_score)}</strong></div><div><span>完成状态</span><strong>{response.status === 'submitted' ? '已提交' : '草稿'}</strong></div></div>
          <ReplayAnswer label="匿名回答" text={item.trial.answer || ''} highlights={highlights} />
          {response.primary_reason || response.global_note ? <div className="study-replay-note"><span>整篇最关键理由</span><p>{response.primary_reason || response.global_note}</p></div> : null}</>}
      </article>;
    })}</section>
    <section className="study-replay-section"><h3>配对判断 <span>{pairwise.length} 组</span></h3>{pairwise.map((item: any, index: number) => {
      const response = item.response;
      const leftHighlights = parseReplayHighlights(response?.left_highlights_json);
      const rightHighlights = parseReplayHighlights(response?.right_highlights_json);
      return <article className="study-replay-trial" key={item.trial.trial_id}><div className="study-replay-trial-heading"><span>{index + 1}</span><div><p>问题</p><h4>{item.trial.question}</h4></div></div>
        {!response ? <p className="study-replay-unanswered">尚未作答</p> : <><div className="study-replay-summary"><div><span>更像作者</span><strong className={response.choice || ''}>{response.choice === 'left' ? 'A' : response.choice === 'right' ? 'B' : '尚未选择'}</strong></div><div><span>确定程度</span><strong>{replayConfidence(response.confidence)}</strong></div><div><span>完成状态</span><strong>{response.status === 'submitted' ? '已提交' : '草稿'}</strong></div></div>
          <div className="study-replay-pair"><ReplayAnswer label="候选 A" text={item.trial.left?.text || ''} highlights={leftHighlights} selected={response.choice === 'left'} /><ReplayAnswer label="候选 B" text={item.trial.right?.text || ''} highlights={rightHighlights} selected={response.choice === 'right'} /></div>
          {response.selected_reason ? <div className="study-replay-note"><span>选择理由</span><p>{response.selected_reason}</p></div> : null}
          {response.rejected_reason ? <div className="study-replay-note"><span>不选择理由</span><p>{response.rejected_reason}</p></div> : null}
          {!response.selected_reason && response.global_note ? <div className="study-replay-note"><span>旧版整体理由</span><p>{response.global_note}</p></div> : null}</>}
      </article>;
    })}</section>
    {detail.exploratory_feedback ? <section className="study-replay-section study-replay-feedback"><h3>体验反馈</h3><p>{detail.exploratory_feedback}</p></section> : null}
  </aside></div>;
}

function StudyLoading() { return <main className="study-loading"><FlaskConical size={24} /><span>正在读取实验</span></main>; }
function StudyUnavailable({ error }: { error: string }) { return <main className="study-loading"><FlaskConical size={24} /><h1>实验材料尚未就绪</h1><p>{error || '请先在本机准备 Study 1 material_bank.json。'}</p><a href="/">返回产品</a></main>; }
