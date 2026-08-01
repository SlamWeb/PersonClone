import { FormEvent, useState } from 'react';
import { ArrowRight, LockKeyhole } from 'lucide-react';

import { AuthState, bootstrapAuth, loginAuth } from './api';

export function AuthScreen({
  configured,
  onAuthenticated
}: {
  configured: boolean;
  onAuthenticated: (state: AuthState) => void;
}) {
  const [username, setUsername] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [password, setPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!configured && password !== confirmation) {
      setError('两次输入的密码不一致');
      return;
    }
    setSubmitting(true);
    setError('');
    try {
      const state = configured
        ? await loginAuth(username, password)
        : await bootstrapAuth({
            username,
            password,
            display_name: displayName.trim() || username
          });
      onAuthenticated(state);
    } catch (cause) {
      setError(String((cause as Error).message || cause));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="auth-page">
      <section className="auth-panel" aria-labelledby="auth-title">
        <div className="auth-mark" aria-hidden="true">
          <LockKeyhole size={20} />
        </div>
        <div className="auth-heading">
          <span>PersonaForge</span>
          <h1 id="auth-title">{configured ? '登录' : '创建管理员'}</h1>
          <p>{configured ? '继续你的作者对话。' : '首次启动只需完成一次。'}</p>
        </div>

        <form className="auth-form" onSubmit={submit}>
          <label>
            用户名
            <input
              autoComplete="username"
              autoFocus
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              minLength={2}
              maxLength={32}
              required
            />
          </label>
          {!configured ? (
            <label>
              显示名称
              <input
                autoComplete="name"
                value={displayName}
                onChange={(event) => setDisplayName(event.target.value)}
                maxLength={80}
                placeholder="可选"
              />
            </label>
          ) : null}
          <label>
            密码
            <input
              type="password"
              autoComplete={configured ? 'current-password' : 'new-password'}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              minLength={8}
              maxLength={256}
              required
            />
          </label>
          {!configured ? (
            <label>
              确认密码
              <input
                type="password"
                autoComplete="new-password"
                value={confirmation}
                onChange={(event) => setConfirmation(event.target.value)}
                minLength={8}
                maxLength={256}
                required
              />
            </label>
          ) : null}
          {error ? <div className="auth-error" role="alert">{error}</div> : null}
          <button className="auth-submit" type="submit" disabled={submitting}>
            <span>{submitting ? '请稍候' : configured ? '登录' : '创建并进入'}</span>
            <ArrowRight size={17} />
          </button>
        </form>
      </section>
    </main>
  );
}
