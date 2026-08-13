import { FormEvent, useEffect, useState } from 'react';
import { ShieldCheck, UserPlus, UsersRound, X } from 'lucide-react';

import { AdminUser, createAdminUser, fetchAdminUsers } from './api';

export function MemberManagementDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [username, setUsername] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [password, setPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [role, setRole] = useState<'admin' | 'member'>('member');
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  async function refresh() {
    setLoading(true);
    setError('');
    try {
      setUsers(await fetchAdminUsers());
    } catch (reason) {
      setError(String((reason as Error).message || reason));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (open) void refresh();
  }, [open]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError('');
    setNotice('');
    if (password !== confirmation) {
      setError('两次输入的密码不一致');
      return;
    }
    setSubmitting(true);
    try {
      const user = await createAdminUser({
        username: username.trim(),
        password,
        display_name: displayName.trim() || username.trim(),
        role
      });
      setUsers((items) => [...items, user]);
      setNotice(`已创建 ${user.display_name} 的${user.role === 'admin' ? '管理员' : '成员'}账号`);
      setUsername('');
      setDisplayName('');
      setPassword('');
      setConfirmation('');
      setRole('member');
    } catch (reason) {
      setError(String((reason as Error).message || reason));
    } finally {
      setSubmitting(false);
    }
  }

  if (!open) return null;
  return <div className="member-overlay" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
    <aside className="member-drawer" role="dialog" aria-modal="true" aria-label="成员管理">
      <header className="member-header"><div><p>协作空间</p><h2>成员</h2><span>每个账号的对话与长期记忆相互隔离。</span></div><button className="icon-button" type="button" title="关闭" onClick={onClose}><X size={18} /></button></header>
      <div className="member-drawer-body">
        <form className="member-create-form" onSubmit={submit}>
          <div className="member-section-heading"><UserPlus size={17} /><div><h3>添加协作者</h3><p>创建后将账号和初始密码单独发给对方。</p></div></div>
          <div className="member-form-grid"><label>用户名<input autoComplete="off" value={username} onChange={(event) => setUsername(event.target.value)} minLength={2} maxLength={32} placeholder="例如 partner" required /></label><label>显示名称<input autoComplete="off" value={displayName} onChange={(event) => setDisplayName(event.target.value)} maxLength={80} placeholder="例如 袁康宇" /></label></div>
          <label>初始密码<input type="password" autoComplete="new-password" value={password} onChange={(event) => setPassword(event.target.value)} minLength={8} maxLength={256} required /></label>
          <label>确认初始密码<input type="password" autoComplete="new-password" value={confirmation} onChange={(event) => setConfirmation(event.target.value)} minLength={8} maxLength={256} required /></label>
          <fieldset><legend>权限</legend><label className={`member-role-option${role === 'member' ? ' selected' : ''}`}><input type="radio" name="member-role" checked={role === 'member'} onChange={() => setRole('member')} /><span><strong>协作者</strong><small>可使用聊天、作者库与评估；不能管理实验参与者。</small></span></label><label className={`member-role-option${role === 'admin' ? ' selected' : ''}`}><input type="radio" name="member-role" checked={role === 'admin'} onChange={() => setRole('admin')} /><span><strong>管理员</strong><small>可管理实验、参与码、导出和成员账号。</small></span></label></fieldset>
          {error ? <p className="member-error" role="alert">{error}</p> : null}{notice ? <p className="member-notice">{notice}</p> : null}
          <button className="member-create-button" type="submit" disabled={submitting}>{submitting ? '正在创建' : '创建账号'}</button>
        </form>
        <section className="member-list-section"><div className="member-section-heading"><UsersRound size={17} /><div><h3>现有成员</h3><p>{loading ? '正在读取成员' : `共 ${users.length} 位`}</p></div></div><div className="member-list">{users.map((user) => <article key={user.id}><span className="member-avatar">{(user.display_name || user.username).slice(0, 1).toUpperCase()}</span><div><strong>{user.display_name}</strong><small>{user.username}</small></div><span className={`member-role ${user.role}`}><ShieldCheck size={13} />{user.role === 'admin' ? '管理员' : '协作者'}</span></article>)}</div></section>
      </div>
    </aside>
  </div>;
}
