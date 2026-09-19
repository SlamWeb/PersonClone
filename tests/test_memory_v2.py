"""Offline contract tests. Model answers are fixtures, not a semantic quality benchmark."""

import json
import sqlite3

import pytest

from test_user_memory import FakeLlm, FakeEncoder, make_turn
from personaforge.web.user_memory import (
    UserMemoryStore, update_user_memories, consolidate_user_memories, memory_retrieval_text,
    recall_user_memories,
)


def atom(mid='m1', quote='我通常喜欢详细解释', **overrides):
    return dict(kind='semantic', memory_key='user.preference.answer_length',
                content='用户通常喜欢详细解释。', sensitivity='normal', importance=3,
                confidence=0.95, event_status='stable', source_message_ids=[mid],
                evidence_quotes=[quote], **overrides)


def update(store, turns, llm, **kw):
    return update_user_memories(store, 'o', author='a', conversation_id='c',
                                user_turns=turns, llm=llm, **kw)


def test_backlog_drains_oldest_windows_and_retains_tail(tmp_path):
    store = UserMemoryStore(tmp_path)
    store.advance_window_checkpoint('o', 'c', 10)
    turns = [make_turn(f'm{i}', '普通问题', i) for i in range(11, 21)]
    llm = FakeLlm([{'candidates': []}] * 3)
    result = update(store, list(reversed(turns)), llm)
    assert [b['through_sequence'] for b in result['batches']] == [13, 16, 19]
    assert [mid for b in result['batches'] for mid in b['source_message_ids']] == [f'm{i}' for i in range(11,20)]
    assert result['pending_turns'] == 1
    assert store.window_checkpoint('o', 'c') == 19


def test_failed_extraction_keeps_checkpoint_at_last_successful_batch(tmp_path):
    store = UserMemoryStore(tmp_path)
    turns = [make_turn(f'm{i}', '普通问题', i) for i in range(1, 7)]
    with pytest.raises(ValueError):
        update(store, turns, FakeLlm([{'candidates': []}, {}]))
    assert store.window_checkpoint('o', 'c') == 3
    result = update(store, turns, FakeLlm([{'candidates': []}]))
    assert result['batches'][0]['source_message_ids'] == ['m4','m5','m6']
    assert store.window_checkpoint('o', 'c') == 6


def test_evidence_and_checkpoint_rollback_together(tmp_path):
    store = UserMemoryStore(tmp_path)
    with store._connect() as db:
        db.execute("CREATE TRIGGER fail_checkpoint BEFORE INSERT ON user_memory_checkpoints BEGIN SELECT RAISE(ABORT, 'disk failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        update(store, [make_turn('m1','我通常喜欢详细解释')], FakeLlm([{'candidates':[atom()]}]), force_flush=True)
    assert store.window_checkpoint('o','c') == 0
    assert store.list_evidence('o') == []


def test_consolidation_retry_does_not_reextract_or_duplicate(tmp_path):
    store = UserMemoryStore(tmp_path)
    turns = [make_turn('m1','我通常喜欢详细解释')]
    with pytest.raises(ValueError):
        update(store, turns, FakeLlm([{'candidates':[atom()]}, {}]), force_flush=True)
    assert store.window_checkpoint('o','c') == 1  # extraction succeeded, Dream did not
    assert len(store.list_evidence('o', status='pending')) == 1
    approved = {**atom(), 'decision':'approve'}
    llm = FakeLlm([{'memories':[approved]}])
    update(store, turns, llm, force_flush=True)
    update(store, turns, FakeLlm([]), force_flush=True)
    assert len(llm.calls) == 1
    assert len(store.list_evidence('o')) == len(store.list_active('o')) == 1


def test_memory_and_evidence_status_rollback_together(tmp_path):
    store = UserMemoryStore(tmp_path)
    turns = [make_turn('m1','我通常喜欢详细解释')]
    update(store, turns, FakeLlm([{'candidates':[atom()]}]), window_size=1)
    with store._connect() as db:
        db.execute("CREATE TRIGGER fail_status BEFORE UPDATE ON user_memory_evidence BEGIN SELECT RAISE(ABORT, 'disk failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        consolidate_user_memories(store,'o',llm=FakeLlm([{'memories':[{**atom(),'decision':'approve'}]}]),force=True)
    assert store.list_active('o') == []
    assert len(store.list_evidence('o',status='pending')) == 1


def test_topic_evidence_accumulates_before_one_consolidation(tmp_path):
    store = UserMemoryStore(tmp_path)
    turns = [make_turn(f'm{i}', '我通常喜欢详细解释', i) for i in range(1,4)]
    candidates = [atom(f'm{i}') for i in range(1,4)]
    candidates[1]['memory_key'] = 'user.preference.response_length'
    candidates[2]['memory_key'] = 'user.communication.conciseness'
    for i in range(2):
        update(store, turns[:i+1], FakeLlm([{'candidates':[candidates[i]]}]), window_size=1)
    assert store.list_active('o') == []
    merged = {**atom(), 'decision':'approve', 'source_message_ids':['m1','m2','m3']}
    result = update(store, turns, FakeLlm([{'candidates':[candidates[2]]},{'memories':[merged]}]), window_size=1)
    assert len(store.list_active('o')) == 1
    assert len(store.list_evidence('o',status='consolidated')) == 3
    assert len(result['consolidation']['attempted_topics']) == 1
    assert {e['topic_key'] for e in store.list_evidence('o')} == {'user.preference.answer_length'}


@pytest.mark.parametrize('query,candidate,reason', [
    ('Redis 的 AOF 是什么？', {**atom(quote='Redis 的 AOF 是什么？'), 'content':'用户正在学习 Redis。'}, 'general_knowledge_question'),
    ('你好', {**atom(quote='你好'), 'source_message_ids':['assistant-1']}, 'non_user_evidence_id'),
    ('是不是短线一定亏光？', {**atom(quote='是不是短线一定亏光？'), 'content':'用户认为短线一定亏光。'}, 'question_promoted_to_belief'),
    ('我哥正在找工作', {**atom(quote='我哥正在找工作'), 'content':'用户正在找工作。'}, 'third_party_attribution_mismatch'),
    ('简单说', {**atom(quote='简单说'), 'content':'用户长期喜欢简短回答。'}, 'temporary_request_not_memory'),
])
def test_unsafe_or_transient_atoms_never_enter_store(tmp_path,query,candidate,reason):
    store=UserMemoryStore(tmp_path)
    result=update(store,[make_turn('m1',query)],FakeLlm([{'candidates':[candidate]}]),force_flush=True)
    assert result['rejections'] == [{'reason':reason}]
    assert store.list_evidence('o') == store.list_active('o') == []


def test_explicit_correction_supersedes_and_keeps_evidence(tmp_path):
    store=UserMemoryStore(tmp_path)
    old_quote='我弟比我小两岁'
    old={**atom(quote=old_quote),'content':'用户弟弟比用户小两岁。','memory_key':'family.brother.age'}
    update(store,[make_turn('m1',old_quote)],FakeLlm([{'candidates':[old]},{'memories':[{**old,'decision':'approve'}]}]),force_flush=True)
    old_id=store.list_active('o')[0].id
    query='前面说错了，不是小两岁，是小一岁。'
    new={**atom('m2',query),'content':'用户弟弟比用户小一岁。','memory_key':'family.brother.age'}
    update(store,[make_turn('m2',query,2)],FakeLlm([{'candidates':[new]},{'memories':[{**new,'decision':'approve','revision_mode':'replace'}]}]))
    assert store.list_active('o')[0].supersedes_id == old_id
    assert store.get('o',old_id).status == 'superseded'
    assert len(store.list_evidence('o')) == 2


def test_forget_suppresses_pending_evidence_and_cannot_dream_it_back(tmp_path):
    store=UserMemoryStore(tmp_path)
    a=atom()
    update(store,[make_turn('m1','我通常喜欢详细解释')],FakeLlm([{'candidates':[a]},{'memories':[{**a,'decision':'approve'}]}]),force_flush=True)
    update(store,[make_turn('m2','我通常喜欢详细解释',2)],FakeLlm([{'candidates':[atom('m2')]}]),window_size=1)
    store.forget('o',store.list_active('o')[0].id)
    consolidate_user_memories(store,'o',llm=FakeLlm([]),force=True)
    assert store.list_active('o') == []
    assert {e['status'] for e in store.list_evidence('o')} == {'forgotten'}


def test_restricted_evidence_payload_and_trace_do_not_keep_exact_values(tmp_path):
    store=UserMemoryStore(tmp_path)
    query='我哥十倍杠杆贷款，两天亏掉20w，我担心他又贷款。'
    a={**atom(quote=query),'content':'用户担心哥哥亏掉20w后继续贷款。','sensitivity':'restricted'}
    result=update(store,[make_turn('m1',query)],FakeLlm([{'candidates':[a]},{'memories':[{**a,'decision':'approve'}]}]),force_flush=True)
    serialized=json.dumps([store.list_evidence('o'),result,[m.to_api() for m in store.list_active('o')]],ensure_ascii=False)
    assert '20w' not in serialized and '十倍' not in serialized
    assert store.list_evidence('o')[0]['evidence_quotes_json'] == '[]'


def test_context_dependent_query_recall_uses_recent_user_context(tmp_path):
    store=UserMemoryStore(tmp_path)
    candidate={**atom(),'memory_key':'family.brother.trading','content':'用户担心哥哥高风险交易。'}
    store.save_revision('o', **candidate, source_author='a',source_conversation_id='c')
    text=memory_retrieval_text('那我现在怎么办？',[make_turn('m1','我哥又要充值炒股')])
    hits=recall_user_memories(store,'o',text,encoder=FakeEncoder(),model='fake')
    assert '我哥又要充值炒股' in text
    assert hits[0].memory.memory_key == 'family.brother.trading'
    assert memory_retrieval_text('Redis 的 AOF 是什么？',[make_turn('m1','我哥炒股')]) == 'Redis 的 AOF 是什么？'


def test_evidence_is_owner_isolated(tmp_path):
    store=UserMemoryStore(tmp_path)
    update(store,[make_turn('m1','我通常喜欢详细解释')],FakeLlm([{'candidates':[atom()]}]),window_size=1)
    assert store.list_evidence('other') == []
    assert consolidate_user_memories(store,'other',llm=FakeLlm([]),force=True)['operations'] == []


def test_manual_correction_suppresses_old_pending_evidence(tmp_path):
    store=UserMemoryStore(tmp_path)
    a=atom()
    update(store,[make_turn('m1','我通常喜欢详细解释')],FakeLlm([{'candidates':[a]},{'memories':[{**a,'decision':'approve'}]}]),force_flush=True)
    update(store,[make_turn('m2','我通常喜欢详细解释',2)],FakeLlm([{'candidates':[atom('m2')]}]),window_size=1)
    corrected=store.correct('o',store.list_active('o')[0].id,'用户以后偏好简洁解释。')
    consolidate_user_memories(store,'o',llm=FakeLlm([]),force=True)
    assert store.list_active('o')[0].id == corrected.id
    assert len(store.list_evidence('o',status='superseded')) == 1


def test_new_state_supersedes_old_state_without_latest_wins_for_local_brevity(tmp_path):
    store=UserMemoryStore(tmp_path)
    old={**atom(quote='我正在秋招'),'content':'用户正在秋招。','memory_key':'user.career.search','event_status':'ongoing','kind':'episodic'}
    update(store,[make_turn('m1','我正在秋招')],FakeLlm([{'candidates':[old]},{'memories':[{**old,'decision':'approve'}]}]),force_flush=True)
    old_id=store.list_active('o')[0].id
    new={**atom('m2','我已经签约，秋招结束了'),'content':'用户已经签约，秋招结束。','memory_key':'user.career.search','event_status':'historical','kind':'episodic'}
    update(store,[make_turn('m2','我已经签约，秋招结束了',2)],FakeLlm([{'candidates':[new]},{'memories':[{**new,'decision':'approve','revision_mode':'replace'}]}]),window_size=1)
    assert store.list_active('o')[0].supersedes_id == old_id
    assert '正在秋招' not in store.list_active('o')[0].content
    assert len(store.list_evidence('o')) == 2


def test_consolidator_synthesis_replaces_string_concat(tmp_path):
    store=UserMemoryStore(tmp_path)
    old=atom()
    update(store,[make_turn('m1','我通常喜欢详细解释')],FakeLlm([{'candidates':[old]},{'memories':[{**old,'decision':'approve'}]}]),force_flush=True)
    quote='请记住我喜欢有具体例子'
    new={**atom('m2',quote),'content':'用户喜欢具体例子。'}
    synthesis={**new,'decision':'approve','revision_mode':'extend','content':'用户偏好详细且结合实例的解释。'}
    update(store,[make_turn('m2',quote,2)],FakeLlm([{'candidates':[new]},{'memories':[synthesis]}]))
    memory=store.list_active('o')[0]
    assert memory.content == synthesis['content']
    assert memory.source_message_ids == ['m1','m2']


def test_temporary_brevity_preserves_existing_detailed_preference(tmp_path):
    store=UserMemoryStore(tmp_path)
    update(store,[make_turn('m1','我通常喜欢详细解释')],FakeLlm([{'candidates':[atom()]},{'memories':[{**atom(),'decision':'approve'}]}]),force_flush=True)
    original=store.list_active('o')[0]
    transient={**atom('m2','简单说'),'content':'用户喜欢简短回答。'}
    update(store,[make_turn('m2','简单说',2)],FakeLlm([{'candidates':[transient]}]),force_flush=True)
    assert store.list_active('o')[0].id == original.id


def test_checkpoint_compare_and_swap_prevents_duplicate_batch(tmp_path):
    store=UserMemoryStore(tmp_path)
    update(store,[make_turn('m1','我通常喜欢详细解释')],FakeLlm([{'candidates':[atom()]}]),window_size=1)
    with pytest.raises(RuntimeError, match='checkpoint changed'):
        store.commit_evidence_batch('o','c',0,1,[atom()])
    assert len(store.list_evidence('o')) == 1
