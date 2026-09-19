import pytest

from personaforge.persona.writer import build_writer_messages
from personaforge.persona.context_budget import messages_tokens
from personaforge.web.multiturn import parse_turn_plan


def test_budget_drops_old_history_before_memories_and_preserves_current_query():
    old=[{'role':'user','content':'旧话题'*2500},{'role':'assistant','content':'旧回答'*2500}]
    recent=[{'role':'user','content':'我现在想问学习方法'},{'role':'assistant','content':'先提一个问题'}]
    trace={}
    query='只用中文，逐步解释 Redis AOF，不超过三点。'
    messages=build_writer_messages(query=query,parent_hits=[],response_depth='normal',
        conversation_messages=old+recent,user_memories=['用户偏好渐进解释'],
        input_token_budget=4000,budget_trace=trace,recent_history_message_count=2)
    assert messages[-1]['content'] == query
    assert trace['dropped_components'] == ['relevant_old_history']
    assert trace['final_injected_memory_count'] == 1
    assert messages_tokens(messages)*1.25 <= 4000
    assert trace['before']['relevant_old_history_tokens'] > trace['after']['relevant_old_history_tokens']
    assert trace['after']['total_writer_input_tokens'] == messages_tokens(messages)
    assert len(old) == 2  # caller data untouched


def test_oversize_protected_request_fails_explicitly_without_silent_truncation():
    with pytest.raises(ValueError,match='current request, explicit constraints'):
        build_writer_messages(query='必须保留'+ '完整请求'*4000,parent_hits=[],input_token_budget=1000)


def test_budget_never_silently_drops_previous_explicit_constraint():
    history=[{'role':'user','content':'以后必须只用中文。'+ '条件'*3000},
             {'role':'assistant','content':'好的。'}]
    with pytest.raises(ValueError,match='explicit constraints'):
        build_writer_messages(query='继续',parent_hits=[],conversation_messages=history,
                              input_token_budget=1000)


def test_gate_accepts_zero_and_caps_unique_available_ids():
    plan=parse_turn_plan({'memory_ids':[]},query='Redis 的 AOF 是什么？',available_turn_ids=set(),
                        available_memory_ids={'private-fact'})
    assert plan.memory_ids == []
    plan=parse_turn_plan({'memory_ids':['m1','m1','bad','m2','m3','m4','m5']},query='我的情况',
                        available_turn_ids=set(),available_memory_ids={f'm{i}' for i in range(1,6)})
    assert plan.memory_ids == ['m1','m2','m3','m4']
