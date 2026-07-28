from __future__ import annotations

from personaforge.ingest.retrieve import ParentHit
from personaforge.persona.writer import (
    build_prompt_pack,
    build_writer_messages,
    generate_answer,
    pack_author_context,
    writer_system_prompt,
)


class FakeTextClient:
    def __init__(self, text: str = "生成答案"):
        self.text = text
        self.messages = []

    def complete_text(self, messages, *, temperature=0.7, max_tokens=2048):
        self.messages = messages
        return self.text


def _parent_hit() -> ParentHit:
    return ParentHit(
        rank=1,
        parent_id="zhihu:answer:1",
        score=0.1,
        title="标题",
        path="answer/answer-1.md",
        parent={"doc_id": "zhihu:answer:1", "title": "真实标题", "text": "第一段。\n\n第二段。"},
    )


def test_pack_author_context_uses_parent_title_and_text_without_rank_metadata() -> None:
    context = pack_author_context([_parent_hit()])

    assert "真实标题" in context
    assert "第一段" in context
    assert "rank" not in context
    assert "score" not in context


def test_writer_prompt_forbids_material_references() -> None:
    messages = build_writer_messages(
        query="如何看待某事？",
        parent_hits=[_parent_hit()],
        objective_background="客观背景。",
    )
    combined = "\n".join(message["content"] for message in messages)

    assert "不要提“材料”" in combined
    assert "不要使用“材料1/材料2”" in combined
    assert "不要进入 advice mode" in combined
    assert "契约训诫" in combined
    assert "不要复用反例里的说法" in combined
    assert "客观背景" in combined


def test_generate_answer_calls_text_client() -> None:
    client = FakeTextClient("回答正文")

    result = generate_answer(query="问题", parent_hits=[_parent_hit()], llm=client)

    assert result.answer == "回答正文"
    assert client.messages
    assert result.parent_titles == ["标题"]
    assert result.writer_prompt == "current"


def test_build_prompt_pack_renders_pasteable_chatgpt_markdown() -> None:
    prompt_pack = build_prompt_pack(
        query="问题",
        parent_hits=[_parent_hit()],
        objective_background="客观背景。",
        writer_prompt="strong_identity",
    )

    assert "# PersonaForge ChatGPT Prompt Pack" in prompt_pack
    assert "writer_prompt: `strong_identity`" in prompt_pack
    assert "## SYSTEM PROMPT" in prompt_pack
    assert "## USER PROMPT" in prompt_pack
    assert "真实标题" in prompt_pack
    assert "客观背景。" in prompt_pack


def test_strong_identity_prompt_is_generic_identity_immersion() -> None:
    prompt = writer_system_prompt("strong_identity")

    assert "公开表达身份" in prompt
    assert "不是“模仿文风”" in prompt
    assert "如果历史表达显示这个创作者常给建议，就给建议" in prompt
    assert "不要把创作者改写成通用知乎答主" in prompt
    assert "当前回答默认不用引号，最多使用一处" in prompt
    assert "模拟人物内心话" in prompt


def test_current_prompt_avoids_model_generated_quote_labels() -> None:
    prompt = writer_system_prompt("current")

    assert "当前回答默认不用引号，最多使用一处" in prompt
    assert "模拟人物内心话" in prompt
