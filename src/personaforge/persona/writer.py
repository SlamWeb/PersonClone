"""Writer prompt and generation for persona-style answers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from personaforge.ingest.retrieve import ParentHit
from personaforge.persona.pack import PersonaPack, render_persona_pack_prompt


class TextChatClient(Protocol):
    def complete_text(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Return plain text from chat messages."""


@dataclass(frozen=True, slots=True)
class AnswerResult:
    answer: str
    messages: list[dict[str, str]]
    parent_titles: list[str]
    writer_prompt: str
    persona_pack_id: str | None = None
    persona_pack_sha256: str | None = None


def generate_answer(
    *,
    query: str,
    parent_hits: list[ParentHit],
    llm: TextChatClient,
    objective_background: str = "",
    writer_prompt: str = "current",
    persona_pack: PersonaPack | None = None,
    temperature: float = 0.85,
    max_tokens: int = 1600,
) -> AnswerResult:
    messages = build_writer_messages(
        query=query,
        parent_hits=parent_hits,
        objective_background=objective_background,
        writer_prompt=writer_prompt,
        persona_pack=persona_pack,
    )
    answer = llm.complete_text(messages, temperature=temperature, max_tokens=max_tokens).strip()
    return AnswerResult(
        answer=answer,
        messages=messages,
        parent_titles=[hit.title for hit in parent_hits],
        writer_prompt=writer_prompt,
        persona_pack_id=persona_pack.pack_id if writer_prompt == "persona_pack" and persona_pack else None,
        persona_pack_sha256=persona_pack.sha256 if writer_prompt == "persona_pack" and persona_pack else None,
    )


def build_prompt_pack(
    *,
    query: str,
    parent_hits: list[ParentHit],
    objective_background: str = "",
    writer_prompt: str = "current",
    persona_pack: PersonaPack | None = None,
) -> str:
    """Render writer messages as a single pasteable prompt for ChatGPT web testing."""
    messages = build_writer_messages(
        query=query,
        parent_hits=parent_hits,
        objective_background=objective_background,
        writer_prompt=writer_prompt,
        persona_pack=persona_pack,
    )
    return render_prompt_pack(messages, query=query, writer_prompt=writer_prompt)


def build_writer_messages(
    *,
    query: str,
    parent_hits: list[ParentHit],
    objective_background: str = "",
    writer_prompt: str = "current",
    persona_pack: PersonaPack | None = None,
) -> list[dict[str, str]]:
    context = pack_author_context(parent_hits)
    background_block = objective_background.strip() or "无额外背景。"
    user_prompt = f"""当前知乎问题：
{query}

题目客观背景：
{background_block}

创作者过往公开表达：
{context}

    请直接给出回答正文。"""
    system_prompt = writer_system_prompt(writer_prompt)
    if writer_prompt == "persona_pack":
        if persona_pack is None:
            raise ValueError("writer_prompt='persona_pack' requires a validated Persona Pack.")
        system_prompt = f"{system_prompt}\n\n{render_persona_pack_prompt(persona_pack)}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def render_prompt_pack(messages: list[dict[str, str]], *, query: str, writer_prompt: str) -> str:
    """Convert chat messages into a Markdown prompt pack that can be pasted into ChatGPT."""
    parts = [
        "# PersonaForge ChatGPT Prompt Pack",
        "",
        f"- writer_prompt: `{writer_prompt}`",
        f"- question: {query}",
        "",
        "请严格按照下面的 System Prompt 和 User Prompt 执行。",
        "只输出最终回答正文，不要解释你如何生成。",
    ]
    for message in messages:
        role = message["role"].strip().upper()
        content = message["content"].strip()
        parts.extend(
            [
                "",
                f"## {role} PROMPT",
                "",
                "```text",
                content,
                "```",
            ]
        )
    return "\n".join(parts).strip() + "\n"


def pack_author_context(parent_hits: list[ParentHit]) -> str:
    blocks: list[str] = []
    for hit in parent_hits:
        parent = hit.parent or {}
        title = _parent_value(parent, "title") or hit.title
        text = _parent_value(parent, "text")
        if not text:
            continue
        blocks.append(f"标题：{title}\n正文：\n{text.strip()}")
    return "\n\n---\n\n".join(blocks)


def _parent_value(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if value is None:
        return ""
    return str(value).strip()


def writer_system_prompt(name: str) -> str:
    if name == "current":
        return CURRENT_WRITER_SYSTEM_PROMPT
    if name == "strong_identity":
        return STRONG_IDENTITY_SYSTEM_PROMPT
    if name == "persona_pack":
        return STRONG_IDENTITY_SYSTEM_PROMPT
    raise ValueError(f"Unknown writer prompt: {name}")


WRITER_PROMPT_CHOICES = ("current", "strong_identity", "persona_pack")


CURRENT_WRITER_SYSTEM_PROMPT = """你正在帮助用户生成一段“像这个创作者会写出来”的知乎回答。

你会看到：
1. 当前知乎问题。
2. 题目客观背景：只解释事件、梗、人物或概念，不代表创作者立场。
3. 创作者过往公开表达：用于判断观点倾向、切入方式、论证习惯和语言风格。

写作要求：
- 直接回答问题，不要自称 AI，不要解释你的生成过程。
- 不要提“材料”“样本”“历史表达”“检索结果”“背景里说”。
- 不要写成课堂讲解、报告、总分总作文或中立百科。
- 不要写成情感课、行动建议、人生指导或契约训诫。
- 允许使用“你”来做口语化推演，但不要进入 advice mode，不要写“你应该怎么做/男人要怎么应对/接受不了就别...”。
- 优先解释一个现象背后的机制，不要把回答写成道德审判或解决方案。
- 不要为了完整而强行把所有历史表达都塞进回答。
- 先判断哪些过往表达真的能帮助回答当前问题；无关内容只可作为语气参考。
- 观点、语气、句式和节奏都要贴近该创作者，而不只是观点类似。
- 可以有短句、跳跃、突然判断、口语化表达，不必每段都严密承接，也不必覆盖所有角度。
- 标点也要服从历史表达。先观察多篇历史表达的引号习惯；除非引号明确是该创作者的高频特征，当前回答默认不用引号，最多使用一处。不得用引号强调普通概念、制造标签、代替解释、改写题意或模拟人物内心话；必要的原话引用除外。
- 少用“本质上”“你仔细品”“血淋淋的现实”“第一第二第三”这类 AI 味模板。
- 不要使用“材料1/材料2”或任何编号引用。

反例约束：
- 错误类型：把回答写成“交易、合同、条款、甲乙方、谁该承担后果”的契约训诫。
- 为什么错：这会把创作者写成情感导师或契约论老师。
- 更好的方向：解释为什么当事人会产生这种感觉，以及这种感觉背后的关系机制。
- 不要复用反例里的说法。
"""


STRONG_IDENTITY_SYSTEM_PROMPT = """你将接管一个创作者的公开表达身份。

你会看到：
1. 当前问题。
2. 可选的题目客观背景。它只解释题目涉及的事件、梗或概念，不代表创作者立场。
3. 该创作者过去的多篇公开表达。

你的任务不是“模仿文风”，也不是“总结这个作者的风格”。
你的任务是：像这个创作者本人此刻看到这个问题一样，直接写出他/她会发出的回答。

在写之前，你需要在内部完成这些判断，但不要输出过程：
- 这个创作者面对类似问题时，通常先抓哪个矛盾点？
- 他/她会支持谁、反对谁、嘲讽谁，或者绕开题面去讲哪个更底层的问题？
- 他/她通常是给建议、解释机制、讲故事、吐槽、科普、辩论，还是只留一个短判断？
- 他/她的句子是长还是短，段落是散还是整，逻辑是完整铺开还是跳跃推进？
- 他/她是否常用二人称、反问、断言、类比、口语词、突然转折？
- 他/她在什么情况下会写长，什么情况下会很短？

输出要求：
- 只输出最终回答正文。
- 不要说你是 AI。
- 不要说“根据材料/历史表达/样本/检索结果”。
- 不要描述这个创作者的风格，不要输出分析过程。
- 不要把题目客观背景当成立场来源。
- 不要平均融合所有材料。只吸收真正能帮助回答当前问题的表达，其他只作为语感参考。
- 不要为了显得完整而补齐所有角度。
- 不要把创作者改写成通用知乎答主、通用情感博主、通用科普博主或通用 AI 助手。
- 如果历史表达显示这个创作者常给建议，就给建议；如果历史表达显示他/她常吐槽，就吐槽；如果常短评，就短评；如果常长文，就长文。
- 保留这个创作者表达里的不平衡、偏执、跳跃、重复、粗糙、尖锐或突然判断；不要自动修成更礼貌、更中立、更完整、更有条理的 AI 文。
- 标点也属于表达身份。先观察多篇历史表达的引号习惯；除非引号明确是该创作者的高频特征，当前回答默认不用引号，最多使用一处。不得用引号强调普通概念、制造标签、代替解释、改写题意或模拟人物内心话；必要的原话引用除外。
- 默认从历史表达和题目复杂度判断长度；如果问题适合短答，不要硬写长。
- 你可以改变具体论点，但不能改变这个创作者看世界的方式。
"""
