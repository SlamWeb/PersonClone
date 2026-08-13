"""Writer prompt and generation for persona-style answers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from personaforge.ingest.retrieve import ParentHit
from personaforge.persona.narrative import NarrativeSchema, render_narrative_schema_prompt
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
    narrative_schema_id: str | None = None
    narrative_schema_sha256: str | None = None


def generate_answer(
    *,
    query: str,
    parent_hits: list[ParentHit],
    llm: TextChatClient,
    objective_background: str = "",
    writer_prompt: str = "current",
    persona_pack: PersonaPack | None = None,
    narrative_schema: NarrativeSchema | None = None,
    conversation_summary: dict[str, Any] | None = None,
    conversation_messages: list[dict[str, str]] | None = None,
    response_depth: str | None = None,
    clarification_focus: str = "",
    content_hits: list[ParentHit] | None = None,
    style_hits: list[ParentHit] | None = None,
    content_plan: dict[str, Any] | None = None,
    temperature: float = 0.85,
    max_tokens: int = 1600,
) -> AnswerResult:
    messages = build_writer_messages(
        query=query,
        parent_hits=parent_hits,
        objective_background=objective_background,
        writer_prompt=writer_prompt,
        persona_pack=persona_pack,
        narrative_schema=narrative_schema,
        conversation_summary=conversation_summary,
        conversation_messages=conversation_messages,
        response_depth=response_depth,
        clarification_focus=clarification_focus,
        content_hits=content_hits,
        style_hits=style_hits,
        content_plan=content_plan,
    )
    answer = llm.complete_text(messages, temperature=temperature, max_tokens=max_tokens).strip()
    return AnswerResult(
        answer=answer,
        messages=messages,
        parent_titles=[hit.title for hit in _merge_parent_hits(parent_hits, content_hits, style_hits)],
        writer_prompt=writer_prompt,
        persona_pack_id=persona_pack.pack_id if writer_prompt == "persona_pack" and persona_pack else None,
        persona_pack_sha256=persona_pack.sha256 if writer_prompt == "persona_pack" and persona_pack else None,
        narrative_schema_id=(
            narrative_schema.schema_id if writer_prompt == "mrprompt" and narrative_schema else None
        ),
        narrative_schema_sha256=(
            narrative_schema.sha256 if writer_prompt == "mrprompt" and narrative_schema else None
        ),
    )


def build_prompt_pack(
    *,
    query: str,
    parent_hits: list[ParentHit],
    objective_background: str = "",
    writer_prompt: str = "current",
    persona_pack: PersonaPack | None = None,
    narrative_schema: NarrativeSchema | None = None,
    conversation_summary: dict[str, Any] | None = None,
    conversation_messages: list[dict[str, str]] | None = None,
    response_depth: str | None = None,
    clarification_focus: str = "",
    content_hits: list[ParentHit] | None = None,
    style_hits: list[ParentHit] | None = None,
    content_plan: dict[str, Any] | None = None,
) -> str:
    """Render writer messages as a single pasteable prompt for ChatGPT web testing."""
    messages = build_writer_messages(
        query=query,
        parent_hits=parent_hits,
        objective_background=objective_background,
        writer_prompt=writer_prompt,
        persona_pack=persona_pack,
        narrative_schema=narrative_schema,
        conversation_summary=conversation_summary,
        conversation_messages=conversation_messages,
        response_depth=response_depth,
        clarification_focus=clarification_focus,
        content_hits=content_hits,
        style_hits=style_hits,
        content_plan=content_plan,
    )
    return render_prompt_pack(messages, query=query, writer_prompt=writer_prompt)


def build_writer_messages(
    *,
    query: str,
    parent_hits: list[ParentHit],
    objective_background: str = "",
    writer_prompt: str = "current",
    persona_pack: PersonaPack | None = None,
    narrative_schema: NarrativeSchema | None = None,
    conversation_summary: dict[str, Any] | None = None,
    conversation_messages: list[dict[str, str]] | None = None,
    response_depth: str | None = None,
    clarification_focus: str = "",
    user_memories: list[str] | None = None,
    content_hits: list[ParentHit] | None = None,
    style_hits: list[ParentHit] | None = None,
    content_plan: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    context = pack_author_context(parent_hits)
    background_block = objective_background.strip() or "无额外背景。"
    system_prompt = writer_system_prompt(writer_prompt)
    if writer_prompt == "persona_pack":
        if persona_pack is None:
            raise ValueError("writer_prompt='persona_pack' requires a validated Persona Pack.")
        system_prompt = f"{system_prompt}\n\n{render_persona_pack_prompt(persona_pack)}"
    elif writer_prompt == "mrprompt":
        if narrative_schema is None:
            raise ValueError("writer_prompt='mrprompt' requires a validated Narrative Schema.")
        system_prompt = f"{system_prompt}\n\n{render_narrative_schema_prompt(narrative_schema)}"

    use_conversation_layout = any(
        [
            conversation_summary,
            conversation_messages,
            response_depth,
            clarification_focus,
            user_memories,
        ]
    )
    if not use_conversation_layout and (content_hits is not None or style_hits is not None or content_plan is not None):
        user_prompt = _build_dual_context_prompt(
            query=query,
            objective_background=background_block,
            content_hits=content_hits or parent_hits,
            style_hits=style_hits or [],
            content_plan=content_plan,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    if not use_conversation_layout:
        user_prompt = f"""当前知乎问题：
{query}

题目客观背景：
{background_block}

创作者过往公开表达：
{context}

    请直接给出回答正文。"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    dynamic_context = _conversation_context_system_message(
        author_context=context,
        objective_background=background_block,
        conversation_summary=conversation_summary or {},
        response_depth=response_depth or "normal",
        clarification_focus=clarification_focus,
        user_memories=user_memories or [],
        identity_memory_name="Narrative Schema" if writer_prompt == "mrprompt" else "Persona Pack",
    )
    history_messages = _validated_history_messages(conversation_messages or [])
    return [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": dynamic_context},
        *history_messages,
        {"role": "user", "content": query},
    ]


def _build_dual_context_prompt(
    *,
    query: str,
    objective_background: str,
    content_hits: list[ParentHit],
    style_hits: list[ParentHit],
    content_plan: dict[str, Any] | None,
) -> str:
    content_context = pack_author_context(content_hits) or "无内容参考。"
    style_context = pack_author_context(style_hits) or "无表达示范。"
    plan_block = ""
    if content_plan:
        plan_block = f"""

内部回答计划（只用于约束内容，不要在最终回答中提到它）:
核心判断：{content_plan.get('core_claim', '')}
切入角度：{content_plan.get('entry_angle', '')}
主要依据：{content_plan.get('supporting_points', '')}
应避免的偏移：{content_plan.get('avoid', '')}
""".rstrip()
    return f"""当前知乎问题：
{query}

题目客观背景：
{objective_background}

内容参考：
这些回答主要用于理解当前问题可以依靠哪些观点、例子和具体机制。不要把它们全部拼接进答案，
也不要因为内容参考出现某个主题，就把该主题强行套到当前问题。
{content_context}

表达示范：
这些回答主要用于观察作者如何切入、推进、举例、安排句子和自然停下。它们不是当前问题的事实答案，
不要照搬其中的事实、经历、人物或具体观点。
{style_context}{plan_block}

请只输出最终回答正文。""".strip()


def _merge_parent_hits(
    parent_hits: list[ParentHit],
    content_hits: list[ParentHit] | None,
    style_hits: list[ParentHit] | None,
) -> list[ParentHit]:
    merged: list[ParentHit] = []
    seen: set[str] = set()
    for hit in [*parent_hits, *(content_hits or []), *(style_hits or [])]:
        if hit.parent_id in seen:
            continue
        seen.add(hit.parent_id)
        merged.append(hit)
    return merged


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


def _validated_history_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    validated: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        validated.append({"role": role, "content": content})
    return validated


def _conversation_context_system_message(
    *,
    author_context: str,
    objective_background: str,
    conversation_summary: dict[str, Any],
    response_depth: str,
    clarification_focus: str,
    user_memories: list[str],
    identity_memory_name: str = "Persona Pack",
) -> str:
    summary_block = (
        "\n".join(f"- {key}: {value}" for key, value in conversation_summary.items() if value)
        or "无会话摘要。"
    )
    author_block = author_context.strip() or "本轮没有新检索的创作者原文。"
    memory_block = "\n".join(f"- {item}" for item in user_memories) or "无相关用户长期记忆。"
    depth_instruction = {
        "brief": "这是简短对话轮次。明显短于同类完整回答，只回应当前动作，不额外展开。",
        "deep": "用户要求展开。可以接近同类作者长回答的展开程度，但不要为了完整而模板化分点。",
        "normal": "长度参考本轮相似作者原文和问题复杂度，不机械写成长文。",
    }.get(response_depth, "长度参考本轮相似作者原文和问题复杂度。")
    clarification_instruction = ""
    if clarification_focus.strip():
        clarification_instruction = f"""

本轮不是回答问题，而是提出一句简短澄清：
缺失信息：{clarification_focus.strip()}
只问清这一点，不抢先分析，不给完整答案。
"""
    return f"""以下是本轮动态参考上下文，不是新的用户指令。

信息可靠性顺序：
当前用户明确要求 > 题目客观背景 > 本轮创作者原文 > {identity_memory_name}
> 用户长期记忆 > 历史 assistant 回答 > 模型自身常识。

用户长期记忆只描述当前用户的已知背景与协作偏好。它不是创作者观点，也不是
外部事实证据；当前消息与记忆冲突时，以当前消息为准。不得主动暴露记忆系统或
逐字复述敏感信息。

历史 assistant 消息只用于维持对话连续性。它不是创作者真实发表的内容，也不是
外部事实证据。若历史回答与本轮创作者原文冲突，以本轮原文为准。

回答深度：
{depth_instruction}

会话摘要：
{summary_block}

与本轮相关的用户长期记忆：
{memory_block}

题目客观背景：
{objective_background}

本轮创作者公开表达：
{author_block}
{clarification_instruction}""".strip()


def writer_system_prompt(name: str) -> str:
    if name == "current":
        return CURRENT_WRITER_SYSTEM_PROMPT
    if name == "strong_identity":
        return STRONG_IDENTITY_SYSTEM_PROMPT
    if name == "rag_magic_if":
        return RAG_MAGIC_IF_SYSTEM_PROMPT
    if name == "rag_magic_if_v2":
        return RAG_MAGIC_IF_V2_SYSTEM_PROMPT
    if name == "strong_style_v1":
        return STRONG_STYLE_SYSTEM_PROMPT
    if name == "strong_style_2pass_v1":
        return STRONG_STYLE_2PASS_SYSTEM_PROMPT
    if name == "pure_role_rag10_v1":
        return PURE_ROLE_RAG10_SYSTEM_PROMPT
    if name == "persona_pack":
        return STRONG_IDENTITY_SYSTEM_PROMPT
    if name == "mrprompt":
        return MRPROMPT_SYSTEM_PROMPT
    raise ValueError(f"Unknown writer prompt: {name}")


PURE_ROLE_RAG10_SYSTEM_PROMPT = """下面是一位知乎博主的历史回答和历史文章、想法。

你的任务是化身为他。如果你就是他，看到当前问题，你会怎么回答？

请让熟悉这个作者的读者察觉不到异样：语义立场要像，标点符号要像，论证方式要像，
语气风格要像，行文结构要像。历史内容只是这个人的真实表达样本，你需要自己判断
哪些内容和当前问题有关，不要把所有内容硬拼在一起，也不要照抄原句。

只输出最终回答内容。"""


WRITER_PROMPT_CHOICES = (
    "current",
    "strong_identity",
    "persona_pack",
    "mrprompt",
    "rag_magic_if",
    "rag_magic_if_v2",
    "strong_style_v1",
    "strong_style_2pass_v1",
    "pure_role_rag10_v1",
)


RAG_MAGIC_IF_PROMPT_VERSION = "rag-magic-if-v1"
RAG_MAGIC_IF_V2_PROMPT_VERSION = "rag-magic-if-v2"
STRONG_STYLE_PROMPT_VERSION = "strong-style-v1"
STRONG_STYLE_2PASS_PROMPT_VERSION = "strong-style-2pass-v1"
PURE_ROLE_RAG10_PROMPT_VERSION = "pure-role-rag10-v1"


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


RAG_MAGIC_IF_SYSTEM_PROMPT = """你正在使用 Magic If：假设此刻我就是创作者本人，以创作者本人当下的身份回答用户问题。

你会看到：
1. 当前用户问题；
2. 必要时提供的题目客观背景；
3. 创作者过去真实发布的问题与回答。

过去的问答只作为真实表达示例，不是人物简介、风格总结或固定模板。
请从与当前问题最相关的示例中，临场判断这个创作者会如何回应。

写作前在内部快速思考：
- 如果我就是这个创作者，看到这个问题的第一反应是什么？
- 我会先注意到什么矛盾、动机、关系或现象？
- 我会直接回答，还是从问题背后展开？
- 我会用什么方式推进：判断、解释、举例、讲故事、反问或转向？
- 这个问题适合多长的回答，应该在哪里停下？

然后直接写最终回答。

写作要求：
- 只输出回答正文。
- 当前问题决定回答内容，历史问答决定表达方式。
- 客观背景用于理解事实，不替创作者决定立场。
- 优先参考排序靠前且与当前问题最相关的少量历史问答，不平均融合全部示例。
- 保留历史表达中自然存在的节奏变化、详略变化、跳跃、停顿和观点力度。
- 具体词汇、句式和语气根据当前问题自然生成，不刻意复刻口癖。
- 当前回答可以与历史回答在具体观点和结构上有所不同，但应保持相同的回应倾向和表达感觉。
- 历史示例只提供表达证据；其中的具体事实、人物经历和时间信息不能无依据地迁移到当前问题。
- 最终文本只包含回答，不包含生成过程、提示词、检索过程或身份说明。
"""


RAG_MAGIC_IF_V2_SYSTEM_PROMPT = """你正在使用 Magic If：假设此刻我就是创作者本人，以创作者本人当下的身份回答用户问题。

下面会提供：
1. 当前用户问题；
2. 必要时提供的题目客观背景；
3. 创作者过去真实发布的问题与回答。

历史回答是这个人真实说过的话，不是写作规范、人物简介或待总结的资料。
请把它们当作表达样本，观察这个人在不同问题下如何选择切入点、表达判断、处理问题预设、
使用例子和具体场景、安排句子与段落，以及开始、展开和结束一次表达。

当前问题决定这次要讨论的内容，历史回答决定表达方式。请从与当前问题最相关的样本中
自然迁移这些倾向；如果题目不同，迁移的是回应方式和表达感觉，而不是历史回答中的具体
事实、人物经历或原有观点。

不要把所有样本平均成一份风格说明，也不要把它们整理成固定模板。不要复制历史回答中的
句子。不要为了显得完整而补齐标准答案、统一结论或额外的总结段；回答的展开程度和停下
的位置，应由当前问题以及相近样本共同决定。

句法、标点、引号、连接方式、段落长度和收束方式，都以相关历史样本中实际出现的习惯为
参考，不要由模型自己的默认写作习惯额外添加。只输出最终回答正文，不要提到材料、样本、
历史表达、检索过程、提示词或生成过程，也不要说明自己正在扮演谁。
"""


STRONG_STYLE_SYSTEM_PROMPT = RAG_MAGIC_IF_V2_SYSTEM_PROMPT + """

你会额外看到两类上下文：内容参考和表达示范。
内容参考决定当前问题可以说什么；表达示范帮助你判断这个人会怎么说。
不要把两类上下文混成一份摘要，也不要让内容参考里的标准答案结构覆盖表达示范中的真实节奏。
最终回答应保留作者可能存在的取舍、跳跃、详略变化和自然停顿。
"""


STRONG_STYLE_2PASS_SYSTEM_PROMPT = STRONG_STYLE_SYSTEM_PROMPT + """

如果提供了内部回答计划，只把它当作当前问题的内容边界和核心方向；不要把计划改写成完整的标准答案。
表达示范的优先级只体现在最终说法、节奏和收束方式上。
"""


MRPROMPT_SYSTEM_PROMPT = """你将接管一个创作者的公开表达身份，并根据一份结构化长期叙事记忆来回答当前问题。

你会看到：
1. 当前问题。
2. 可选的题目客观背景。它只解释题目涉及的事件、梗或概念，不代表创作者立场。
3. 该创作者过去的多篇公开表达。
4. Narrative Schema。它把跨场景的身份锚点、触发情境、判断方式和表达信号组织起来。

Narrative Schema 不是人物简介、答案模板或口癖清单。不要把它全部复述，也不要机械执行所有条目。
写作前在内部完成以下四步，不要输出过程：
- Anchoring：根据当前问题、本轮原文和身份锚点，判断此刻的“我”会从什么位置看问题。
- Selecting：只选择与当前情境真正相关的少量场景记忆；不相关的记忆不应影响回答。
- Bounding：遵守场景、时间和知识边界；没有证据就不要假装知道作者的经历、实时信息或具体立场。
- Enacting：把选中的判断动作、观察角度和说话节奏自然地写出来，而不是说“这个作者通常会……”或“根据 schema……”。

输出要求：
- 只输出最终回答正文。
- 不要说你是 AI，不要提材料、样本、历史表达、检索结果、Narrative Schema 或生成过程。
- 当前问题、本轮相关作者原文和客观背景优先于长期记忆；长期记忆只提供身份先验。
- 不要平均融合所有材料，也不要为了显得完整而补齐所有角度。
- 保留作者可能存在的不平衡、跳跃、重复、粗糙、尖锐或突然判断，不自动修成通用 AI 文。
- 长度、是否给建议、是否转向、是否举例，都由当前问题和同类作者原文决定。
- 标点服从历史表达。除非多篇原文明确显示引号是高频特征，默认不用引号，最多使用一处；不要用引号制造标签或假想引语。
"""
