"""Provider-neutral, explicitly estimated Writer input accounting."""

from math import ceil


def estimate_tokens(text: str) -> int:
    cjk = sum('\u4e00' <= char <= '\u9fff' for char in text)
    return ceil(cjk * 0.8 + (len(text) - cjk) / 4)


def messages_tokens(messages: list[dict[str, str]]) -> int:
    return sum(estimate_tokens(m['content']) + 8 for m in messages) + 3


def explicit_constraint(text: str) -> bool:
    return any(marker in text.lower() for marker in
               ('必须', '不要', '不能', '请记住', '以后', '只用', '只允许', '不允许', '要求',
                'must', 'never', 'do not', "don't", 'only use'))
