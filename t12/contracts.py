from dataclasses import dataclass


@dataclass
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    raw: dict | None = None


@dataclass(frozen=True)
class LLMResult:
    content: str
    usage: Usage
    latency_ms: float = 0.0
    model: str = ""
    finish_reason: str | None = None


def sum_usage(items: list[Usage | None]):
    """多轮调用累加。返回（聚合 Usage，usage 缺失次数）。

    回喂时message 逐轮变长，一条样本的真实 token 是每轮之和，
    不是最后一次的值。缺失的轮次单独计数，不参与求和
    """
    prompt = completion = total = 0
    missing = 0
    for u in items:
        if u is None:
            missing += 1
            continue
        p = u.prompt_tokens or 0
        c = u.completion_tokens or 0
        prompt += p
        completion += c

        total += u.total_tokens if u.total_tokens is not None else p + c

    return (
        Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        ),
        missing,
    )
