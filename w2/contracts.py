# contracts.py -- 跨层共享的返回契约：零依赖，不碰网络、不碰配置
from dataclasses import dataclass


@dataclass(frozen=True)
class Usage:
    """单次调用的 token 计量。None = 拿不到，绝不写成 0。"""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    raw: dict | None = None  # provider 原始 usage，跨厂商归一时用


@dataclass(frozen=True)
class LLMResult:
    """LlmFn 的唯一返回类型。具名对象，避免位置语义。"""

    content: str  # 空回复兜底 ""，绝不把 None 交给 parse
    usage: Usage | None = None
    latency_ms: float = 0.0
    model: str = ""
    finish_reason: str | None = None


def sum_usage(items: list[Usage | None]) -> tuple[Usage, int]:
    """多轮调用累加。返回 (聚合 Usage, usage 缺失次数)。

    回喂时 messages 逐轮变长，一条样本的真实 token 是每轮之和，
    不是最后一次的值。缺失的轮次单独计数，不参与求和。
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
