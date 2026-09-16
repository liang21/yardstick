from dataclasses import dataclass
from typing import Any

@dataclass
class RunRecord:
    # 身份标识
    run_id: str
    case_id: str
    repeat_idx: str

    # 可重复性四元祖
    model: str
    params: dict[str, Any]
    prompt_version: str
    sdk_fingerprint: str | None

    # 输入输出
    question: str
    output: str | None
    finish_reason: str | None

    # 运行时信息
    error: str | None
    latency_ms: int
    prompt_tokens: int
    collection_tokens: int
    ts: float
