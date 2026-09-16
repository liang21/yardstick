import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ["LLM_API_KEY"], api_base=os.environ["LLM_API_BASE"])

PROMPT_TEMPLATE = """你是一个客服助手，请简洁准确地回答用户问题。

用户问题：{question}"""
PROMPT_VERSION = "v1"


def call_once(question: str, **params) -> dict:
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=os.environ["LLM_MODEL"],
            messages=[
                {"role": "user", "content": PROMPT_VERSION.format(question=question)},
            ],
            **params,
        )
        choice = resp.choices[0]
        return {
            "output": choice.message.content,
            "finish_reason": choice.finish_reason,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "error": None,
            "sdk_fingerprint": getattr(resp, "system_fingerprint", None),
        }
    except Exception as e:
        return {
            "output": None,
            "finish_reason": None,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": f"{type(e).__name__}: {e}",
            "sdk_fingerprint": None,
        }


def append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
