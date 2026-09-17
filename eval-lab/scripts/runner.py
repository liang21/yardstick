import collections
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# 允许直接以文件路径运行（PyCharm 默认方式），而不只是 python -m scripts.runner
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analyze import RunRecord

load_dotenv(ROOT / ".env")

client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ["LLM_BASE_URL"])

PROMPT_TEMPLATE = """你是一个客服助手，请简洁准确地回答用户问题。

用户问题：{question}"""
QUESTION = "用一个词形容今天的天气"
PROMPT_VERSION = "v1"

MODEL = os.environ["LLM_MODEL"]
DEFAULT_PARAMS = {
    "temperature": 0.0,
    "max_tokens": 20,
    "stream": False,
}


def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


def call_once(question: str, **params) -> dict:
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": question},
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


# ---------- 主流程 ----------
def main(repeats: int = 5, params: dict | None = None):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = ROOT / "results" / f"{run_id}.jsonl"
    df = pd.read_csv(ROOT / "data" / "questions.csv")
    p = {**DEFAULT_PARAMS, **(params or {})}

    print(f"run_id={run_id}  {len(df)} 题 × {repeats} 次 = {len(df) * repeats} 次调用")
    for row in df.itertuples():
        for i in range(repeats):
            r = call_once(row.question, **p)
            rec = RunRecord(
                run_id=run_id, case_id=row.case_id, repeat_idx=i,
                model=MODEL, params=p, prompt_version=PROMPT_VERSION,
                question=row.question, ts=time.time(), **r,
            )
            append_jsonl(out_path, asdict(rec))

    # 跑完立刻给基础统计，不要只写文件不看
    res = pd.read_json(out_path, lines=True)
    print(f"\n成功 {res['error'].isna().sum()}/{len(res)}")
    print(f"finish_reason 分布:\n{res['finish_reason'].value_counts(dropna=False)}")
    print(f"截断(length) {(res['finish_reason'] == 'length').sum()} 条")
    print(f"token: 输入 {res['prompt_tokens'].sum()}，输出 {res['completion_tokens'].sum()}")
    print(f"\n每题唯一答案数:\n{res[res['error'].isna()].groupby('case_id')['output'].nunique()}")
    print(f"\n结果文件: {out_path}")


if __name__ == "__main__":
    N = 20
    for temp in [0.0, 0.3, 0.7, 1.0, 1.5]:
        outputs, errors = [], []
        for _ in range(N):
            r = call_once(QUESTION, temperature=temp, max_tokens=1024)
            if r["error"]:
                errors.append(r["error"])
                continue
            outputs.append((r["output"] or "").strip())
        if errors:
            print(f"T={temp}  失败 {len(errors)}/{N}，首个错误: {errors[0]}")
        if not outputs:
            print(f"T={temp}  无有效样本\n")
            continue
        counter = collections.Counter(outputs)
        print(f"T={temp}  有效样本={len(outputs)}  唯一答案数={len(counter)}  "
              f"最高频占比={counter.most_common(1)[0][1] / len(outputs):.0%}")
        print(f"  分布: {[(o[:30], c) for o, c in counter.most_common(5)]}\n")
