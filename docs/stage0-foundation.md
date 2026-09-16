# 阶段 0：打地基 —— 详细学习资料

> 周期：2 周 | 每周 8–10 小时
> 目标：能用 Python 批量调模型、把结果结构化落盘、用 pandas 分析输出的不确定性

---

## 一、本阶段在整条路径中的定位

后面四个阶段（RAG 评测、judge 校准、Agent 评测、CI 回归）全部建立在一个能力上：**把一批输入跑过模型，拿到可分析的结构化结果**。

这个"跑批"脚本你会用整整五个月，所以阶段 0 的真实目标不是"学会 Python"，而是：

1. **建立不确定性的体感** —— 亲眼看到同一个问题跑 5 次给出 5 个不同答案
2. **写出第一版评测 runner** —— 后面每个阶段都在它基础上加东西
3. **养成记录元数据的习惯** —— 这是阶段 4"四元版本追溯"的地基

> 如果你只带走一件事：**评测结果必须连同"是什么模型、什么参数、什么 prompt 版本、第几次重复"一起落盘。** 只存答案的脚本，一周后就变成垃圾。

---

## 二、前置准备

### 2.1 环境

```bash
# 推荐 uv（快），也可以用 venv
uv venv && source .venv/bin/activate
uv pip install openai pandas python-dotenv tenacity jupyterlab
```

建议目录结构，从第一天就按这个来：

```
eval-lab/
├── .env                 # API key，务必加进 .gitignore
├── data/
│   └── questions.csv    # 输入
├── results/             # 原始结果（jsonl），只追加不修改
├── scripts/
│   ├── runner.py        # 跑批
│   └── analyze.py       # 分析
└── notebooks/           # 探索性分析
```

### 2.2 模型接入

大多数国内外服务都兼容 OpenAI SDK 协议，用 `base_url` 切换即可。建议**至少接两家**，后面阶段 4 做 A/B 时会用上。

`.env`：

```
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.example.com/v1
LLM_MODEL=your-model-name
```

### 2.3 成本预算

阶段 0 用**最便宜的小模型**跑通流程即可，不要一上来就用旗舰模型。

粗算：100 问题 × 5 次重复 × (输入 100 token + 输出 300 token) = 20 万 token。按国产小模型价格，一轮成本通常在几块钱以内。你大概会跑 10–20 轮，预算控制在 100 元内足够。

**从第一天就在脚本里统计 token 和成本**，这是阶段 4"评测成本怎么裁剪"那道面试题的答案来源。

---

## 三、两周日程

| 天 | 模块 | 产出 |
|---|---|---|
| D1–D2 | 环境 + 第一次调用 + 结构化落盘 | `runner.py` v0，能跑通 5 个问题 |
| D3–D4 | 采样参数实验 | 温度分布实验报告 |
| D5–D7 | 并发、重试、断点续跑、成本统计 | `runner.py` v1，能稳定跑 500 次调用 |
| D8–D10 | pandas 核心操作 | `analyze.py` 能出基础统计 |
| D11–D12 | 输出变异性分析 | 一致率 / 长度 CV / 相似度 |
| D13–D14 | 整合 + 写验收结论 | 完整跑批报告 |

---

## 四、第 1 周

### 模块 1（D1–D2）：第一次调用与数据 schema

#### 要点

新手最容易犯的错：把模型返回的文本直接 `print` 出来看，或者只存一列答案到 CSV。**结果一旦丢失上下文就无法复现**。

先设计记录结构，再写调用代码。

#### 评测记录的最小 schema

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class RunRecord:
    # 身份标识
    run_id: str            # 一次跑批的唯一 ID，如 20260915-143022
    case_id: str           # 问题 ID，必须稳定不变
    repeat_idx: int        # 第几次重复，0-based

    # 可复现性四元组（阶段 4 会扩展）
    model: str
    params: dict[str, Any]     # temperature / top_p / max_tokens / seed
    prompt_version: str        # 如 "v1"，prompt 模板的版本号
    sdk_fingerprint: str | None  # 服务端返回的 system_fingerprint（若有）

    # 输入输出
    question: str
    output: str | None
    finish_reason: str | None   # stop / length / tool_calls / content_filter

    # 运行时信息
    error: str | None
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    ts: float                   # 时间戳
```

`finish_reason` 这一列是新手最常漏的。**`finish_reason == "length"` 表示答案被 `max_tokens` 截断了**，如果不单独统计，你会把"截断"误判成"模型答不好"——这是阶段 1 之后最常见的误诊来源之一。

#### 最简调用

```python
import os, time, json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],
)

PROMPT_TEMPLATE = """你是一个客服助手，请简洁准确地回答用户问题。

用户问题：{question}"""
PROMPT_VERSION = "v1"

def call_once(question: str, **params) -> dict:
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=os.environ["LLM_MODEL"],
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
            **params,
        )
        choice = resp.choices[0]
        return {
            "output": choice.message.content,
            "finish_reason": choice.finish_reason,
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "sdk_fingerprint": getattr(resp, "system_fingerprint", None),
            "error": None,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }
    except Exception as e:
        return {
            "output": None, "finish_reason": None,
            "prompt_tokens": 0, "completion_tokens": 0,
            "sdk_fingerprint": None,
            "error": f"{type(e).__name__}: {e}",
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }
```

#### 为什么用 JSONL 而不是 CSV

结果一律写 **JSONL**（每行一个 JSON 对象），理由：

- 可以**边跑边追加**，中途崩了已跑的不丢
- 天然支持嵌套字段（`params` 是个 dict）
- 模型输出里常有逗号、换行、引号，CSV 转义容易出事
- pandas 一行 `pd.read_json(path, lines=True)` 直接读

```python
def append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

#### D1–D2 练习

1. 准备 `data/questions.csv`，两列：`case_id,question`，先放 5 条
2. 跑通单次调用，结果写入 `results/{run_id}.jsonl`
3. 把 `max_tokens` 故意设成 20，观察 `finish_reason` 变成 `length`

---

### 模块 2（D3–D4）：采样参数

#### 参数含义

模型每一步输出前，会给词表里每个 token 一个分数（logit）。采样参数决定**怎么从这些分数里挑一个 token**。

| 参数 | 作用 | 评测时的建议 |
|---|---|---|
| `temperature` | 对 logits 做缩放：`softmax(logits / T)`。T 越小分布越尖锐，越接近贪心 | 评测默认 0，除非在测创造性任务 |
| `top_p` | 核采样：只从累积概率达到 p 的最小 token 集合里采 | 与 temperature 二选一调，别同时乱调 |
| `top_k` | 只从概率最高的 k 个 token 里采 | 部分服务才支持 |
| `max_tokens` | 输出上限 | **设足够大**，否则截断污染评测结果 |
| `seed` | 请求随机种子 | 仅 best-effort，不保证复现（见下） |
| `stop` | 遇到指定字符串就停 | 结构化输出时有用 |
| `frequency_penalty` / `presence_penalty` | 抑制重复 | 评测中一般保持默认 0 |

**一个常见误区**：很多人以为 temperature 和 top_p 是叠加生效的两个"随机度旋钮"，于是同时设成 0.7/0.9，结果无法解释行为。实际它们作用在采样链路的不同环节，同时调会让变量不可控。**评测里保持一个变，另一个用默认值。**

#### 核心实验：温度对输出分布的影响

```python
import collections

QUESTION = "用一个词形容今天的天气"  # 短输出，方便统计分布

for temp in [0.0, 0.3, 0.7, 1.0, 1.5]:
    outputs = []
    for _ in range(20):
        r = call_once(QUESTION, temperature=temp, max_tokens=20)
        outputs.append((r["output"] or "").strip())
    counter = collections.Counter(outputs)
    print(f"T={temp}  唯一答案数={len(counter)}  最高频占比={counter.most_common(1)[0][1]/20:.0%}")
    print(f"  分布: {counter.most_common(5)}\n")
```

**你应该观察到**：温度升高，唯一答案数增加，最高频答案占比下降。但**即使 T=0，唯一答案数也不一定等于 1**——这就引出验收题。

#### 验收题：temperature=0 为什么也不能保证完全复现

这是阶段 0 唯一的必答题，也是很多 AI 评测岗的面试开场题。完整答案有五层：

**1. temperature=0 并不等于"无随机性"，只等于贪心解码**

T=0 时实现上通常直接取 argmax（很多服务内部会把 0 替换成一个极小值避免除零）。它消除的是**采样随机性**，不是**计算不确定性**。

**2. 浮点计算本身不可复现**

GPU 上的并行归约（如矩阵乘法的加法累加）**顺序不固定**，而浮点加法不满足结合律：`(a+b)+c ≠ a+(b+c)`。同一批权重、同一个输入，两次前向传播算出的 logits 可能在末几位有差异。

**3. batch 组成会影响你的结果**

推理服务会把多个用户的请求打包成 batch。batch 大小不同 → 框架选择的 CUDA kernel 不同 → 数值路径不同。也就是说，**同时有多少别的用户在请求，会影响你这次的 logits 末位**。MoE 架构更明显：专家路由可能受 batch 内其他 token 影响。

**4. 微小差异会被放大成完全不同的答案**

当 top-1 和 top-2 token 的 logit 极其接近时，末位的浮点差异足以让 argmax 翻转。而自回归生成是逐 token 展开的——**一个 token 翻了，后续整个序列就发散了**。这就是为什么有时两次结果不是"差一个词"，而是完全两段话。

**5. 服务端不受你控制**

- 模型版本静默更新（同一个模型名，后端权重换了）
- 负载均衡到不同型号的 GPU、不同推理后端版本
- 上下文缓存命中与否可能走不同代码路径
- `seed` 参数被明确定义为 best-effort，配合 `system_fingerprint` 只能**告诉你环境变没变**，不能保证结果不变

**对评测工作的直接推论**：

> 任何单次运行的评测分数都不可信。评测结论必须建立在多次重复的统计量上，并报告波动范围。

#### D3–D4 练习

1. 跑完上面的温度实验，记录成一张表
2. 用 T=0 跑同一个**长输出**问题 10 次（比如"解释一下什么是向量检索"），统计有几个唯一答案、答案长度分布
3. 用自己的话把上面五层原因写成一段 200 字的说明，存进笔记

---

### 模块 3（D5–D7）：并发、重试、断点续跑

#### 为什么必须做并发

100 问题 × 5 重复 = 500 次调用，串行每次 3 秒就是 25 分钟。后面评测集到 200 条、再加 judge 调用，串行完全没法迭代。

#### 四件必须做对的事

| 事项 | 不做的后果 |
|---|---|
| **并发上限（Semaphore）** | 打爆速率限制，大量 429 |
| **指数退避重试** | 429 之后立刻重试 → 雪崩，越重试越限流 |
| **断点续跑** | 跑到 80% 崩了，全部重来 |
| **失败也落盘** | 失败被当成"空答案"算进分数，指标虚低且找不到原因 |

#### 完整 runner

```python
# scripts/runner.py
import asyncio, json, os, random, time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
client = AsyncOpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],
    timeout=60.0,
)

MODEL = os.environ["LLM_MODEL"]
PROMPT_VERSION = "v1"
PROMPT_TEMPLATE = "你是一个客服助手，请简洁准确地回答用户问题。\n\n用户问题：{question}"

MAX_CONCURRENCY = 8
MAX_RETRIES = 4
REPEATS = 5
PARAMS = {"temperature": 0.0, "max_tokens": 1024}


async def call_with_retry(question: str) -> dict:
    """带指数退避的单次调用。"""
    last_err = None
    for attempt in range(MAX_RETRIES):
        t0 = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user",
                           "content": PROMPT_TEMPLATE.format(question=question)}],
                **PARAMS,
            )
            choice = resp.choices[0]
            return {
                "output": choice.message.content,
                "finish_reason": choice.finish_reason,
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "sdk_fingerprint": getattr(resp, "system_fingerprint", None),
                "error": None,
                "attempts": attempt + 1,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            # 指数退避 + 抖动，避免所有失败请求同时重试
            backoff = (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(backoff)

    return {
        "output": None, "finish_reason": None,
        "prompt_tokens": 0, "completion_tokens": 0, "sdk_fingerprint": None,
        "error": last_err, "attempts": MAX_RETRIES, "latency_ms": 0,
    }


async def run_one(sem, out_path, run_id, case_id, question, repeat_idx):
    async with sem:
        result = await call_with_retry(question)
        record = {
            "run_id": run_id,
            "case_id": case_id,
            "repeat_idx": repeat_idx,
            "model": MODEL,
            "params": PARAMS,
            "prompt_version": PROMPT_VERSION,
            "question": question,
            "ts": time.time(),
            **result,
        }
        # 逐条落盘：中途崩了，已完成的不丢
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def load_done_keys(out_path: Path) -> set:
    """读取已完成的 (case_id, repeat_idx)，支持断点续跑。"""
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("error") is None:      # 失败的允许重跑
                    done.add((r["case_id"], r["repeat_idx"]))
            except json.JSONDecodeError:
                continue
    return done


async def main(run_id: str | None = None):
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(f"results/{run_id}.jsonl")
    out_path.parent.mkdir(exist_ok=True)

    df = pd.read_csv("data/questions.csv")
    done = load_done_keys(out_path)
    print(f"run_id={run_id}  已完成 {len(done)} 条")

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [
        run_one(sem, out_path, run_id, row.case_id, row.question, i)
        for row in df.itertuples()
        for i in range(REPEATS)
        if (row.case_id, i) not in done
    ]
    print(f"待跑 {len(tasks)} 条")

    t0 = time.perf_counter()
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0

    ok = sum(1 for r in results if r["error"] is None)
    pt = sum(r["prompt_tokens"] for r in results)
    ct = sum(r["completion_tokens"] for r in results)
    print(f"完成 {ok}/{len(results)}，耗时 {elapsed:.1f}s")
    print(f"token: 输入 {pt}，输出 {ct}")


if __name__ == "__main__":
    asyncio.run(main())
```

#### 并发数怎么定

不要拍脑袋设 50。做一次**阶梯测试**：并发从 2 → 4 → 8 → 16 → 32，每档跑 50 次调用，记录总耗时和错误率。

你会看到一条典型曲线：耗时先下降，到某个点后**错误率陡增而耗时不再下降**——那个拐点前一档就是你的工作并发数。

这个测试对你来说应该很熟悉，本质就是压测找拐点。

#### D5–D7 练习

1. 把 `questions.csv` 扩到 100 条（可以先从产品文档 FAQ 里抄）
2. 跑完整批次，中途 `Ctrl+C` 一次，重新运行，确认断点续跑生效
3. 做并发阶梯测试，确定你的工作并发数
4. 在脚本里加上成本估算（按你的服务定价）

---

## 五、第 2 周

### 模块 4（D8–D10）：pandas 核心操作

只需要掌握评测分析用得上的那部分，不用系统学 pandas。

#### 必须会的 8 个操作

```python
import pandas as pd

df = pd.read_json("results/20260915-143022.jsonl", lines=True)

# 1. 看整体形状与缺失
df.shape
df.info()
df["error"].notna().sum()          # 失败条数

# 2. 筛选
failed = df[df["error"].notna()]
truncated = df[df["finish_reason"] == "length"]   # 被截断的，必须单独看

# 3. 分组统计 —— 评测分析的主力
df.groupby("case_id")["completion_tokens"].agg(["mean", "std", "min", "max"])

# 4. 唯一值计数 —— 看变异性
df.groupby("case_id")["output"].nunique().value_counts().sort_index()

# 5. 频次分布
df["finish_reason"].value_counts(dropna=False)

# 6. 新增派生列
df["output_len"] = df["output"].fillna("").str.len()

# 7. 透视表 —— 多维度对比（阶段 4 做 A/B 时天天用）
df.pivot_table(index="case_id", columns="repeat_idx",
               values="output_len", aggfunc="first")

# 8. 合并 —— 把结果和标准答案对齐
gold = pd.read_csv("data/questions.csv")
merged = df.merge(gold, on="case_id", how="left", suffixes=("", "_gold"))
```

#### 一个容易踩的坑

`params` 字段是 dict，`read_json` 读进来是 object 列，不能直接 groupby。展开成列：

```python
params_df = pd.json_normalize(df["params"])
df = pd.concat([df.drop(columns=["params"]), params_df.add_prefix("param_")], axis=1)
# 现在有 param_temperature、param_max_tokens 列，可以直接 groupby
```

---

### 模块 5（D11–D12）：输出变异性分析

这是阶段 0 的核心分析任务：**量化"同一个问题跑 5 次到底有多不一样"**。

```python
# scripts/analyze.py
import difflib
import re
import pandas as pd


def normalize(text: str) -> str:
    """去掉不影响语义的差异，避免把格式波动当成内容波动。"""
    if not isinstance(text, str):
        return ""
    t = text.strip().lower()
    t = re.sub(r"\s+", "", t)                  # 去所有空白
    t = re.sub(r"[，。！？、；：,.!?;:]", "", t)  # 去标点
    return t


def analyze(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df = df[df["error"].isna()].copy()
    df["norm"] = df["output"].apply(normalize)
    df["output_len"] = df["output"].fillna("").str.len()

    rows = []
    for case_id, g in df.groupby("case_id"):
        outs = g["output"].fillna("").tolist()
        # 两两相似度的均值：衡量"有多接近"，而非只看是否完全相同
        sims = [
            difflib.SequenceMatcher(None, outs[i], outs[j]).ratio()
            for i in range(len(outs)) for j in range(i + 1, len(outs))
        ]
        mean_len = g["output_len"].mean()
        rows.append({
            "case_id": case_id,
            "n": len(g),
            "unique_raw": g["output"].nunique(),        # 原始唯一数
            "unique_norm": g["norm"].nunique(),         # 归一化后唯一数
            "mean_similarity": sum(sims) / len(sims) if sims else 1.0,
            "len_mean": round(mean_len, 1),
            "len_cv": round(g["output_len"].std() / mean_len, 3) if mean_len else 0,
            "truncated": (g["finish_reason"] == "length").sum(),
        })

    return pd.DataFrame(rows).sort_values("mean_similarity")


if __name__ == "__main__":
    res = analyze("results/20260915-143022.jsonl")
    print(res.head(20))            # 相似度最低的 20 条 = 最不稳定的问题
    print("\n=== 汇总 ===")
    print(f"完全一致的问题占比: {(res['unique_raw'] == 1).mean():.1%}")
    print(f"归一化后一致占比:   {(res['unique_norm'] == 1).mean():.1%}")
    print(f"平均相似度:         {res['mean_similarity'].mean():.3f}")
    print(f"存在截断的问题数:   {(res['truncated'] > 0).sum()}")
```

#### 怎么解读结果

| 现象 | 含义 | 后续动作 |
|---|---|---|
| 原始唯一数 > 1 但归一化后 = 1 | 只是空白/标点波动 | 评测时先归一化，别当成内容差异 |
| 平均相似度高（>0.9）但唯一数 > 1 | 措辞微调，语义可能一致 | 说明**不能用精确匹配打分**，必须用语义指标或 judge |
| 平均相似度低（<0.6） | 答案实质性发散 | 这类问题本身可能有歧义，或模型没把握 —— **它们是阶段 2 评测集的黄金素材** |
| `truncated > 0` | max_tokens 不够 | 调大后重跑，否则后续所有分数都被污染 |

#### 附加实验：重复几次才够

把 `REPEATS` 从 5 调到 10，跑一批，然后对每个问题计算：前 3 次、前 5 次、前 10 次的平均长度。观察什么时候这个均值开始稳定。

这个直觉在阶段 4 决定"每轮 A/B 要跑几次"时直接复用。

---

### 模块 6（D13–D14）：整合与交付

把两周的东西收成一份**跑批报告**，这是阶段 0 的交付物：

```markdown
# 阶段 0 跑批报告

## 环境
- 模型 / 服务商 / 参数 / prompt 版本
- 数据：100 条问题，来源说明

## 运行指标
- 总调用数、成功率、失败原因分布
- 工作并发数（附阶梯测试数据）
- 平均延迟 / P95 延迟
- token 消耗与成本

## 输出稳定性
- 完全一致率、归一化一致率、平均相似度
- 长度变异系数分布
- 最不稳定的 10 个问题及初步原因猜测

## 温度实验
- T ∈ {0, 0.3, 0.7, 1.0, 1.5} 的唯一答案数曲线

## 结论
- temperature=0 不可完全复现的原因（自己的话，200 字）
- 对后续评测设计的启示（至少 3 条）
```

---

## 六、验收自测

能答出以下六题，阶段 0 就算过关。

**1. temperature=0 为什么也不能保证完全复现？**
见前文五层答案：贪心≠无随机、浮点非结合律、batch 组成影响 kernel 选择、argmax 翻转后自回归放大、服务端版本与硬件不受控。

**2. `finish_reason == "length"` 为什么必须单独统计？**
表示输出被 max_tokens 截断。如果混在一起算分，会把"没说完"误判成"说错了"，导致指标虚低且归因错误。

**3. 为什么结果要用 JSONL 而不是 CSV？**
可追加、支持嵌套字段、避免模型输出中逗号引号换行的转义问题、pandas 可直接读。

**4. 为什么重试要用指数退避而不是固定间隔？**
固定间隔重试会让所有失败请求在同一时刻再次涌向服务端，加剧限流形成雪崩。退避加随机抖动可以把重试请求在时间上打散。

**5. 一个问题跑 5 次，原始唯一答案数是 4，但归一化后是 1，说明什么？**
差异只在空白和标点，语义完全一致。说明**打分前必须做归一化**，也说明这个 case 并不是真正的不稳定案例。

**6. 你的评测记录里为什么要存 `prompt_version` 和 `model`？**
评测分数只有在"prompt + 模型 + 参数 + 数据集"四者确定时才有意义。缺任意一项，事后无法解释分数变化来自哪里，阶段 4 的回归和 A/B 就做不了。

---

## 七、常见坑清单

| 坑 | 表现 | 防范 |
|---|---|---|
| 只存答案不存元数据 | 一周后不知道这批结果是什么配置跑的 | 按 schema 落盘 |
| 失败当空答案 | 指标莫名偏低，查不出原因 | `error` 字段单独统计，失败不计入分子分母 |
| max_tokens 设太小 | 大量截断被当成质量差 | 先跑一批看长度分布再定 |
| 不落盘边跑边攒内存 | 跑到 90% 崩溃全部重来 | 逐条 append 到 jsonl |
| 并发拍脑袋设 50 | 429 满屏 | 阶梯测试找拐点 |
| temperature 和 top_p 同时乱调 | 行为不可解释 | 评测固定一个，只变另一个 |
| 单次运行下结论 | 把噪声当成效果 | 至少重复 3–5 次，报告波动 |
| 用精确匹配当通过标准 | 语义正确但措辞不同被判失败 | 阶段 0 就该看到这个现象 |

---

## 八、延伸阅读

- 你所用服务商的 API 文档，重点看：采样参数说明、速率限制与错误码、`seed` 与 `system_fingerprint` 的语义
- `asyncio` 官方文档中的 Semaphore 与 `gather` 部分
- pandas 官方 10 Minutes to pandas（只看 groupby 和 merge 两节即可）
- 检索关键词：`LLM inference non-determinism`、`batch invariance`，能找到关于浮点与 batch 影响的工程讨论

---

## 九、进入阶段 1 的标志

- [ ] `runner.py` 能稳定跑 500 次调用，支持断点续跑，失败有记录
- [ ] 每条结果都带模型、参数、prompt 版本、重复序号
- [ ] 能用 pandas 在 10 分钟内出一份变异性分析
- [ ] 能向别人讲清楚 temperature=0 不可复现的原因
- [ ] 手上有一份阶段 0 跑批报告

全部打勾，就可以开始阶段 1：搭 RAG 并跑第一套评测。
