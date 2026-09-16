# 阶段 1：第一套 RAG 评测 —— 详细学习资料

> 周期：3 周 | 目标：搭出一个**可观测**的 RAG，写 50 条带标准答案的评测集，跑通自动评测，并能把分数归因到具体环节

---

## 一、本阶段的核心认知

新手做 RAG 评测最典型的失败：跑出一个 0.72 的总分，然后不知道该跟谁说。

**RAG 是一条流水线，不是一个模型。** 一次回答质量差，可能出在：

```
文档质量 → 切块 → embedding → 向量检索 → 重排 → prompt 拼装 → 生成
```

任何一环。你的核心价值是**把分数拆到环上**，告诉算法同学"是检索漏了，不是模型笨"。

由此推出本阶段第一条铁律：

> **RAG 必须是可观测的。** 如果你的 pipeline 只返回一个 answer 字符串，那它没法评测。中间产物（检索到的 chunk、相似度分数、重排后的顺序、最终 prompt）全部要暴露出来。

第二条铁律：

> **检索侧优先用确定性指标，不要什么都用 LLM 打分。**
> 检索有标准答案（该命中哪个 chunk），可以直接算 Recall@k、MRR、NDCG，便宜、快、稳定、可复现。只有生成侧的"有没有幻觉""答没答到点上"才必须用 LLM judge。
> 一上来全套 Ragas 的人，花了十倍的钱得到更不稳定的结论。

---

## 二、三周日程

| 周 | 模块 | 产出 |
|---|---|---|
| W1 | 搭可观测 RAG | pipeline 能返回全链路中间产物 |
| W2 | 手写 50 条评测集 | 带参考答案 + 应命中 chunk 的黄金集 v0 |
| W3 | 检索指标 + Ragas + 归因 | 分环节评测报告 + 一次消融实验 |

---

## 三、第 1 周：搭一个可观测的 RAG

### 3.1 需要理解的六个环节

| 环节 | 关键决策 | 会引入什么错误 |
|---|---|---|
| **文档预处理** | 格式转换、去页眉页脚、保留标题层级 | 表格被打碎、代码块被截断 |
| **切块 chunking** | 大小、重叠、切分依据 | 一个完整答案被切成两半，谁都不完整 |
| **embedding** | 模型选择、是否区分 query/passage 前缀 | 中文语义相似度不准、长文本被截断 |
| **向量检索** | top_k、是否混合检索（向量 + BM25） | 关键词型 query 向量检索失效（如订单号、错误码） |
| **重排 rerank** | 是否启用、重排模型、重排后保留几条 | 不用重排时，相关 chunk 排在第 8 位进不了 prompt |
| **生成** | prompt 模板、上下文顺序、是否要求引用 | 幻觉、答非所问、无视上下文 |

**中文场景两个高频坑，提前知道**：

1. **向量检索对精确标识符很弱**。用户问"错误码 E4021 是什么意思"，语义检索可能召回一堆讲错误处理的通用段落。解法是混合检索（向量 + BM25/关键词）。这类 case 一定要放进评测集。
2. **切块切碎表格和列表**。文档里"退款政策"是一张表，按固定字数切会把表头和内容分开，检索到的 chunk 看起来相关但信息不全。

### 3.2 可观测 pipeline 的返回结构

不管你用 LangChain、LlamaIndex 还是手写，**输出必须长这样**：

```python
from dataclasses import dataclass, field

@dataclass
class RetrievedChunk:
    chunk_id: str          # 稳定 ID，评测集里的 gold chunk 靠它对齐
    doc_id: str
    text: str
    score: float           # 检索得分
    rank: int              # 检索后的排名
    rerank_score: float | None = None
    rerank_rank: int | None = None

@dataclass
class RAGTrace:
    # 输入
    question: str
    # 中间产物 —— 评测的全部依据在这里
    rewritten_query: str | None      # 若做了 query 改写
    retrieved: list[RetrievedChunk]  # 检索原始结果（重排前）
    used_chunks: list[str]           # 最终进入 prompt 的 chunk_id
    final_prompt: str
    # 输出
    answer: str
    # 运行时
    retrieve_ms: int
    generate_ms: int
    prompt_tokens: int
    completion_tokens: int
    # 配置指纹（沿用阶段 0 的习惯）
    config: dict = field(default_factory=dict)
```

`config` 至少包含：`chunk_size`、`chunk_overlap`、`embed_model`、`top_k`、`rerank_model`、`rerank_top_n`、`llm_model`、`prompt_version`。

**这个 config 字段就是阶段 4 做 A/B 的基础**，现在偷懒不存，三个月后要重跑全部实验。

### 3.3 最简可观测实现

```python
# rag/pipeline.py
import time
import chromadb
from chromadb.utils import embedding_functions

class ObservableRAG:
    def __init__(self, collection, llm_client, config: dict):
        self.collection = collection
        self.llm = llm_client
        self.config = config

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        res = self.collection.query(
            query_texts=[question],
            n_results=self.config["top_k"],
        )
        return [
            RetrievedChunk(
                chunk_id=cid,
                doc_id=meta.get("doc_id", ""),
                text=doc,
                score=1 - dist,          # chroma 返回距离，转成相似度
                rank=i,
            )
            for i, (cid, doc, meta, dist) in enumerate(zip(
                res["ids"][0], res["documents"][0],
                res["metadatas"][0], res["distances"][0],
            ))
        ]

    def build_prompt(self, question: str, chunks: list[RetrievedChunk]) -> str:
        context = "\n\n".join(
            f"[文档{i+1}] {c.text}" for i, c in enumerate(chunks)
        )
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def run(self, question: str) -> RAGTrace:
        t0 = time.perf_counter()
        retrieved = self.retrieve(question)
        t1 = time.perf_counter()

        used = retrieved[: self.config.get("context_n", len(retrieved))]
        prompt = self.build_prompt(question, used)

        resp = self.llm.chat.completions.create(
            model=self.config["llm_model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        t2 = time.perf_counter()

        return RAGTrace(
            question=question,
            rewritten_query=None,
            retrieved=retrieved,
            used_chunks=[c.chunk_id for c in used],
            final_prompt=prompt,
            answer=resp.choices[0].message.content,
            retrieve_ms=int((t1 - t0) * 1000),
            generate_ms=int((t2 - t1) * 1000),
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            config=self.config,
        )
```

### 3.4 prompt 模板的一个关键设计

评测友好的 prompt 应该**强制模型标注引用来源**：

```python
PROMPT_TEMPLATE = """你是一个客服助手。请**仅根据下面提供的文档**回答用户问题。

规则：
1. 如果文档中没有足够信息回答，直接回复"文档中未找到相关信息"，不要编造。
2. 回答中涉及的每个事实，都要在句末标注来源，格式如 [文档1]。
3. 回答要简洁，不要复述问题。

文档：
{context}

用户问题：{question}

回答："""
```

好处有三个：

- 引用标注可以**程序化提取**，直接算"引用是否指向真正包含答案的 chunk"，不需要 LLM judge
- 强制拒答规则，让"无答案问题"变成可测的行为
- 后面阶段 2 做 judge 时，引用是很好的中间信号

### 3.5 W1 练习

1. 准备 30–50 份真实文档（帮助文档、API 文档、工单归档皆可）
2. 实现切块，**给每个 chunk 一个稳定 ID**（建议 `{doc_id}#{chunk_index}`，不要用随机 UUID，重建索引后要能对上）
3. 跑通 pipeline，把 `RAGTrace` 序列化成 JSONL
4. 人工翻看 10 条 trace，回答：检索到的 chunk 里有几条是真正相关的？

---

## 四、第 2 周：手写 50 条评测集

### 4.1 为什么必须手工写

这一周会很枯燥，但它的价值不在评测集本身，在于**你会被迫回答"什么叫答对了"**。

用模型批量生成问题，你得到的是"文档里已经写得很清楚的问题"——模型只会从文档反推问题，天然全都能答对，评出来分数虚高且没有区分度。

### 4.2 评测集 schema

```jsonl
{
  "case_id": "refund_001",
  "question": "订单已经发货了还能退款吗",
  "question_type": "single_hop",
  "difficulty": "easy",
  "answerable": true,
  "gold_answer": "已发货订单可以申请退款，但需要先拒收或寄回商品，退款在商品签收后 3 个工作日内原路退回。",
  "gold_chunk_ids": ["refund_policy#2", "refund_policy#3"],
  "gold_facts": ["需要先拒收或寄回", "3个工作日", "原路退回"],
  "source": "线上真实query",
  "note": "用户常混淆发货前后的流程差异"
}
```

三个字段值得单独说：

- **`gold_chunk_ids`** —— 检索评测的标准答案。没有它就只能靠 LLM 判断检索质量，又慢又贵又不稳。
- **`gold_facts`** —— 把参考答案拆成原子事实点。用于算**覆盖率**（答案命中了几个要点），比整体打分更细粒度、也更容易和人工对齐。
- **`answerable`** —— 标记这题文档里到底有没有答案。

### 4.3 50 条怎么分配

不要 50 条全是"文档里写得清楚的常见问题"。建议分布：

| 类型 | 条数 | 考什么 |
|---|---|---|
| 单跳事实问题 | 15 | 基础能力，保证不倒退 |
| 多跳问题（答案跨 2+ 文档） | 8 | 检索召回与整合 |
| **无答案问题** | 8 | 会不会编 —— **最重要的一类** |
| 精确标识符问题（订单号、错误码、版本号） | 5 | 向量检索的已知弱项 |
| 口语化 / 有错别字 | 5 | 真实用户的样子 |
| 歧义问题 | 4 | 会不会反问澄清，还是瞎猜一个 |
| 前提错误问题 | 3 | 会不会顺着错误前提编 |
| 超范围问题（问天气、闲聊） | 2 | 边界行为 |

**无答案问题怎么构造**：找文档里"沾边但没写"的东西。比如文档写了退款流程，没写退款失败怎么办——就问"退款失败了怎么处理"。这类问题比问"今天几号"有价值得多，因为它会检索到高相似度但无答案的 chunk，正是幻觉高发区。

**前提错误问题的例子**：文档里写退款周期是 7 天，你问"退款要 30 天这么久是为什么"。理想行为是纠正前提，最差行为是顺着编一套理由。

### 4.4 工作方法

一天写 10–15 条，分三步：

1. 先从**线上真实 query 日志**里抄（有的话）。没有就从同事、客服那里要
2. 对每条问题，自己去文档里找答案，记录 `gold_chunk_ids`
3. 写 `gold_answer` 和 `gold_facts`

**在这个过程中你会发现文档本身的问题**——写得矛盾、过时、缺失。把这些记下来，这是你给团队的第一份额外价值。

### 4.5 W2 练习

1. 完成 50 条评测集，存成 `data/eval_set_v0.jsonl`
2. 做一个**自检**：随机抽 10 条，隔天重新判断一次自己的 `gold_answer`，看有几条你自己都改了主意 —— 改了 3 条以上，说明标准不清晰，阶段 2 的 rubric 会很难写
3. 统计文档问题清单

---

## 五、第 3 周：分环节评测

### 5.1 先评检索（不用 LLM，便宜又可靠）

```python
# eval/retrieval_metrics.py
import numpy as np


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """前 k 条里，命中了多少比例的 gold chunk。"""
    if not gold_ids:
        return float("nan")     # 无答案问题不算检索召回
    topk = set(retrieved_ids[:k])
    return len(topk & set(gold_ids)) / len(gold_ids)


def hit_rate_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """前 k 条里，是否至少命中一条 gold chunk。"""
    if not gold_ids:
        return float("nan")
    return float(bool(set(retrieved_ids[:k]) & set(gold_ids)))


def mrr(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """第一条正确结果排名的倒数 —— 衡量排序质量。"""
    gold = set(gold_ids)
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """考虑位置折损的排序质量，gold chunk 排得越靠前越高。"""
    gold = set(gold_ids)
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, cid in enumerate(retrieved_ids[:k]) if cid in gold
    )
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0
```

**怎么用这些指标**：

| 现象 | 诊断 |
|---|---|
| Recall@20 高，Recall@5 低 | 相关内容能找到但**排序差** → 加重排 |
| Recall@20 也低 | **切块或 embedding 问题** → 调 chunk_size、换 embedding、加 BM25 混合检索 |
| MRR 低但 Hit Rate 高 | 排序差，同上 |
| 精确标识符类问题 Recall 显著低于整体 | 向量检索弱项确认 → 必须上混合检索 |

**先修检索再谈生成。** 检索都没召回，生成环节再优化也是无源之水。

### 5.2 再评生成

生成侧分两层，从便宜到贵：

**第一层：确定性检查（不花钱）**

```python
import re

def citation_precision(answer: str, used_chunks: list[str],
                       gold_chunk_ids: list[str]) -> float:
    """答案里标注的引用，有多少指向真正包含答案的 chunk。"""
    cited_idx = [int(m) - 1 for m in re.findall(r"\[文档(\d+)\]", answer)]
    cited_ids = [used_chunks[i] for i in cited_idx if 0 <= i < len(used_chunks)]
    if not cited_ids:
        return float("nan")
    gold = set(gold_chunk_ids)
    return sum(1 for cid in cited_ids if cid in gold) / len(cited_ids)


def fact_coverage(answer: str, gold_facts: list[str]) -> float:
    """要点覆盖率的字面版本，作为粗筛。语义版本用 embedding 或 judge。"""
    if not gold_facts:
        return float("nan")
    hit = sum(1 for f in gold_facts if f in answer)
    return hit / len(gold_facts)


REFUSAL_PATTERNS = ["未找到相关信息", "无法回答", "文档中没有"]

def is_refusal(answer: str) -> bool:
    return any(p in answer for p in REFUSAL_PATTERNS)
```

有了 `is_refusal`，**无答案问题就变成一个二分类指标**：

- **正确拒答率** = 无答案问题中正确拒答的比例
- **过度拒答率** = 有答案问题中却拒答的比例（这个同样重要，很多团队只优化前者，把系统调成什么都不敢答）

这两个数字放一起看，比单一的"幻觉率"有信息量得多。

**第二层：Ragas 四指标**

| 指标 | 计算原理 | 评的是哪一环 |
|---|---|---|
| **faithfulness** 忠实度 | 把答案拆成若干原子断言，逐条判断能否由 context 推出，算支持比例 | **生成**：有没有超出检索内容编造 |
| **answer_relevancy** 答案相关性 | 由答案反向生成若干问题，与原问题算 embedding 相似度取均值 | **生成**：有没有答非所问 |
| **context_precision** 上下文精确率 | 检索结果中相关 chunk 是否排在前面（类似 MAP） | **检索/重排** |
| **context_recall** 上下文召回率 | 参考答案的每个句子能否由 context 支撑 | **检索/切块** |

注意：`context_precision` 和 `context_recall` 与你上面手算的 IR 指标**功能重叠**，但它靠 LLM 判断语义相关，你的靠 `gold_chunk_ids` 精确匹配。

**建议**：主用你自己的 IR 指标（便宜、可复现），Ragas 的上下文指标只在没有 gold chunk 标注的 case 上用作补充。Ragas 的价值主要在 `faithfulness`。

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

ds = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,           # list[list[str]]
    "ground_truth": gold_answers,
})
result = evaluate(ds, metrics=[faithfulness, answer_relevancy])
print(result)
```

**跑之前先看成本**：faithfulness 对每条 case 要调好几次 LLM（拆断言 + 逐条判断）。50 条 case 可能产生几百次调用。先在 5 条上试跑，估算单条成本再放开。

### 5.3 归因矩阵（本阶段核心交付物）

把指标组合成诊断结论：

| Recall@k | faithfulness | 拒答行为 | 诊断 | 找谁 |
|---|---|---|---|---|
| 低 | — | 正确拒答 | 检索没召回，但模型诚实 | 检索/切块负责人 |
| 低 | 低 | 强答 | 检索没召回 + 模型编造 | 两边都要修，**优先修 prompt 的拒答约束** |
| 高 | 低 | 强答 | 内容找到了，生成环节幻觉 | prompt / 模型 |
| 高 | 高 | — | 但 answer_relevancy 低 → 答非所问 | prompt 结构 |
| 高 | 高 | 过度拒答 | 拒答规则太严 | prompt 调松 |
| 高 | 高 | 正常 | 若人工仍觉得不好 → **可能是 gold_answer 或问题本身有问题** | 回头改评测集 |

最后一行很重要：**当所有指标都好但人还是不满意时，先怀疑评测集**。这是进入阶段 2 的动机。

### 5.4 消融实验

做一次单变量对照，这是阶段 4 A/B 实验的预演：

| 实验 | 变量 | 观察指标 |
|---|---|---|
| 切块大小 | 256 / 512 / 1024 tokens | Recall@5、faithfulness |
| 重叠 | 0 / 10% / 20% | Recall@5 |
| top_k | 3 / 5 / 10 / 20 | Recall@k、成本、faithfulness（context 变长会不会更容易幻觉） |
| 重排 | 开 / 关 | MRR、NDCG@5 |
| 混合检索 | 纯向量 / 向量+BM25 | 精确标识符子集的 Recall |

**每次只改一个变量**，其余 config 固定。结果记成一张表。

注意一个反直觉现象：**top_k 调大，Recall 上升但 faithfulness 可能下降**——上下文里塞了更多不相关内容，模型更容易被带偏。这个 trade-off 是很好的报告素材。

### 5.5 W3 练习

1. 实现检索指标，跑出 Recall@5/@10/@20、MRR、NDCG
2. 实现拒答检测，算出正确拒答率与过度拒答率
3. 跑一次 Ragas faithfulness，记录成本
4. 完成归因矩阵，挑 5 条低分 case 做人工深挖，写出根因
5. 完成至少 2 组消融实验

---

## 六、验收自测

**1. 为什么检索侧不该全用 LLM judge？**
检索有明确的 gold chunk 标注，可以用 Recall/MRR/NDCG 这类确定性指标，便宜、快、完全可复现。LLM judge 慢、贵、本身带噪声，应该留给没有标准答案的语义判断。

**2. Recall@20 高但 Recall@5 低，说明什么？该改哪里？**
相关内容能被检索到，但排序不好。应该加重排模型，或调整检索策略，而不是去改 prompt。

**3. 为什么"过度拒答率"和"正确拒答率"要一起看？**
只优化正确拒答率，会把系统调成什么都不敢答，用户体验崩塌。两个数字构成一组 trade-off，类似 precision/recall。

**4. top_k 从 5 调到 20，Recall 上升但 faithfulness 下降，怎么解释？**
更多上下文提高了召回，但也引入更多不相关内容，模型更容易被无关信息带偏产生幻觉，同时成本上升。需要在两者间取平衡，或引入重排只保留高质量的几条。

**5. 所有自动指标都达标，但人工评估仍觉得答得不好，第一步该做什么？**
先怀疑评测集：gold_answer 是否真的代表好答案、问题是否有歧义、指标是否覆盖了用户真正在意的维度（如语气、结构、可操作性）。

**6. 为什么 chunk_id 要用稳定 ID 而不是 UUID？**
评测集里的 `gold_chunk_ids` 要和索引对齐。用随机 UUID，每次重建索引后 ID 全变，评测集立刻作废。

---

## 七、常见坑

| 坑 | 后果 | 防范 |
|---|---|---|
| pipeline 只返回 answer | 无法归因，只能拿到一个总分 | 从第一天就返回完整 trace |
| chunk_id 不稳定 | 重建索引后评测集失效 | 用 `doc_id#index` 形式 |
| 评测集用模型批量生成 | 分数虚高、没有区分度 | 至少手写核心部分 |
| 评测集里没有无答案问题 | 测不出幻觉，上线才发现 | 占比不低于 15% |
| 没有精确标识符类 case | 向量检索弱项被完全掩盖 | 单独一组，单独看指标 |
| 一上来全套 Ragas | 成本高、结论不稳、还是不知道该改哪 | 先跑 IR 指标和确定性检查 |
| 消融实验同时改两个变量 | 结论不可解释 | 严格单变量 |
| 单次运行下结论 | 忽略采样噪声（阶段 0 已验证） | 至少重复 3 次 |

---

## 八、进入阶段 2 的标志

- [ ] pipeline 返回完整 trace，中间产物可分析
- [ ] 50 条评测集，含 gold_chunk_ids 和 gold_facts
- [ ] 检索指标、拒答指标、faithfulness 全部跑通
- [ ] 能拿着归因矩阵，对每个低分 case 说出应该找谁修
- [ ] 完成至少 2 组消融实验并有结论
- [ ] **感受到 50 条不够用**——覆盖不全、分数波动大、个别 case 的对错自己都拿不准

最后一条是最重要的信号：它正是阶段 2 存在的理由。
