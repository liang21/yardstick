# 阶段 4：评测流水线与回归 —— 详细学习资料

> 周期：3 周 | 目标：把前三阶段的脚本变成团队能持续用的工程能力——可追溯、进 CI、有统计学依据的 A/B

---

## 一、本阶段的核心认知

前三个阶段你做的是**一次性评测**。本阶段要解决的是：

> **"这次改动到底让系统变好了还是变坏了？"** —— 一个必须每天回答、且必须有统计学依据的问题。

三条铁律：

> **1. 分数没有来历就没有价值。** 说不清是哪个 prompt 版本、哪个模型、哪份评测集跑出来的 0.82，等于没有这个数。
>
> **2. AI 评测天然 flaky，阈值必须建立在历史波动之上。** 把传统测试"红了就是有 bug"的直觉直接搬过来，你会被噪声淹没。
>
> **3. 没有置信区间的 A/B 结论不是结论。** "B 比 A 高 4 个点"这句话本身不包含任何信息。

这是你作为资深测试最能发挥老本行的一个阶段。

---

## 二、三周日程

| 周 | 模块 | 产出 |
|---|---|---|
| W1 | Tracing + 版本追溯 + 实验管理 | 任一分数可追溯到完整配置 |
| W2 | CI 集成 + 分层触发 + 阈值设计 | PR 触发的冒烟 + 每晚全量 |
| W3 | A/B 实验与统计 | 一份带置信区间的评测报告 |

---

## 三、W1：可追溯性

### 3.1 四元版本追溯

一个评测分数必须能回答四个问题：

| 维度 | 记什么 | 怎么保证 |
|---|---|---|
| **代码** | git commit sha、是否 dirty | `git rev-parse HEAD` |
| **提示词** | prompt 模板版本 + 内容 hash | prompt 存成文件并纳入 git，同时记 hash |
| **模型与参数** | 模型名、temperature、max_tokens、服务端 fingerprint | 从阶段 0 就在记 |
| **数据集** | 评测集版本 + 内容 hash + 条数 | 阶段 2 的版本化 |

**prompt 必须当代码管理**（OWASP LLM01 的缓解措施之一也是这条）。写在 Python 字符串里、随手改了就跑，是最常见的不可追溯来源。

```python
# eval/provenance.py
import hashlib
import json
import subprocess
from pathlib import Path


def file_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def git_info() -> dict:
    def run(cmd):
        return subprocess.run(cmd, capture_output=True, text=True,
                              shell=True).stdout.strip()
    return {
        "commit": run("git rev-parse HEAD")[:12],
        "branch": run("git rev-parse --abbrev-ref HEAD"),
        "dirty": bool(run("git status --porcelain")),
    }


def build_provenance(prompt_path: str, dataset_path: str,
                     model_config: dict) -> dict:
    """每次跑批必须调用，结果写进 run 的元数据。"""
    return {
        "git": git_info(),
        "prompt": {"path": prompt_path, "hash": file_hash(prompt_path)},
        "dataset": {"path": dataset_path, "hash": file_hash(dataset_path)},
        "model": model_config,
        "judge": {"version": "v5", "model": "...", "prompt_hash": "..."},
    }
```

**别忘了 judge 也要版本化。** judge 换了一版，历史分数就不可比了——这是最隐蔽的一类陷阱：系统没变，分数掉了，查半天发现是有人改了 judge prompt。

### 3.2 Tracing

Tracing 解决的是"**为什么这条 case 分数低**"。指标告诉你哪里有问题，trace 告诉你为什么。

OpenTelemetry 已经有 GenAI 的语义约定，Langfuse、Phoenix、LangSmith 等工具都基于它。选一个自托管友好的接入（Langfuse 和 Phoenix 都有开源核心）。

**span 层级建议**：

```
eval_run (root)
└── case:refund_012
    └── rag_pipeline
        ├── retrieve          [attrs: query, top_k, chunk_ids, scores]
        ├── rerank            [attrs: model, input_n, output_n]
        └── generate          [attrs: model, prompt, tokens, finish_reason]
    └── judge
        ├── dimension:correctness    [attrs: score, reasoning]
        └── dimension:faithfulness   [attrs: score, reasoning]
```

**关键做法：在 span 上打 `case_id` 和 `run_id` 标签**，这样从评测报告里的一行低分，能一键跳到完整调用链。没有这个关联，tracing 就只是个日志看板。

### 3.3 实验记录

每次跑批产出一条实验记录：

```jsonl
{
  "run_id": "20260920-101500",
  "name": "chunk512-vs-chunk256",
  "provenance": { ... },
  "dataset_version": "v1",
  "n_cases": 200,
  "repeats": 3,
  "metrics": {
    "recall@5": {"mean": 0.81, "std": 0.012},
    "faithfulness": {"mean": 3.42, "std": 0.09},
    "correct_refusal_rate": {"mean": 0.88, "std": 0.03},
    "over_refusal_rate": {"mean": 0.07, "std": 0.02}
  },
  "cost_cny": 12.4,
  "wall_seconds": 340,
  "note": "对照组 chunk_size=256"
}
```

存成 `experiments/*.jsonl`，纳入 git。三个月后你能画出所有指标的完整演进曲线。

### 3.4 W1 练习

1. 把 prompt 全部移到独立文件，纳入 git
2. 实现 `build_provenance`，接进 runner
3. 接入 Langfuse 或 Phoenix，打上 `case_id` / `run_id` 标签
4. 验证：从报告里挑一条低分 case，30 秒内能看到它的完整调用链
5. 补齐历史实验记录

---

## 四、W2：CI 集成

### 4.1 分层触发（测试金字塔的 AI 版）

| 层级 | 规模 | 触发时机 | 耗时/成本预算 | 卡点 |
|---|---|---|---|---|
| **冒烟** | 25–30 条核心 case，重复 1 次 | 每次 PR | < 3 分钟，< ¥2 | 硬卡，红了不许合 |
| **全量** | 200 条，重复 3 次 | 每晚定时 | < 30 分钟，< ¥50 | 告警，不阻塞 |
| **深度** | 全量 + Agent + 多轮 + 红队，重复 5 次 | 发版前 / 手动 | 不限 | 人工评审 |

**冒烟集怎么挑**（这是你的测试经验直接复用）：

- 每个意图类型至少 2 条
- 全部 `CRITICAL` 级 case（越权、编造关键事实）
- 历史上出过 bug 的回归 case
- 分数方差最小的那批（方差大的 case 在小样本下会假报警）

最后一条是 AI 评测特有的。**用阶段 0 学的方差分析，把不稳定的 case 排除出冒烟集。**

### 4.2 pytest 组织

```python
# tests/test_eval_smoke.py
import json
import pytest
from pathlib import Path

from eval.runner import run_eval
from eval.metrics import aggregate

BASELINE = json.loads(Path("eval/baseline.json").read_text())
SMOKE_SET = "data/eval_set_v1_smoke.jsonl"


@pytest.fixture(scope="session")
def results():
    """整个 session 只跑一次评测，各测试项共享结果。"""
    return run_eval(dataset=SMOKE_SET, repeats=1)


# ---------- 绝对阈值：不可逾越的底线 ----------

def test_no_critical_failures(results):
    critical = [r for r in results if r["grade"] == "CRITICAL_FAIL"]
    assert not critical, f"出现严重失败: {[c['case_id'] for c in critical]}"


def test_faithfulness_floor(results):
    m = aggregate(results)["faithfulness"]["mean"]
    assert m >= 3.0, f"忠实度 {m:.2f} 低于绝对底线 3.0"


def test_over_refusal_ceiling(results):
    """防止为了降幻觉把系统调成什么都不敢答。"""
    r = aggregate(results)["over_refusal_rate"]["mean"]
    assert r <= 0.15, f"过度拒答率 {r:.1%} 超过上限"


# ---------- 相对阈值：相对基线不许退化 ----------

@pytest.mark.parametrize("metric", ["recall@5", "faithfulness", "correct_refusal_rate"])
def test_no_regression(results, metric):
    cur = aggregate(results)[metric]["mean"]
    base = BASELINE[metric]["mean"]
    tol = BASELINE[metric]["tolerance"]     # 见 4.3
    assert cur >= base - tol, (
        f"{metric} 退化: {cur:.3f} vs 基线 {base:.3f}，容差 {tol:.3f}"
    )


# ---------- 逐 case 回归：核心 case 不许挂 ----------

GOLDEN_CASES = ["refund_001", "refund_012", "agent_007"]

@pytest.mark.parametrize("case_id", GOLDEN_CASES)
def test_golden_case(results, case_id):
    r = next(x for x in results if x["case_id"] == case_id)
    assert r["grade"].startswith("PASS"), f"{case_id} 失败: {r.get('reason')}"
```

### 4.3 阈值怎么定：不能拍脑袋

这是 AI 评测和传统测试差别最大的地方。**评测有噪声，阈值设太紧会天天误报，设太松则漏掉真实退化。**

正确做法：**用历史波动决定容差。**

```python
# eval/calibrate_threshold.py
import numpy as np
import pandas as pd


def compute_tolerance(history: pd.DataFrame, metric: str,
                      z: float = 2.0) -> dict:
    """
    history: 近 N 次「无代码改动」的基线跑批结果。
    容差 = z 倍的历史标准差，即只有超出正常波动才算退化。
    """
    vals = history[metric].dropna()
    mu, sigma = vals.mean(), vals.std(ddof=1)
    return {
        "mean": round(mu, 4),
        "std": round(sigma, 4),
        "tolerance": round(z * sigma, 4),
        "n_runs": len(vals),
        "note": f"基于 {len(vals)} 次基线跑批，{z}σ 容差",
    }
```

**操作步骤**：

1. 代码冻结，连续跑 10 次冒烟评测（可以分散在一天里，覆盖不同时段的服务负载）
2. 算每个指标的均值和标准差
3. 容差取 2σ（约 95% 置信），高风险指标可以用 1.5σ
4. 写进 `baseline.json`，**每次评测集或 judge 变更后重新标定**

**怎么判断阈值合理**：连续跑一周基线，误报次数应该接近 0。如果一周误报 3 次以上，说明容差太紧或该指标本身不稳定（考虑换指标或增加重复次数）。

### 4.4 降噪的三个手段

| 手段 | 效果 | 代价 |
|---|---|---|
| 增加重复次数 | 标准误按 1/√n 下降 | 成本线性上升 |
| 固定随机性（temperature=0、固定检索种子） | 减少但不能消除（见阶段 0） | 无 |
| **配对设计**（见 W3） | **显著提升灵敏度** | 无额外成本 |
| 剔除高方差 case | 冒烟集更稳 | 覆盖率下降 |

**重复次数怎么定**：用历史数据算。若单次标准差 σ，你想检测出 Δ 的变化，大致需要 n ≈ (2·z·σ/Δ)² 次重复。想检测 2% 的变化而 σ=3%，需要的 n 会很大——这时应该转向配对设计，而不是盲目加重复。

### 4.5 成本控制

CI 每天要跑几十次，成本会失控。四个手段：

1. **冒烟集只跑 25–30 条**，不要图覆盖全
2. **judge 用级联**（阶段 2.8）：确定性检查 → 小模型 → 大模型只跑边界 case
3. **缓存**：输入完全相同（prompt hash + 参数 hash）时复用结果，只在 prompt 或代码变动时重跑
4. **只在相关路径变动时触发**：改文档不触发 Agent 评测

在实验记录里持续跟踪 `cost_cny`，做一张成本趋势图——这是向上汇报时最有说服力的材料。

### 4.6 W2 练习

1. 挑出冒烟集（附带方差数据说明为什么挑这些）
2. 代码冻结跑 10 次基线，标定全部阈值
3. 写 pytest 测试，接进 CI（GitHub Actions / GitLab CI 均可）
4. 连续观察一周，统计误报次数，调整容差
5. 实现结果缓存，测算成本下降

---

## 五、W3：A/B 实验与统计

### 5.1 实验设计：一定要配对

**独立设计**（错误做法）：A 方案跑一批 case，B 方案跑另一批。case 之间的难度差异会淹没方案差异。

**配对设计**（正确做法）：**同一批 case，两个方案各跑一遍**，逐 case 对比差值。

配对设计消除了 case 难度这个最大的变异来源，**在同样样本量下灵敏度高得多**。这是免费的提升，没有理由不用。

```python
# 配对数据结构
# case_id | score_A | score_B | diff
# refund_001 |  3.2  |  3.6   | +0.4
# refund_002 |  2.8  |  2.9   | +0.1
# ...
```

### 5.2 统计检验怎么选

| 数据 | 检验方法 | 说明 |
|---|---|---|
| 配对连续分数 | **Bootstrap 置信区间**（首选） | 不依赖正态假设，直观 |
| 配对连续分数 | 配对 t 检验 | 需近似正态 |
| 配对有序/非正态 | **Wilcoxon 符号秩检验** | 稳健 |
| 配对二分类（通过/不通过） | **McNemar 检验** | 专门用于配对二分类 |
| Pairwise 判优劣 | 二项检验 / Bradley-Terry | 直接比较胜负 |

**推荐默认用 Bootstrap**，理由：不需要判断分布假设、直接给出置信区间（比 p 值更有信息量）、易于向非统计背景的同事解释。

```python
# eval/ab_test.py
import numpy as np
import pandas as pd
from scipy import stats


def paired_bootstrap(a: np.ndarray, b: np.ndarray,
                     n_boot: int = 10000, seed: int = 42) -> dict:
    """配对 bootstrap：对差值重采样，给出均值差的置信区间。"""
    rng = np.random.default_rng(seed)
    diff = b - a
    n = len(diff)
    boot = np.array([
        diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "mean_a": a.mean(),
        "mean_b": b.mean(),
        "mean_diff": diff.mean(),
        "ci_95": (lo, hi),
        "significant": not (lo <= 0 <= hi),      # 区间不含 0 即显著
        "n_pairs": n,
        # 效应量：差值相对于波动的大小，回答"有意义吗"
        "cohens_d": diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else 0,
    }


def paired_tests(a: np.ndarray, b: np.ndarray) -> dict:
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:      # 全部差值为 0
        w_stat, w_p = np.nan, 1.0
    return {"t_p": t_p, "wilcoxon_p": w_p}


def mcnemar(a_pass: np.ndarray, b_pass: np.ndarray) -> dict:
    """配对二分类：只看「A过B不过」和「A不过B过」两格。"""
    b01 = int(((a_pass == 1) & (b_pass == 0)).sum())   # A过B不过
    b10 = int(((a_pass == 0) & (b_pass == 1)).sum())   # A不过B过
    if b01 + b10 == 0:
        return {"b01": 0, "b10": 0, "p": 1.0}
    p = stats.binomtest(b10, b01 + b10, 0.5).pvalue
    return {"b01": b01, "b10": b10, "p": p,
            "net_gain": b10 - b01}
```

### 5.3 多重比较：一个容易被忽略的陷阱

你一次 A/B 会看很多指标：recall、faithfulness、拒答率、过度拒答率、可操作性……**看得越多，至少一个偶然显著的概率越高。**

看 10 个指标，每个用 0.05 的阈值，全部无效时仍有约 40% 概率至少一个"显著"。

**处理方式**：

```python
from statsmodels.stats.multitest import multipletests

p_values = [r["t_p"] for r in results_by_metric]
reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
```

- **预先声明主指标**（primary metric），只用它做决策，其余为参考——这是最干净的做法
- 或用 Benjamini-Hochberg 控制错误发现率

**在报告里明确写出主指标是什么**，防止事后挑一个好看的指标当结论（p-hacking）。

### 5.4 效应量：显著 ≠ 有意义

样本量够大时，0.3% 的提升也能显著。但它值得为此换模型、增加 30% 成本吗？

**报告里必须同时给出**：

- **统计显著性** —— 这个差异是不是噪声
- **效应量** —— 这个差异有多大（Cohen's d，或直接给绝对提升）
- **业务意义** —— 换算成业务语言：忠实度 +0.2 分大约等于每千次对话少 X 次编造
- **成本变化** —— B 方案贵了多少

四者齐全，才算一个可以拿去做决策的结论。

### 5.5 报告模板

```markdown
# A/B 实验报告：混合检索 vs 纯向量检索

## 实验设计
- 假设：加入 BM25 混合检索能提升精确标识符类问题的召回
- **主指标：精确标识符子集的 Recall@5**（决策依据）
- 次要指标：整体 Recall@5、faithfulness、P95 延迟、单次成本
- 设计：配对，同一批 200 条评测集（v1，hash a3f9c2），各跑 3 次取均值
- 控制变量：模型、prompt、chunk 配置、top_k 全部固定，仅检索策略不同
- provenance：commit 8f2a1c，prompt v3 (hash 7d21)，judge v5

## 结果

| 指标 | A（纯向量） | B（混合） | 差值 | 95% CI | 显著 |
|---|---|---|---|---|---|
| **精确标识符 Recall@5（主）** | 0.412 | 0.736 | +0.324 | [0.241, 0.408] | 是 |
| 整体 Recall@5 | 0.798 | 0.841 | +0.043 | [0.018, 0.069] | 是 |
| faithfulness | 3.41 | 3.44 | +0.03 | [-0.06, 0.12] | 否 |
| P95 延迟 | 820ms | 1150ms | +330ms | — | — |
| 单次成本 | ¥0.021 | ¥0.023 | +¥0.002 | — | — |

多重比较：3 个指标做 BH 校正，结论不变。

## 解读
- 主指标提升 32.4 个百分点，效应量 Cohen's d = 1.42（大），置信区间远离 0
- faithfulness 无显著变化，说明提升来自检索侧，未引入新的幻觉风险
- 代价是 P95 延迟增加 40%，成本增加约 10%

## 结论与建议
建议上线。主指标提升幅度远超延迟代价；若对延迟敏感，可只对含数字/编码模式的 query 走混合检索。

## 局限
- 评测集中精确标识符类仅 12 条，子集样本偏小，CI 较宽
- 未覆盖多语言 query
- 延迟数据来自测试环境，生产网络条件可能不同
```

**"局限"这一节一定要写。** 主动暴露不确定性，是评测专业性的标志，也是面试时的加分点。

### 5.6 W3 练习

1. 实现 bootstrap、配对检验、McNemar
2. 做一次完整 A/B（建议：混合检索 vs 纯向量，或详细版 vs 简略版工具描述）
3. 预先声明主指标，做多重比较校正
4. 按模板写出完整报告，含局限
5. 做一个反面练习：**故意用独立设计（非配对）重跑同样的数据，对比置信区间宽度**，亲眼看到配对设计的价值

---

## 六、验收自测

**1. 为什么容差不能拍脑袋定？怎么定？**
评测有固有噪声，容差必须建立在历史波动之上。做法是代码冻结连续跑 10 次基线，算标准差，容差取 2σ。评测集或 judge 变更后必须重新标定。

**2. judge 换了一版，历史分数还能比吗？**
不能。judge 是评测系统的一部分，换版后尺子变了。必须给 judge 也做版本管理，并在分数上标注 judge 版本。这是最隐蔽的一类不可比陷阱。

**3. A/B 为什么一定要用配对设计？**
配对消除了 case 难度这个最大的变异来源，同样样本量下灵敏度显著更高，且不增加任何成本。

**4. "B 比 A 高 4 个点"这句话缺什么？**
缺置信区间（这 4 个点是不是噪声）、缺效应量（有多大）、缺成本对比（值不值）、缺主指标声明（是不是事后挑的好看指标）。

**5. 一次 A/B 看了 10 个指标，有 1 个显著，能下结论吗？**
不能。多重比较会大幅抬高假阳性概率。应该预先声明主指标，或用 BH 等方法校正后再看。

**6. CI 里的评测天天误报怎么办？**
先确认不是真退化，然后按顺序检查：容差是否太紧（重新标定）、冒烟集里是否混入高方差 case（剔除）、重复次数是否不足（增加）。不要简单粗暴地把阈值一路放宽，那等于关掉了卡点。

**7. 为什么过度拒答率也要设卡点？**
只卡幻觉指标，最简单的"优化"方式就是让系统什么都不敢答。双向卡点才能防止这种退化。

---

## 七、常见坑

| 坑 | 后果 | 防范 |
|---|---|---|
| prompt 写在代码字符串里随手改 | 分数不可追溯 | prompt 当代码，独立文件 + hash |
| judge 不做版本管理 | 系统没变分数却掉了，查一整天 | judge prompt 也纳入 provenance |
| 阈值拍脑袋 | 天天误报，最后大家无视 CI | 用历史 2σ 标定 |
| 冒烟集追求覆盖全 | 跑得慢、成本高、噪声大 | 25–30 条，剔除高方差 case |
| 独立设计做 A/B | 灵敏度低，真实提升测不出来 | 一律配对 |
| 只报均值不报区间 | 把噪声当成效果，决策错误 | bootstrap CI 必须给 |
| 看了 10 个指标挑一个显著的 | p-hacking，结论不可信 | 预先声明主指标 |
| 显著就上线 | 忽略效应量和成本 | 四要素齐全才决策 |
| 修改评测集后直接对比历史分数 | 结论完全错误 | 分数绑定数据集版本，变更后重建基线 |
| 没有成本监控 | CI 账单失控被叫停 | 每次 run 记 cost，做趋势图 |

---

## 八、进入阶段 5 的标志

- [ ] 任意一个历史分数，能追溯到代码、prompt、模型、数据集、judge 五项版本
- [ ] 从报告中一条低分 case，能一键跳转到完整 trace
- [ ] 冒烟评测进 CI，PR 触发，连续一周误报 ≤ 1 次
- [ ] 阈值有标定数据支撑，不是拍脑袋
- [ ] 完成至少一次配对 A/B，报告含置信区间、效应量、成本、局限
- [ ] 有成本趋势图，能回答"评测一个月花多少钱、怎么降"

到这里，你的离线评测体系已经完整。阶段 5 解决最后一个问题：**离线分数很好，线上还是有人投诉。**
