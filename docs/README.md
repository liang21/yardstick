# Yardstick · 文档索引

> 一套从零搭建 AI 应用评测体系的实践记录。
> 核心命题：**评测系统本身也需要被评测。**

## 阅读顺序

| 文件 | 内容 | 周期 |
|---|---|---|
| [00-roadmap.md](00-roadmap.md) | 总路径、阶段总览、求职衔接 | — |
| [stage0-foundation.md](stage0-foundation.md) | 打地基：批量调模型、结构化落盘、不确定性体感 | 2 周 |
| [stage1-rag-eval.md](stage1-rag-eval.md) | RAG 评测：可观测 pipeline、检索指标、分环节归因 | 3 周 |
| [stage2-dataset-and-judge.md](stage2-dataset-and-judge.md) | **评测集工程 + judge 校准（分水岭）** | 5 周 |
| [stage3-agent-eval.md](stage3-agent-eval.md) | Agent 轨迹评测、多轮对话评测 | 4 周 |
| [stage4-pipeline-regression.md](stage4-pipeline-regression.md) | 版本追溯、CI 回归、配对 A/B 与统计 | 3 周 |
| [stage5-online-redteam.md](stage5-online-redteam.md) | 线上评测闭环、红队、合规映射 | 3 周 |

## 贯穿全篇的几条主线

这些概念在多个阶段反复出现，是同一套思维的不同应用：

**截断与中断信号**
`finish_reason`（阶段 0）→ `stop_reason`（阶段 3）→ 线上异常信号（阶段 5）

**双向 trade-off 指标**
正确拒答率 / 过度拒答率（阶段 1）→ CI 双向卡点（阶段 4）→ ASR / 误拦率（阶段 5）

**级联降本**
judge 级联判定（阶段 2）→ 线上级联评分（阶段 5）

**版本化**
评测集版本（阶段 2）→ 四元追溯 + judge 版本（阶段 4）→ 回流后重建基线（阶段 5）

## 核心产出清单

走完全程应该拿到的可验证证据：

- [ ] 200+ 条分层评测集 + 代表性论证（分布对齐数据）
- [ ] rubric 迭代记录 + 人人 Kappa 曲线
- [ ] **judge 校准迭代表** ← 最有说服力的一项
- [ ] Agent 轨迹评测与分级判定（含 `PASS_WRONG_PATH`）
- [ ] 配对 A/B 报告（置信区间 + 效应量 + 成本 + 局限）
- [ ] CI 阈值标定数据（历史 2σ）
- [ ] 线上评测闭环图
- [ ] 红队报告 + TC260 31 种风险映射表

## 建议的仓库结构

```
yardstick/
├── docs/                  # 本目录
├── datasets/              # 评测集（标注：不得进入任何训练流程）
├── eval/
│   ├── runner.py          # 跑批（阶段 0）
│   ├── metrics/           # 检索、生成、Agent 指标
│   ├── judge/             # judge prompt 与版本
│   ├── calibrate.py       # 校准（阶段 2）
│   └── provenance.py      # 版本追溯（阶段 4）
├── experiments/           # 实验记录 jsonl
├── redteam/               # 攻击用例库（阶段 5）
├── reports/               # A/B 报告、红队报告、校准迭代表
└── tests/                 # CI 评测（阶段 4）
```

## 使用建议

不要通读。按周推进，每周只翻当周模块，做完练习再往下。每个阶段末尾有验收自测和进入下一阶段的标志，用它们自查。
