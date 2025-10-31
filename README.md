# SCPC 周度复盘系统

## 项目简介
SCPC（Scene-Competition-Parent-Child）面向亚马逊跨境运营团队，按周自动生成场景/竞品/父体/子体的诊断与行动建议。全部量化计算在后端完成，轻量 LLM 仅负责结构化裁决与文本摘要。系统遵循可插拔流水线与幂等落库约束，可无缝接入周会与预算分配流程。

核心能力：
- **周度主跑**：按周全量重算场景/竞品/父体/子体特征与决策；
- **特征工程标准化**：实现 LMDI 分解、引流款库存覆盖、GMROI/PPAD 等收益优先指标；
- **结构化输出**：产出场景/竞品/父体/子体 JSON、预算计划与 Markdown 周报，全部经 Schema 校验；
- **审计与闸门**：预算调整遵循 ±20% 闸门，大幅变更需人工审批后执行；
- **场景/竞品缓存**：针对相同场景与父体重复运行时复用特征结果，避免重复计算与数据库压力。

## 目录结构
```
configs/                 # YAML 配置（调度、阈值等）
schema.sql               # MySQL 建表示例，覆盖 RAW/特征/结果层
scpc/
  etl/                   # SQLAlchemy 数据读取（全部参数化）
  features/              # S/C/P/C 特征计算函数
  llm/                   # DeepSeek 客户端与编排器
  jobs/                  # 周度主跑 CLI 入口
  prompts/               # 对应 Schema 的 Prompt 模板
  reports/               # Markdown 周报拼装工具
  schemas/               # JSON Schema（用于 LLM 输出校验）
  tests/                 # 特征与 LLM 编排的最小单测
```

## 环境变量与 .env
项目默认通过环境变量配置 DeepSeek 与 MySQL 凭证，可在仓库根目录创建 `.env`：
```
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_API_KEY=your-deepseek-key
DEEPSEEK_TIMEOUT=30
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_DATABASE=scpc
MYSQL_USER=scpc_user
MYSQL_PASSWORD=change-me
```
可以复制 `.env.example` 并替换为真实值；`.env` 已加入 `.gitignore`，避免凭证泄露。代码通过 `scpc.settings` 自动加载 `.env`，并在任务启动时校验缺失项。

## 配置文件
- `configs/prod.yaml`：定义时区、Cron 调度、特征参数（如 `theta_days`、`alpha_effective_woc`）、预算闸门以及输出目录前缀；
- `configs/scoring_rules.yaml`：给出父体/子体的基准阈值（例如 GMROI、PPAD）供收益优先策略使用。

## 数据模型
`schema.sql` 提供最小可运行的建表语句，覆盖：
- 场景与关键词、竞品映射、父体/子体漏斗；
- 多渠道库存与广告投放；
- 预估毛利、特征层（parent/child_features）与 JSON 结果层；
- job_runs 与 change_log 可扩展用于审计。
部署前请在 MySQL 8 实例执行建表脚本，或在现有库上通过 `ALTER TABLE` 兼容更新。

## 任务入口
- 周度主跑：`python -m scpc.jobs.weekly_main --scene_id <SCENE> --parent_id <PARENT>`

任务会在启动阶段读取 `.env` 中的数据库与 DeepSeek 配置，日志记录（脱敏）后的运行环境，随后按流水线依次执行特征计算、LLM 裁决、预算分配与报告生成（目前示例实现仍为桩代码，可在此基础上扩展）。场景与竞品特征结果会缓存在进程内的 `FeatureCache` 中，便于在多父体任务中重用。

## 特征工程亮点
- `scpc/features/parent.py`：实现 LMDI 漏斗分解与引流款库存健康度 `lead_stock_ok`，同时给出证据结构。
- `scpc/features/child.py`：计算多渠道有效覆盖 `effective_woc`、GMROI/PPAD、库存风险等级，并兼容缺失毛利场景。
- `scpc/features/scene.py`、`scpc/features/competition.py`：覆盖大盘趋势、竞品对比与父体加权指标。

## LLM 编排
- `scpc/llm/deepseek_client.py` 提供 `create_client_from_env()`，通过 `.env` 生成 DeepSeek 客户端；
- `scpc/llm/orchestrator.py` 负责 Prompt 装配、Schema 校验、重试与回退；
- Prompt 模板位于 `scpc/prompts/`，输出 Schema 位于 `scpc/schemas/`。

## 报告与预算
- `scpc/reports/builder.py` 汇总 S/C/P/C 结论与预算计划为 Markdown 周报；
- 预算分配遵循收益优先策略（GMROI > PPAD > 稳定度），并结合库存风险与闸门策略。

## 测试
运行 `pytest` 可验证父体/子体特征计算与 LLM Schema 校验逻辑。建议在扩展特征、接入真实数据或完善编排后同步新增测试用例。

## 下一步建议
- 接入真实 SQLAlchemy 引擎并落地 UPSERT；
- 扩充 LLM 回退策略与 Prompt 细化；
- 建立 job_runs / change_log 持久化与监控面板。
