# SCPC 周度复盘系统

## 项目简介
SCPC（Scene-Competition-Parent-Child）面向亚马逊跨境运营团队，按周自动生成场景/竞品/父体/子体的诊断与行动建议。全部量化计算在后端完成，轻量 LLM 仅负责结构化裁决与文本摘要。系统遵循可插拔流水线与幂等落库约束，可无缝接入周会与预算分配流程。

核心能力：
- **周度主跑**：按周全量重算场景/竞品/父体/子体特征与决策；
- **特征工程标准化**：实现 LMDI 分解、引流款库存覆盖、GMROI/PPAD 等收益优先指标，并对接页面/流量一体化竞争特征；
- **结构化输出**：产出场景/竞品/父体/子体 JSON、预算计划与 Markdown 周报，全部经 Schema 校验；
- **审计与闸门**：预算调整遵循 ±20% 闸门，大幅变更需人工审批后执行；
- **场景/竞品缓存**：针对相同场景与父体重复运行时复用特征结果，避免重复计算与数据库压力。

## Scene 模块：AI 分析大盘
`scpc/etl/scene_pipeline.py` 提供 Doris 视图驱动的场景级大盘分析流水线，覆盖关键词清洗、场景特征聚合、驱动词拆解与 LLM 消费接口。模块按照 PRD v4.1 的 Ground Truth 和幂等约束设计，可直接落库到 `bi_amz_scene_*` 三张结果表。

### 数据契约
- **入参**：`scene`（场景名称）、`mk`（站点，如 US/UK/DE）、`weeks_back`（默认 60 周，决定最小 `year*100+week_num`）。
- **读取对象**：
  - `bi_amz_vw_scene_keyword`：获取场景关键词集合与 weight。
  - `bi_amz_vw_kw_week`：拉取关键词×周度事实（vol/rank/clickShare/conversionShare 及 ASIN 证据）。
  - `bi_amz_mv_scene_week`：覆盖率与周起始日（周日）。当 MV 不可用时会退化为事实聚合。
- **写入对象**（全部 `UPSERT INTO` 幂等）：
  - `bi_amz_scene_kw_week_clean`
  - `bi_amz_scene_features`
  - `bi_amz_scene_drivers`

### 流程概览
1. **时间对齐**：`scene_pipeline` 读取关键词清单后，基于事实和覆盖率表统一周索引，确保缺失 `startDate` 时也能由 `year/week_num` 推导为周日。
2. **关键词清洗**（`scpc.etl.scene_clean`）：
   - 线性插补连续 ≤2 周缺口，超过则 `gap_flag=1` 不做插补；
   - 对每个关键词执行 P1/P99 Winsorize 并计算稳健 z 值；
   - 使用 3 周移动平均输出平滑体量 `vol_s`，连同边界、gap 标记落库。
3. **场景特征**（`scpc.etl.scene_features`）：
   - 以关键词权重聚合 `VOL`，计算 WoW/YoY、去季节环比、8 周斜率、波动率、覆盖率、新词占比等；
   - 样本 ≥26 周尝试 STL 分解获取季节因子，失败则回退至季节天真 + MA(4) 并降低 `confidence`；
   - 基于 WoW/YoY/Slope 分段输出 `strength_bucket`，并生成 4 周滚动预测 `forecast_p10/p50/p90`。
4. **驱动词拆解**（`scpc.etl.scene_drivers`）：
   - 计算关键词对 WoW/YoY 变化的相对贡献 `contrib`，并保留绝对量变、排名与份额变化；
   - 区分 `horizon`（WoW/YoY）与 `direction`（pos/neg），按照 TopN（默认 10）落库。
5. **日志与幂等**：通过 `scpc.db.io.replace_into` 批量写入 Doris，日志记录写入行数与回退路径，确保相同参数重复执行结果一致。

### LLM 消费
- `scpc/llm/summarize_scene.py` 读取最新的 `scene_features` 与 `scene_drivers`，按 PRD 约束构造 Prompt 并校验 JSON Schema（位于 `scpc/schemas/scene_*.schema.json`）。
- 若 `confidence < 0.6` 或覆盖率不足，提示词会追加“谨慎判断”提醒，满足执行/复盘场景的风控要求。

### 运行示例
```bash
python -m scpc.etl.scene_pipeline \
  --scene "Storage Rack" \
  --mk US \
  --weeks-back 60 \
  --write \
  --with-llm \
  --emit-json \
  --emit-md \
  --outputs-dir storage/outputs/scene
```
默认只在 `--write` 传入时写库；不传 `--write` 将返回清洗/特征/驱动 DataFrame 的行数摘要，便于本地调试。运行前请通过环境变量或仓库根目录的 `.env` 提供 Doris 2.x 配置（`DORIS_HOST/PORT/USER/PASSWORD/DATABASE`），`scpc.db.engine` 会自动拼接连接串；如需自定义可直接设置 `DB_URI` 覆盖。`SCENE_TOPN` 环境变量可控制驱动词 TopN。运行过程中会在 `SCPC_LOG_DIR`（默认 `storage/logs`）下生成以场景和站点命名的日志文件，所有 `ERROR` 级别记录都会标注触发失败的具体函数调用，便于排查。

若数据库已存在最新的 `scene_features/scene_drivers` 数据，可使用 `--llm-only` 直接触发 DeepSeek 调用并生成 JSON/Markdown（跳过清洗与写库），缩短调试时间：

```bash
python -m scpc.etl.scene_pipeline \
  --scene "Storage Rack" \
  --mk US \
  --llm-only \
  --emit-json \
  --emit-md \
  --outputs-dir storage/outputs/scene
```

### 模块测试
安装依赖：`pip install -r requirements-dev.txt`

`pytest scpc/tests/test_scene_clean.py scpc/tests/test_scene_features.py scpc/tests/test_scene_drivers.py`
覆盖关键词插补/Winsorize、场景特征聚合与 TopN 贡献分解的主要路径。测试会加载 `scpc/tests/data/scene_samples.py`
中构造的“浴室架”场景样例（关键词：shower caddy / shower bag，周度事实来自 Doris 视图快照），可模拟缺失周、长缺口
与 WoW 正负驱动，确保关键逻辑在无真实数据库时也能复现。

## 亚马逊 ASIN 周度对比 ETL

`scpc/etl/asin_week_diff.py` 与 `scpc/jobs/asin_week_diff_job.py` 负责基于 `bi_amz_asin_product_snapshot` 生成周度对比结果
并落库到 `bi_amz_asin_product_week_diff`。ETL 以周日（`sunday` 字段）为锚点，将“本周 vs 上一周”数据左连接后在 Python 中计算排名、价格、
评价、内容、badge 差集以及规则标签（`price_action/rank_trend/promo_action`）。规则阈值与文案来源于 `configs/asin_week_diff_rules.yml`
配置，可按需调整。

- **依赖表**：`bi_amz_asin_product_snapshot`（源）与 `bi_amz_asin_product_week_diff`（目标）。
- **入参**：`--week`（必填），格式 `YYYY-Www`，与 snapshot 表的 `week` 字段一致，例如 `2025-W45`。
- **运行命令**：

  ```bash
  python -m scpc.jobs.asin_week_diff_job --week 2025-W45
  ```

- **前置条件**：指定周的 snapshot 已写入 Doris（`bi_amz_asin_product_snapshot.week = 参数值`），否则 Job 会以 ERROR 级别退出。
- **配置**：可选 `--config` 参数覆盖默认的 `configs/asin_week_diff_rules.yml`，用于自定义价/排/优惠标签阈值。



## 目录结构
```
configs/                 # YAML 配置（调度、阈值等）
schema.sql               # Doris (MySQL 协议) 建表示例，覆盖 RAW/特征/结果层
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
项目默认通过环境变量配置 DeepSeek 与 Doris 凭证，可在仓库根目录创建 `.env`：
```
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
# 可选：为 00 场景整合周报指定更强模型
DEEPSEEK_MODEL_WEEKLY_REPORT_FULL=deepseek-chat-pro
DEEPSEEK_API_KEY=your-deepseek-key
DEEPSEEK_TIMEOUT=30
# Doris 2.x 连接配置（会自动拼接连接串）
DORIS_HOST=127.0.0.1
DORIS_PORT=9030
DORIS_USER=scpc_user
DORIS_PASSWORD=change-me
DORIS_DATABASE=bi_amz
# 可选：指定 CLI 运行日志输出目录（默认 storage/logs）
SCPC_LOG_DIR=storage/logs
```
可以复制 `.env.example` 并替换为真实值；`.env` 已加入 `.gitignore`，避免凭证泄露。`DEEPSEEK_MODEL_WEEKLY_REPORT_FULL` 会在生成《00 场景整合周报》时覆盖默认模型，其余模块沿用 `DEEPSEEK_MODEL`。`scpc.db.engine` 会在运行时查找 `.env` 并根据以上字段拼接连接串，也支持直接使用环境变量提供完整 `DB_URI` 覆盖。

## 配置文件
- `configs/prod.yaml`：定义时区、Cron 调度、特征参数（如 `theta_days`、`alpha_effective_woc`）、预算闸门以及本地 `storage/` 输出目录前缀；
- `configs/scoring_rules.yaml`：给出父体/子体的基准阈值（例如 GMROI、PPAD）供收益优先策略使用。
- `configs/competition_scoring.yaml`：控制竞品 Compare 评分参数（页面/流量权重、阈值等）。

## 数据模型
`schema.sql` 提供最小可运行的建表语句，覆盖：
- 场景与关键词、竞品映射、父体/子体漏斗；
- 多渠道库存与广告投放；
- 预估毛利、特征层（parent/child_features）与 JSON 结果层；
- job_runs 与 change_log 可扩展用于审计。
部署前请在 Doris 2.x 集群执行建表脚本（兼容 MySQL 方言），或在现有库上通过 `ALTER TABLE` 兼容更新。

## 任务入口
- 场景大盘：`python -m scpc.etl.scene_pipeline --scene <SCENE> --mk <MK> --weeks-back 60 --write`
- 周度主跑：`python -m scpc.jobs.weekly_main --scene_id <SCENE> --parent_id <PARENT>`
- 场景周报 JSON：`python -m scpc.jobs.generate_weekly_scene_json --week 2025-W45 --scene_tag 浴室袋 --marketplace US`
  - 默认输出目录为 `storage/weekly_report/{week}/{scene_tag}`，可通过 `--storage` 重写。
- 场景 Markdown 周报：`python -m scpc.jobs.generate_weekly_scene_report --week 2025-W45 --scene_tag 浴室袋 --marketplace US --storage output/weekly_report`
  - 要求 `overall_summary.json` 等五个模块 JSON 已存在于 `{storage}/{week}/{scene_tag}`，命令会在 `reports/` 目录生成 01~05 章 Markdown，并确保 06/07 章与 00 汇总报告一并落地；`--storage` 可省略使用默认 `storage/weekly_report`。
  - 若 `06_traffic_flow.md`、`07_keyword_opportunity.md` 事先已由流量 Job 生成，则会直接复用；如缺失，命令会写入“数据缺失”占位 Markdown，方便运营在目录中一次性拿到 7 个模块与《00 场景整合周报》。

### 流量模块（flow_change/keyword_change）
`python -m scpc.jobs.generate_scene_traffic_json --week 2025-W45 --scene_tag 浴室袋 --marketplace US --storage output/weekly_report`

- **作用**：在已有的「页面变化+竞品分析」基础上，额外输出 `traffic/flow_change.json` 与 `traffic/keyword_change.json`，供后续 LLM 生成「流量 & 核心关键词」章节。Job 会自动完成周维度对齐、ASIN 样本筛选、广告/自然/推荐流量结构计算以及关键词头部池比对。
- **输入**：`week`（ISO 周字符串，周起始 Sunday）、`scene_tag`、`marketplace`，可选 `--storage` 重写输出目录（默认 `storage/weekly_report`）。
- **输出路径**：`{storage}/{week}/{scene_tag}/traffic/flow_change.json` 与 `traffic/keyword_change.json`，结构遵循 PRD 中的 overview/top list/关键词 diff 约定，全部 ASIN 明细均包含 `hyy_asin/marketplace_id/scene_tag` 字段便于 LLM 消费。
- **阈值配置**：`configs/scene_traffic_rules.yml` 定义了广告占比变化(`ad_change_thresholds`)、流量结构判定(`traffic_mix_thresholds`)与关键词画像变化(`keyword_profile_change`)的全局阈值，每个字段都配有中文注释说明含义。调整该文件会直接影响 `ad_change_type`、`traffic_mix_type`、`change_type` 等衍生标签；若配置缺失或文件不存在，程序会记录 WARN 并回退到内置默认值。
- **注意事项**：Job 会在日志中输出样本覆盖数、周度数据是否缺失、配置加载结果以及 JSON 写入路径；若 Doris 缺失上一周流量或关键词数据，依旧会生成 JSON，但对应字段将为 `null` 并记录 WARN 便于排查。

生成 JSON 后，可使用同场景的 Markdown Job 将两章报告落地：

```bash
python -m scpc.jobs.generate_scene_traffic_report \
  --week 2025-W45 \
  --scene_tag 浴室袋 \
  --marketplace US \
  --storage output/weekly_report
```

- **作用**：读取 `traffic/flow_change.json` 与 `traffic/keyword_change.json`，分别调用 DeepSeek 生成 `06_traffic_flow.md`（场景流量结构与投放策略）和 `07_keyword_opportunity.md`（搜索需求与关键词机会）。如任一 JSON 缺失，将写出仅含“数据缺失”说明的 Markdown，并在日志中记录 ERROR。
- **输出路径**：`{storage}/{week}/{scene_tag}/reports/06_traffic_flow.md` 与 `reports/07_keyword_opportunity.md`，文件命名与既有五章一致，方便 Report 汇总 Job 直接拼接。
- **提示**：Job 会沿用统一的 DeepSeek Client/日志框架，可在 `storage/logs` 查看两次 LLM 调用的耗时与 token 信息；若上一周的流量或关键词数据缺失，生成的 Markdown 会显式提示“只基于本周静态分析”。

任务会在启动阶段读取 `.env` 中的数据库与 DeepSeek 配置，日志记录（脱敏）后的运行环境，随后按流水线依次执行特征计算、LLM 裁决、预算分配与报告生成。场景特征结果会缓存在进程内的 `FeatureCache` 中，便于在多父体任务中重用。

## 特征工程亮点
- `scpc/features/parent.py`：实现 LMDI 漏斗分解与引流款库存健康度 `lead_stock_ok`，同时给出证据结构。
- `scpc/features/child.py`：计算多渠道有效覆盖 `effective_woc`、GMROI/PPAD、库存风险等级，并兼容缺失毛利场景。
- `scpc/features/scene.py`：覆盖大盘趋势与父体加权指标。

## LLM 编排
- `scpc/llm/deepseek_client.py` 提供 `create_client_from_env()`，通过 `.env` 生成 DeepSeek 客户端；
- `scpc/llm/orchestrator.py` 负责 Prompt 装配、Schema 校验、重试与回退；
- Prompt 模板位于 `scpc/prompts/`，输出 Schema 位于 `scpc/schemas/`。

## 报告与预算
- `scpc/reports/builder.py` 汇总 S/C/P/C 结论与预算计划为 Markdown 周报；
- 预算分配遵循收益优先策略（GMROI > PPAD > 稳定度），并结合库存风险与闸门策略。

## 测试
运行 `pytest` 可验证父体/子体特征计算与 LLM Schema 校验逻辑。建议在扩展特征、接入真实数据或完善编排后同步新增测试用例；本仓库提供的样例
数据能够复刻常见边界场景，便于离线模拟与回归。

常规测试流程：

1. `make install-dev` 安装/更新开发依赖。
2. `make test` 执行完整的 80+ 项单元测试套件。

在 CI 环境中也可以直接复用上述命令，确保测试体验与本地一致。

## 下一步建议
- 接入真实 SQLAlchemy 引擎并落地 UPSERT；
- 扩充 LLM 回退策略与 Prompt 细化；
- 建立 job_runs / change_log 持久化与监控面板。
