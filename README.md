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

## Competition 模块：竞对特征工程
`scpc/etl/competition_features.py` 将竞争分析扩展为“事实→特征→对比→规则→LLM”的端到端流水线：从 Doris 周度快照抽取实体特征，拼接广告/自然/推荐流量与关键词结构，再生成主配对与逐对配对，衍生环比、场景汇总、落后洞察与根因证据包，为 LLM 判因提供统一事实底座。

### 数据契约
- **输入事实层**：
  - `bi_amz_asin_product_snapshot`：ASIN 页面周级快照（含锚点/补录、价格、排名、素材、优惠等原子字段）。
  - `bi_amz_asin_scene_tag`：场景形态标签映射，补齐 `scene_tag/base_scene/morphology/hyy_asin`。
- **输入流量层（可选）**：
  - `vw_sif_asin_flow_overview_weekly_std`：ASIN×周的广告/自然/推荐流量占比，统一到周日锚点并提供广告内部配比。
  - `vw_sif_keyword_daily_std` + `bi_amz_comp_kw_tag`：关键词日表与类型词典，构造 7D 均值的熵值、HHI、TopK 占比与品牌/竞品等份额。
- **流量特征写入**：`build_traffic_features()` 可直接消化上述视图输出，生成 `bi_amz_comp_traffic_entities_weekly` 对齐的字段集合，供页面快照在实体层融合。
- **特征与对比层输出**：
  1. `clean_competition_entities()` → `bi_amz_comp_entities_clean`：计算净价、排名、内容/社交得分并融合流量结构、关键词集中度，区分我方与竞品实体。
  2. `build_competition_pairs()` → `bi_amz_comp_pairs`：按 Leader/Median 口径输出主配对差异、流量 gap/score、压力带与置信度。
  3. `build_competition_pairs_each()` → `bi_amz_comp_pairs_each`：沉淀我方 ASIN 与每个竞品的逐对差异、流量缺口与对手证据。
  4. `build_competition_delta()` → `bi_amz_comp_delta`：严格 7D 环比窗口，追踪价差、压力、流量得分的 WoW 变化。
  5. `summarise_competition_scene()` → `bi_amz_comp_scene_week_metrics`：场景级周汇总，统计压力分位、流量压力/覆盖率与关键动作。
- **规则与 LLM 层输出**：
  6. `score_lag_signals()` → `bi_amz_comp_lag_insights`：匹配 `bi_amz_comp_lag_rule`，识别页面/流量落后类型、严重度与 Top 竞品。
  7. `assemble_llm_packets()` → `bi_amz_comp_llm_packet`：按 lag_type 打包根因证据 JSON 与 Prompt Hint，涵盖流量证据包。
  8. `create_llm_overview()` → `vw_amz_comp_llm_overview`：供 LLM 快速判定 lag_type 的概览视图（含 `traffic_mix`/`keyword` 信号）。

### 处理流程
1. **流量标准化**：`build_traffic_features` 合并流量周视图与关键词日表，输出广告结构、关键词集中度与类型占比，自动处理 7D 覆盖率与分母保护（`eps`）。
2. **事实→特征**：`clean_competition_entities` 在页面快照的基础上融合流量特征，补齐我方标记并衍生 `price_net/rank_score/social_proof/content_score` 等指标，结果写入 `bi_amz_comp_entities_clean`。
3. **特征→对比（周内）**：
   - `build_competition_pairs` 选取 Leader/Median 竞品，与我方比较价差、排名、内容、社交、徽章及流量结构差异，依据 `configs/competition_scoring.yaml` 和 `default_traffic` 策略计算多维得分与压力带；
   - `build_competition_pairs_each` 输出我方 × 所有竞品的逐对配对，用于落后洞察、Top 对手识别与 LLM 证据复用。
4. **对比→环比/汇总**：
   - `build_competition_delta` 结合上一周主配对与实体特征，计算 WoW 差异、流量 gap 变化与压力变化；
   - `summarise_competition_scene` 汇聚主配对与环比，生成场景周报级指标、动作计数，以及流量压力/覆盖率分位。
5. **规则→洞察**：`score_lag_signals` 根据 `bi_amz_comp_lag_rule` 与 `default_traffic` 规则，对主/逐对配对执行落后判定，产出 `lag_type/opp_type/severity`、流量落后类型与 Top 对手列表。
6. **LLM 判因**：
   - `create_llm_overview` 汇总核心指标（价差、排名、内容、流量得分、置信度等）供 LLM 第一层级判定问题所在；
   - `assemble_llm_packets` 为每种 lag_type 准备根因证据包（我方/竞品/差异/Top 对手/提示语），包含流量 gap、关键词类型缺口等关键信息。
7. **一键编排**：`build_competition_tables` 返回实体、主配对、逐对配对、环比、场景汇总五张 DataFrame；`compute_competition_features` 在此基础上构造 LLM Facts（概览 + 落后洞察 + 证据包），并附带 `traffic_gap/traffic_scores/traffic_confidence` 字段。

### 使用示例
```python
from scpc.etl.competition_features import (
    build_competition_tables,
    build_traffic_features,
    compute_competition_features,
)
from scpc.tests.data import (
    MY_ASINS_SAMPLE,
    build_competition_snapshot_sample,
    build_keyword_daily_sample,
    build_keyword_tag_sample,
    build_scene_tag_sample,
    build_scoring_rules_sample,
    build_traffic_flow_sample,
)

snapshots = build_competition_snapshot_sample()
scene_tags = build_scene_tag_sample()
traffic = build_traffic_features(
    build_traffic_flow_sample(),
    build_keyword_daily_sample(),
    keyword_tags=build_keyword_tag_sample(),
)
tables = build_competition_tables(
    snapshots,
    week="2025W10",
    previous_week="2025W09",
    my_asins=MY_ASINS_SAMPLE,
    scene_tags=scene_tags,
    scoring_rules=build_scoring_rules_sample(),
    traffic=traffic,
)
result = compute_competition_features(
    snapshots=snapshots,
    week="2025W10",
    previous_week="2025W09",
    my_asins=MY_ASINS_SAMPLE,
    scene_tags=scene_tags,
    traffic=traffic,
)
```
`tables` 可直接写入 Doris，`result.as_dict()` 则用于 LLM 消费与 JSON Schema 校验。

### 模块测试
`pytest scpc/tests/test_competition_features.py`
覆盖实体清洗、竞品配对、环比计算与 LLM Facts 组装全链路，并基于 `scpc/tests/data/competition_samples.py` 提供的两周样例数据验证 Sigmoid 打分、压力分档、流量缺口与动作识别逻辑。

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
可以复制 `.env.example` 并替换为真实值；`.env` 已加入 `.gitignore`，避免凭证泄露。`scpc.db.engine` 会在运行时查找 `.env` 并根据以上字段拼接连接串，也支持直接使用环境变量提供完整 `DB_URI` 覆盖。

## 配置文件
- `configs/prod.yaml`：定义时区、Cron 调度、特征参数（如 `theta_days`、`alpha_effective_woc`）、预算闸门以及本地 `storage/` 输出目录前缀；
- `configs/scoring_rules.yaml`：给出父体/子体的基准阈值（例如 GMROI、PPAD）供收益优先策略使用。

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

任务会在启动阶段读取 `.env` 中的数据库与 DeepSeek 配置，日志记录（脱敏）后的运行环境，随后按流水线依次执行特征计算、LLM 裁决、预算分配与报告生成。场景与竞品特征结果会缓存在进程内的 `FeatureCache` 中，便于在多父体任务中重用。

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
运行 `pytest` 可验证父体/子体特征计算与 LLM Schema 校验逻辑。建议在扩展特征、接入真实数据或完善编排后同步新增测试用例；本仓库提供的样例
数据能够复刻常见边界场景，便于离线模拟与回归。

## 下一步建议
- 接入真实 SQLAlchemy 引擎并落地 UPSERT；
- 扩充 LLM 回退策略与 Prompt 细化；
- 建立 job_runs / change_log 持久化与监控面板。
