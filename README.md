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

## Competition 模块
第二层（LLM 判因）竞争对比逻辑及其页面/流量清洗、特征与对比表已下线，当前仅保留特征工程 + Compare 结果（Stage-1/Stage-2/Stage-3 判因全部停用）。

### 主逻辑：事实 → 特征 → 对比
1. **事实表（Facts）**
   - 页面侧：`bi_amz_asin_product_snapshot` 与 `bi_amz_asin_scene_tag` 提供 ASIN 周级快照、场景映射、我方标记。
   - 流量侧：`vw_sif_asin_flow_overview_weekly_std`、`vw_sif_keyword_daily_std`、`bi_amz_comp_kw_tag` 提供广告/自然/推荐流量与 7 天关键词结构。
2. **特征生成（Features）**
   - `clean_competition_entities()` 衍生净价、排名、内容/社交得分，并融合 `build_traffic_features()` 产出的广告结构、关键词集中度等流量特征，落地到 `bi_amz_comp_entities_clean`/`bi_amz_comp_traffic_entities_weekly`。
3. **对比构建（Compare）**
   - 页面主配对：`build_competition_pairs()` 计算我方对 Leader/Median 的价差、排名、内容、社交、徽章差距及对应得分，入库至 `bi_amz_comp_pairs`；
   - 流量主配对：同函数返回的 `traffic_pairs` 描述广告结构与关键词缺口，入库至 `bi_amz_comp_traffic_pairs`；
   - 逐对配对与环比：`build_competition_pairs_each()`、`build_competition_delta()`、`summarise_competition_scene()` 分别刻画具体竞品、WoW 变化及场景周报指标，对应 `bi_amz_comp_pairs_each`、`bi_amz_comp_traffic_pairs_each`、`bi_amz_comp_delta`、`bi_amz_comp_scene_week_metrics`。
4. **环比与汇总（Delta & Summary）**
   - 评分参数取自 `configs/competition_scoring.yaml` 与 `default_traffic`，在主配对与逐对配对中计算 `score_*`、`t_score_*`、`pressure`、`t_pressure` 等结构化分值，形成页面 + 流量两条判断链；
   - LLM 判因阶段（Stage-1/2/3）已整体下线，Compare 生成的结构化事实即为最终产出。

### 数据契约
- **输入事实层**：页面与场景标签 (`bi_amz_asin_product_snapshot`、`bi_amz_asin_scene_tag`)；
- **输入流量层（可选）**：流量周视图与关键词日视图 (`vw_sif_asin_flow_overview_weekly_std`、`vw_sif_keyword_daily_std`、`bi_amz_comp_kw_tag`)；
- **特征/对比输出**：
  1. `clean_competition_entities()` → `bi_amz_comp_entities_clean`
  2. `build_traffic_features()` → `bi_amz_comp_traffic_entities_weekly`
  3. `build_competition_pairs()`/`build_competition_pairs_each()` → 页面与流量主配对、逐对配对
  4. `build_competition_delta()`、`summarise_competition_scene()` → WoW 变化与场景周报指标
  5. `build_competition_tables()` → Doris/Stage-3 消费的结构化事实

### 使用示例
```python
from scpc.etl.competition_features import (
    build_competition_tables,
    build_traffic_features,
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
```
`tables` 可直接写入 Doris，供 Compare/Stage-3 下游消费。

### 运行命令（数据清洗 + 特征 / Compare 入库）
```bash
python -m scpc.etl.competition_pipeline \
  --mk US \
  --scene-tag SCN-USBAG-01 \
  --write
```
该命令对齐 Scene 模块的 CLI 体验：
- `--week` 指定周日口径的 ISO 周；若省略则自动选择目标站点最近一周的快照；`--mk` 指定站点；可重复传入 `--scene-tag` 仅处理目标场景；
- 默认仅打印清洗与特征产出的行数，追加 `--write` 后会将页面与流量特征分别 UPSERT 到 `bi_amz_comp_entities_clean`、`bi_amz_comp_traffic_entities_weekly`，写入前会自动剔除流量专属字段并按 Doris 表结构重排列顺序；
- `--chunk-size` 可调节 Doris UPSERT 批次大小（默认 500），所有步骤会输出详细日志，结果同时写入 `SCPC_LOG_DIR`（默认 `storage/logs`）。
- 当快照、场景映射或筛选条件缺失时会直接报错并在日志中指出缺失原因，方便快速补齐数据后重跑。
- 若启用 `--write`，流水线会在 UPSERT 成功后额外输出 Doris 侧的验证命令及行数判定，便于立即确认指定周度是否落库成功。

若要同步生成对比结果，可启用 Compare：

```bash
python -m scpc.etl.competition_pipeline \
  --mk US \
  --scene-tag SCN-USBAG-01 \
  --with-compare \
  --write \
  --write-compare
```

- `--with-compare` 会在特征计算完成后进一步构建 `bi_amz_comp_pairs/bi_amz_comp_pairs_each/bi_amz_comp_delta` 以及流量侧配对表；
- `--write-compare` 控制 Compare 结果是否 UPSERT 至 Doris，未传入时仅返回行数摘要；
- `--previous-week` 可手动指定对比基准周（默认回推 `--week` 的上一周）；
- `--rule-name`/`--traffic-rule-name` 支持切换页面/流量评分规则；
- `--compare-only` 在特征已入库时跳过清洗阶段，仅执行 Compare，可搭配 `--week`+`--previous-week` 重算 WoW。

示例：特征已存在，仅重算 2025W10 vs 2025W09 的 Compare 并落库：

```bash
python -m scpc.etl.competition_pipeline \
  --mk US \
  --week 2025W10 \
  --previous-week 2025W09 \
  --with-compare \
  --compare-only \
  --write-compare
```

### 写库后校验
当 `--write` 执行完成后，日志会打印 Doris SQL 验证命令，可直接在 MySQL/Doris 客户端运行以下语句，确认实体与流量表是否存在目标周数据（示例：美国站 2025W44）：

```sql
SELECT COUNT(*) AS row_count
FROM bi_amz_comp_entities_clean
WHERE marketplace_id = 'US' AND week = '2025W44';

SELECT COUNT(*) AS row_count
FROM bi_amz_comp_traffic_entities_weekly
WHERE marketplace_id = 'US' AND week = '2025W44';

SELECT COUNT(*) AS row_count
FROM bi_amz_comp_pairs
WHERE marketplace_id = 'US' AND week = '2025W44';

SELECT COUNT(*) AS row_count
FROM bi_amz_comp_pairs_each
WHERE marketplace_id = 'US' AND week = '2025W44';

SELECT COUNT(*) AS row_count
FROM bi_amz_comp_delta
WHERE marketplace_id = 'US' AND week_w0 = '2025W44';

SELECT COUNT(*) AS row_count
FROM bi_amz_comp_scene_week_metrics
WHERE marketplace_id = 'US' AND week = '2025W44';
```

若返回 `row_count > 0`，说明入库成功；否则需回查日志中的 dropped_cols、missing_cols 等诊断信息，确认字段裁剪或 Doris 表结构是否需要调整。

### 模块测试
安装依赖：`pip install -r requirements-dev.txt`

`pytest scpc/tests/test_competition_features.py scpc/tests/test_competition_pipeline.py`

测试用例通过 `scpc/tests/data/competition_samples.py` 构造页面与流量事实表，覆盖 `build_traffic_features`、`clean_competition_entities`、主/逐对配对与环比汇总，确保页面/流量特征与 Compare 生成链路稳定。

> **说明**：原 LLM 判因（Stage-1/2/3）不再提供 CLI 开关或持久化产物，如需诊断可直接消费 Compare 结果或接入外部分析工具。

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
