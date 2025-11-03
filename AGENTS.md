# agents.md  
**AI 驱动的 S-C-P-C 周度复盘系统｜跨境电商·亚马逊运营（Codex 友好版）**

> 本文定义本项目中所有“Agent”的**职责、输入/输出契约、工具权限、运行/回退/审计**规范，供工程与提示协作统一执行。  
> 适用场景：亚马逊跨境电商运营，履约包含 **FBA / 本地发货（FBM-Local）/ 海外仓（OW）**；父体（Parent）含多子体（Child），子体分型 **TD/BS/PF**。  
> 运行栈：**Python 3.12、MySQL 8、DeepSeek（LLM 轻量裁决）**。  
> 约束：**所有数值计算在后端**完成；LLM 只做**结构化裁决**与短文本生成；**严格 JSON Schema 校验**；**周度主跑 + 日度增量**。

---

## 1. 代理总览（Agent Taxonomy）

| Agent 名称 | 角色 | 触发 | 主要输入（Facts） | 主要输出 | 工具权限 |
|---|---|---|---|---|---|
| `SceneAgent` | 场景大盘裁决 | 周度主跑/日度快照 | 关键词周度、VOL 聚合、wow/yoy/wow_sa、slope8、TopN 驱动 | `scene.json` + 摘要 | 读 MySQL、读配置、写结果 |
| `CompetitionAgent` | 竞对强度裁决 | 周度主跑/日度快照 | 子体映射对、两周前台字段、对比特征、动作识别 | `competition.json` + 摘要 | 读/写同上 |
| `ParentAgent` | 父体漏斗诊断（引流款库存口径） | 周度主跑 | 父体两周漏斗、LMDI、lead_stock_ok（多通道 WOC）、S/C 结论 | `parent.json` + 摘要 | 读/写同上 |
| `ChildAgent` | 子体微观定位（TD/BS 供给优先；PF 收益优先） | 周度主跑/日度快照 | 子体漏斗、库存 WOC、多渠道广告、**预估毛利**、竞对对比 | `child.json` + 摘要 | 读/写同上 |
| `BudgetAgent` | 收益优先预算分配 | 周度主跑（在 Child 后） | PF 的 `GMROI_net_ads/PPAD/有效WOC`、当前预算 | `budget_plan.json`（含闸门标记） | 读/写同上 |
| `ReportAgent` | Markdown 周报生成 | 周度主跑（最后） | S/C/P/C JSON + Budget | `scpc_report.md` | 读/写同上 |
| `GatekeeperAgent` | 人工复核闸门 | 预算大变更/上新下架 | 变更清单 | 审批状态 `approved/rejected/pending` | 写 `change_log` |
| `Orchestrator` | 编排/重试/回退/幂等 | 周度主跑/日度快照 | 作业参数 | DAG 级状态、产出落库 | 统一控制 |
| `SnapshotAgent` | 日度增量 | 每日 | 库存/前台/广告增量 | 刷新特征快照 | 读/写同上 |

---

## 2. 通用接口与信封（Message Envelope）

**接口：**
```python
class Agent(BaseModel):
    name: str
    timeout_sec: int
    max_retries: int = 1

    def run(self, ctx: "AgentContext") -> "AgentResult":
        """执行单次裁决/生成。不得做重数值计算。失败抛 AgentError。"""

@dataclass
class AgentContext:
    job_id: str           # 全局唯一，Orchestrator 生成
    iso_week: str         # e.g., 2025-W43
    scene_id: str | None
    parent_id: str | None
    config: dict          # 合并后的配置
    facts: dict          # 已计算好的 Facts JSON（由 features 层产出）
    dry_run: bool = False
```

**返回：**
```python
@dataclass
class AgentResult:
    status: Literal["success","failed","skipped"]
    payload: dict | str   # JSON（LLM输出）或 Markdown（仅 ReportAgent）
    metrics: dict         # tokens、latency、retry_count 等
    artifacts: list[str]  # 写库后的定位键（如 outputs 表主键）
```

**信封规范：**
- 必带：`job_id/iso_week`，以及实体 `scene_id/parent_id`。  
- 每个 Agent 写库时使用 **UPSERT**（`ON DUPLICATE KEY UPDATE`），幂等。

---

## 3. Orchestrator（DAG & 调度）

**DAG（周度主跑）**  
`SceneAgent → CompetitionAgent → ParentAgent → ChildAgent → BudgetAgent → ReportAgent → Persist`

**DAG（日度快照）**  
`SnapshotAgent → {SceneAgent|CompetitionAgent|ChildAgent(可选)} → Persist`

**调度：**  
- 周一 09:00 运行主跑；每日 08:00 刷新快照。  
- Orchestrator 负责：**参数化查询→特征计算→组装 Facts→LLM 调用→Schema 校验→回退/重试→落库→日志**。

---

## 4. 工具与权限（Tools & Adapters）

- **数据库**：`sqlalchemy` + `text()` 参数化；只允许走 `dao/*` 封装。  
- **LLM**：只允许 `scpc/llm/deepseek_client.py`；`response_format="json_object"`；温度 `0–0.2`。  
- **配置**：`configs/prod.yaml`、`scoring_rules.yaml`（仅用于特征阈值，**不直接进入 LLM**）。  
- **存储**：结果与快照写入 MySQL 的 `*_json` 与 `*_features` 表；Markdown 入 `weekly_report` 表。  
- **日志**：`structlog`/`logging`，字段化输出：`job_id/agent/iso_week/duration/rows/err`.

---

## 5. 事实输入（Facts）与 Schema 校验

- 每个 Agent 的输入均为 **Facts JSON**（由 `features/*` 产出）。  
- LLM 输出必须通过 `scpc/schemas/*.schema.json` 校验；**不合规即报错**。  
- 失败处理：**一次重试**（追加“只输出有效 JSON”），仍失败 → 规则回退（Rule-Based）并记录 `job_runs`。

---

## 6. 各 Agent 详规

### 6.1 `SceneAgent`
**目的**：判断大盘强弱、短期展望、TopN 关键词驱动。  
**输入 Facts（字段最小集）**：
```json
{
  "scene_id":"SCN-US-001",
  "period":{"w0":"2025-W43","w_1":"2025-W42","ly_w0":"2024-W43"},
  "vol":{"w0": 1240000, "w_1": 1180000, "w_52": 980000},
  "features":{"wow":0.050,"yoy":0.265,"wow_sa":0.042,"slope8":0.031},
  "top_drivers":[{"kw":"red widget","delta":0.12}, {"kw":"desk widget","delta":-0.05}],  
  "forecast":{"h2w":"mild_up","conf":0.72}
}
```
**输出**：`scene.json`（等级与分值、TopN 驱动、预测摘要、证据 ID、置信度）。  
**失败路径**：缺历史 → 降级为两周环比并标注 `insufficient_data=true`。

---

### 6.2 `CompetitionAgent`
**目的**：裁决竞争强度 P（分值+等级），识别威胁/机会。  
**输入 Facts（最小集）**：
```json
{
  "parent_id":"P-001",
  "pairs":[
    {"our_child":"B0X","comp_child":"C0Y","w0":{"price_net":29.9,"rank":220,"rating":4.4,"rc":2100,"badges":["AC"],"coupon":1},
     "w_1":{"price_net":31.9,"rank":250,"rating":4.4,"rc":2050,"badges":[],"coupon":0}}
  ],
  "features":{"price_gap":0.042,"rank_diff":-0.10,"sp_diff":-0.06,"content_diff":-0.18},
  "actions":[{"id":"F11","type":"price+coupon+AC","note":"竞品A降价5%+新券+AC"}]
}
```
**输出**：`competition.json`（P 分、等级、Top threats/opps、证据、置信度）。  
**失败路径**：映射对覆盖 <80% → 降权并在输出 `notes` 标注。

---

### 6.3 `ParentAgent`
**目的**：父体漏斗诊断（**引流款库存只看 TD**），判定跑赢/跑输/持平与主/辅短板。  
**输入 Facts（最小集）**：
```json
{
  "parent_id":"P-001","period":{"w0":"2025-W43","w_1":"2025-W42"},
  "funnel":{
    "delta_pct":{"impr":-0.039,"ctr":-0.029,"cvr":-0.081,"orders":-0.142},
    "lmdi":{"C_impr":-0.0396,"C_ctr":-0.0296,"C_cvr":-0.0834,"contrib":{"impr":0.26,"ctr":0.19,"cvr":0.55}}
  },
  "scene":{"grade":"缓慢增长","signals":{"wow_sa":0.05}},
  "competition":{"grade":"小幅加剧","score":0.62,"top_threats":[{"id":"F11","note":"竞品A降价+新券+AC"}]},
  "lead_stock":{
    "theta_days":7,"lead_stock_ok":0.68,
    "risk_list":[{"sku":"A1","effective_woc":0.35,"click_weight":0.32,"id":"LS1"}]
  }
}
```
**输出**：`parent.json`（状态、主/辅短板、根因、动作 3-5 条、置信度、证据）。  
**失败路径**：TD 覆盖 <60% → 降权库存因子并在 `notes` 说明。

---

### 6.4 `ChildAgent`
**目的**：子体微观定位。**TD/BS：供给优先**；**PF：收益优先（GMROI/PPAD）**。  
**输入 Facts（最小集）**：
```json
{
  "parent_id":"P-001","iso_week":"2025-W43",
  "children":[
    {"child_asin":"B0-TD-1","type":"TD",
     "funnel":{"w0":{"impr":220000,"clicks":7800,"orders":520,"revenue":25900.0},
               "w_1":{"impr":240000,"clicks":8200,"orders":640,"revenue":31950.0}},
     "inventory":{"woc_fba":0.6,"woc_local":0.5,"woc_overseas":1.2,
                  "effective_woc":1.30,"theta_w":1.0,"risk_level":"LOW"},
     "ads":{"spend":3200.0,"acos":0.28}, "competition":{"price_gap":0.02}},
    {"child_asin":"B0-PF-3","type":"PF",
     "profit":{"est_gp_unit":9.40,"gross_profit_total":2444.0},
     "ads":{"spend":1420.0,"acos":0.22},
     "inventory":{"effective_woc":2.3,"theta_w":1.0,"risk_level":"NONE"}}
  ]
}
```
**输出**：`child.json`（每子体 1–2 条动作、收益/风险说明、置信度；PF 额外输出 `GMROI_net_ads/PPAD`）。  
**失败路径**：缺毛利 → 使用近似口径（毛利率）并下调置信度。

---

### 6.5 `BudgetAgent`
**目的**：收益优先预算分配（PF 维度）。  
**输入**：`child.json` + 当前预算、阈值（`target_gmroi/ppad`、单 SKU 变更上限）。  
**输出**：`budget_plan.json`（from→to、金额、预估 ΔProfit、`pending_review` 标记）。  
**失败路径**：预算不足或冲突 → 输出 `pending_review` 并记录 `change_log`。

---

### 6.6 `ReportAgent`
**目的**：整合 S/C/P/C 与 Budget，生成 `scpc_report.md`（中文）。  
**输入**：四段 JSON + 预算计划。  
**输出**：Markdown（含：概览 KPI、S→C→P→C 因果链、Top 行动、负责人/目标/验证条件）。  
**失败路径**：缺段落 → 标注“LLM 回退/特征缺失”，继续生成。

---

### 6.7 `GatekeeperAgent`
**目的**：对超阈值变更（预算±20%、上新/下架/大券）进行人工审批。  
**输入**：`budget_plan.json` 或策略变更清单。  
**输出**：审批状态与审计落库（`change_log`）。  
**失败路径**：超时未批 → `pending_review`，不执行到生产。

---

## 7. LLM 约束与提示文件

- 统一入口：`scpc/llm/deepseek_client.py`  
- 参数：`temperature=0.1`、`top_p=0.9`、`response_format="json_object"`、`timeout_sec` 可配  
- **禁止**：让 LLM 计算比率、做回归/分解；不得生成新数字超出 Facts  
- **提示文件**位于 `prompts/`：`scene.md`/`competition.md`/`parent.md`/`child.md`  
- **更新流程**：提交 **更新理由与兼容性评估**（不破坏 Schema），经评审后合入

---

## 8. 校验、回退与重试

1) **前置校验**：Facts 完整性（覆盖率、缺失、极值截尾）。  
2) **Schema 校验**：`scpc/schemas/*.schema.json`。  
3) **重试**：1 次，追加约束“只输出有效 JSON”。  
4) **回退**：触发规则引擎（Rule-Based）产出简化版本，并在最终 JSON `notes` 标注“LLM 回退”。  
5) **记录**：所有失败/回退写入 `job_runs`（含 `job_id/agent/error`）。

---

## 9. 幂等、事务与日志

- 写入一律 **UPSERT**；关键步骤包裹事务；拆分为**特征入库**与**结果入库**两个事务段。  
- 日志级别：`info`（起止、耗时、记录数、写入键）、`error`（异常栈、裁决失败原因）。  
- 不打印敏感信息（净价/毛利只输出区间或 hash）。

---

## 10. 配置（摘要）

`configs/prod.yaml`
```yaml
timezone: America/Los_Angeles
scheduling: { weekly_main: "0 9 * * MON", daily_snapshot: "0 8 * * *" }
features:
  theta_days: 7
  lead_min_coverage: 0.60
  alpha_effective_woc: { fba: 1.0, transfer: 0.8, fbm: 0.5 }
llm:
  model: deepseek
  temperature: 0.1
  response_format: json_object
gate:
  budget_change_pct: 0.20
outputs_dir: storage/scpc_outputs/${ISO_WEEK}/
```

---

## 11. 观测与 SLO

- **SLO**：周度主跑成功率 ≥ 99%；单 Agent 90p 延迟 ≤ 15s；LLM 校验失败率 ≤ 2%。  
- **指标**：tokens 用量、重试次数、失败原因分布、各 Agent 端到端耗时。

---

## 12. 测试与样例

- `tests/test_features_parent.py`：LMDI、lead_stock_ok。  
- `tests/test_features_child.py`：effective_woc、GMROI/PPAD、风险分档。  
- `tests/test_llm_schema.py`：用样例 Facts 驱动四 Agent，校验 Schema；包含“失败→重试→回退”的用例。  

**本地跑：**
```bash
python -m scpc.jobs.weekly_main --scene_id SCN-US-001 --parent_id P-001
python -m scpc.jobs.daily_snapshot --parent_id P-001
```

---

## 13. 安全与合规

- 凭证走环境变量/密钥托管；审计落 `change_log`。  
- 对外共享报告前自动脱敏（可配置字段白/黑名单）。  
- LLM 请求体不包含 PII/敏感密钥。

---

## 14. 版本管理

- 本文件版本：`agents.md v1.0`（对应 PRD v4.1）。  
- 变更需提交 MR：说明**影响范围**（Schema/Prompt/DAO/DAG）、**回滚计划**、**兼容性**。

---

### 附：LLM 调用最小示例（ParentAgent）

```python
facts = build_parent_facts(...)      # 全部数值型，含 evidence_id
prompt = load_prompt("prompts/parent.md")
resp = deepseek_client.generate(prompt=prompt, facts=facts,
                                temperature=0.1, response_format="json_object")
obj = validate_json(resp, schema="schemas/parent.schema.json")  # 失败抛异常
upsert_parent_json(parent_id, iso_week, obj, job_id=job_id)
```
