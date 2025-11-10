-- create_views_mvs.sql
-- Target: StarRocks/Doris OLAP
-- Purpose: Create canonical views, materialized view, and result tables for Scene: AI 分析大盘
-- Prerequisites: Base tables exist -> bi_amz_scene_keyword, bi_amz_aba_kw_week_board

/* =========================
   1) SAFE DROPS
   ========================= */
DROP VIEW IF EXISTS bi_amz_vw_scene_keyword;
DROP VIEW IF EXISTS bi_amz_vw_kw_week;
DROP MATERIALIZED VIEW IF EXISTS bi_amz_mv_scene_week;

DROP TABLE IF EXISTS bi_amz_scene_kw_week_clean;
DROP TABLE IF EXISTS bi_amz_scene_features;
DROP TABLE IF EXISTS bi_amz_scene_drivers;

/* =========================
   2) VIEWS
   ========================= */

-- 2.1 关键词规范化视图（直接使用 scene 字段）
CREATE VIEW bi_amz_vw_scene_keyword AS
SELECT
  scene,
  LOWER(TRIM(keyword))                   AS keyword_norm,
  marketplace_id,
  1.0                                    AS weight,    -- 默认 1，可后续外联权重表
  1                                      AS is_active, -- 默认启用
  MAX(update_time)                       AS last_update_time
FROM bi_amz_scene_keyword
GROUP BY scene, LOWER(TRIM(keyword)), marketplace_id;

-- 2.2 周度事实去重视图（同键取最新快照）
CREATE VIEW bi_amz_vw_kw_week AS
SELECT * FROM (
  SELECT
    marketplace_id,
    LOWER(TRIM(keyword))                   AS keyword_norm,
    year,
    week_num,
    startDate,
    endDate,
    estSearchesNum                         AS vol,
    rank,
    clickShare,
    conversionShare,
    asin1, asin1_clickShare, asin1_conversionShare,
    asin2, asin2_clickShare, asin2_conversionShare,
    asin3, asin3_clickShare, asin3_conversionShare,
    update_time,
    ROW_NUMBER() OVER (
      PARTITION BY marketplace_id, LOWER(TRIM(keyword)), year, week_num
      ORDER BY update_time DESC
    ) AS rn
  FROM bi_amz_aba_kw_week_board
) t
WHERE t.rn = 1;

 /* =========================
    3) MATERIALIZED VIEW
    ========================= */

-- 3.1 场景 × 周聚合物化视图
CREATE MATERIALIZED VIEW bi_amz_mv_scene_week
DISTRIBUTED BY HASH(scene, marketplace_id) BUCKETS 16
PROPERTIES (
  "replication_allocation" = "tag.location.default: 3"
)
AS
SELECT
  k.scene,
  k.marketplace_id,
  w.year,
  w.week_num,
  MIN(w.startDate)                                                 AS start_date,     -- 周起始（日）
  SUM(w.vol)                                                       AS vol_raw_sum,
  SUM(CASE WHEN w.vol IS NOT NULL THEN 1 ELSE 0 END)               AS kw_with_data,
  COUNT(*)                                                         AS kw_total,
  (SUM(CASE WHEN w.vol IS NOT NULL THEN 1 ELSE 0 END) * 1.0) 
    / NULLIF(COUNT(*), 0)                                          AS coverage
FROM (SELECT
  scene,
  LOWER(TRIM(keyword))                   AS keyword_norm,
  marketplace_id,
  1.0                                    AS weight,    -- 默认 1，可后续外联权重表
  1                                      AS is_active, -- 默认启用
  MAX(update_time)                       AS last_update_time
FROM bi_amz_scene_keyword
GROUP BY scene, LOWER(TRIM(keyword)), marketplace_id) k
JOIN (SELECT * FROM (
  SELECT
    marketplace_id,
    LOWER(TRIM(keyword))                   AS keyword_norm,
    year,
    week_num,
    startDate,
    endDate,
    estSearchesNum                         AS vol,
    rank,
    clickShare,
    conversionShare,
    asin1, asin1_clickShare, asin1_conversionShare,
    asin2, asin2_clickShare, asin2_conversionShare,
    asin3, asin3_clickShare, asin3_conversionShare,
    update_time,
    ROW_NUMBER() OVER (
      PARTITION BY marketplace_id, LOWER(TRIM(keyword)), year, week_num
      ORDER BY update_time DESC
    ) AS rn
  FROM bi_amz_aba_kw_week_board
) t
WHERE t.rn = 1) w
  ON k.keyword_norm   = w.keyword_norm
 AND k.marketplace_id = w.marketplace_id
WHERE k.is_active = 1
GROUP BY k.scene, k.marketplace_id, w.year, w.week_num;

 /* =========================
    4) RESULT TABLES (for ETL outputs)
    ========================= */

-- 4.1 关键词×周（清洗/插补/平滑后）
CREATE TABLE bi_amz_scene_kw_week_clean (
  scene                VARCHAR(512) NOT NULL COMMENT "场景",
  marketplace_id       VARCHAR(8)   NOT NULL COMMENT "站点",
  keyword_norm         VARCHAR(512) NOT NULL COMMENT "标准化关键字",
  year                 INT          NOT NULL COMMENT "年度",
  week_num             INT          NOT NULL COMMENT "ISO周",
  start_date           DATE         NOT NULL COMMENT "周起始日（周日）",
  vol_s                DOUBLE                COMMENT "平滑后体量",
  gap_flag             TINYINT               COMMENT "是否存在>2周缺口 1/0",
  winsor_low           DOUBLE                COMMENT "P1边界",
  winsor_high          DOUBLE                COMMENT "P99边界",
  z                    DOUBLE                COMMENT "稳健z分数（基于MAD）",
  last_update_time     DATETIME     DEFAULT CURRENT_TIMESTAMP COMMENT "更新时间"
) ENGINE=OLAP
UNIQUE KEY(scene, marketplace_id, keyword_norm, year, week_num)
DISTRIBUTED BY HASH(scene, marketplace_id) BUCKETS 16
PROPERTIES (
  "replication_allocation" = "tag.location.default: 3",
  "enable_unique_key_merge_on_write" = "true"
);

-- 4.2 场景×周特征表（供 API/LLM 消费）
CREATE TABLE bi_amz_scene_features (
  scene                VARCHAR(512) NOT NULL COMMENT "场景",
  marketplace_id       VARCHAR(8)   NOT NULL COMMENT "站点",
  year                 INT          NOT NULL COMMENT "年度",
  week_num             INT          NOT NULL COMMENT "ISO周",
  start_date           DATE         NOT NULL COMMENT "周起始日（周日）",
  VOL                  BIGINT                COMMENT "聚合体量",
  wow                  DOUBLE                COMMENT "环比",
  yoy                  DOUBLE                COMMENT "同比",
  season               DOUBLE                COMMENT "季节因子(均值=1)",
  wow_sa               DOUBLE                COMMENT "去季节环比",
  slope8               DOUBLE                COMMENT "8周动量斜率（归一）",
  breadth_wow_pos      DOUBLE                COMMENT "上涨广度（WoW）",
  breadth_yoy_pos      DOUBLE                COMMENT "上涨广度（YoY）",
  HHI_kw               DOUBLE                COMMENT "关键词集中度HHI",
  volatility_8w        DOUBLE                COMMENT "8周波动率",
  coverage             DOUBLE                COMMENT "覆盖率",
  new_kw_share         DOUBLE                COMMENT "新词量占比",
  strength_bucket      VARCHAR(2)            COMMENT "强弱档位 S1..S5",
  forecast_p10         DOUBLE                COMMENT "未来4周相对增幅p10",
  forecast_p50         DOUBLE                COMMENT "未来4周相对增幅p50",
  forecast_p90         DOUBLE                COMMENT "未来4周相对增幅p90",
  confidence           DOUBLE                COMMENT "置信度(0..1)",
  update_time          DATETIME     DEFAULT CURRENT_TIMESTAMP COMMENT "更新时间"
) ENGINE=OLAP
UNIQUE KEY(scene, marketplace_id, year, week_num)
DISTRIBUTED BY HASH(scene, marketplace_id) BUCKETS 16
PROPERTIES (
  "replication_allocation" = "tag.location.default: 3",
  "enable_unique_key_merge_on_write" = "true"
);

-- 4.3 场景×周驱动词明细（TopN 展开到行）
CREATE TABLE bi_amz_scene_drivers (
  scene                   VARCHAR(512) NOT NULL COMMENT "场景",
  marketplace_id          VARCHAR(8)   NOT NULL COMMENT "站点",
  year                    INT          NOT NULL COMMENT "年度",
  week_num                INT          NOT NULL COMMENT "ISO周",
  start_date              DATE         NOT NULL COMMENT "周起始日（周日）",
  horizon                 VARCHAR(8)   NOT NULL COMMENT "WoW/YoY",
  direction               VARCHAR(4)   NOT NULL COMMENT "pos/neg",
  keyword                 VARCHAR(512) NOT NULL COMMENT "驱动关键词",
  contrib                 DOUBLE                COMMENT "贡献度（相对场景增幅）",
  vol_delta               BIGINT                COMMENT "量变（绝对）",
  rank_delta              INT                   COMMENT "排名变化(负=改善)",
  clickShare_delta        DOUBLE                COMMENT "点击份额变化",
  conversionShare_delta   DOUBLE                COMMENT "转化份额变化",
  is_new_kw               TINYINT               COMMENT "是否新词 1/0",
  update_time             DATETIME     DEFAULT CURRENT_TIMESTAMP COMMENT "更新时间"
) ENGINE=OLAP
UNIQUE KEY(scene, marketplace_id, `year`, week_num, start_date, horizon, direction, keyword)
DISTRIBUTED BY HASH(scene, marketplace_id) BUCKETS 16
PROPERTIES (
  "replication_allocation" = "tag.location.default: 3",
  "enable_unique_key_merge_on_write" = "true"
);

-- 4.4 场景级 LLM 总结表（含周起始日）
CREATE TABLE IF NOT EXISTS bi_amz_scene_summary (
  scene             VARCHAR(512) NOT NULL COMMENT "场景名称",
  marketplace_id    VARCHAR(8)   NOT NULL COMMENT "站点",
  week              VARCHAR(16)  NOT NULL COMMENT "ISO周，例如 2025W45",
  sunday            DATE         NOT NULL COMMENT "该周起始日（周日）",
  confidence        DOUBLE                COMMENT "LLM 输出置信度",
  summary_str       STRING                COMMENT "LLM 最终总结文本",
  llm_model         VARCHAR(64)           COMMENT "LLM 模型名称",
  llm_version       VARCHAR(32)           COMMENT "Prompt/Schema 版本",
  created_at        DATETIME     DEFAULT CURRENT_TIMESTAMP COMMENT "生成时间 (UTC)",
  updated_at        DATETIME     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT "更新时间 (UTC)",
  PRIMARY KEY (scene, marketplace_id, week)
)
ENGINE=OLAP
UNIQUE KEY(scene, marketplace_id, week)
DISTRIBUTED BY HASH(scene, marketplace_id) BUCKETS 16
PROPERTIES (
  "replication_allocation" = "tag.location.default: 3",
  "enable_unique_key_merge_on_write" = "true"
);

/* =========================
   5) NOTES
   - 若 StarRocks 版本不支持本语法的 MVs，可切换为定时 INSERT INTO 结果表替代。
   - 覆盖率 coverage 依赖视图聚合；如需更精细分区，可在 MV 上加 PARTITION BY（视版本支持情况）。
   - BUCKETS 与副本数可按集群规模调整。
   ========================= */
