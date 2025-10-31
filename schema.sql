-- Core schema for the S-C-P-C system.
-- The definitions follow the contracts outlined in PRD v4.1.

CREATE TABLE IF NOT EXISTS scene_keywords(
  scene_id VARCHAR(64) NOT NULL,
  keyword  VARCHAR(256) NOT NULL,
  PRIMARY KEY(scene_id, keyword)
);

CREATE TABLE IF NOT EXISTS keyword_weekly_metrics(
  keyword   VARCHAR(256) NOT NULL,
  iso_week  CHAR(8) NOT NULL,
  search_volume BIGINT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY(keyword, iso_week)
);

CREATE TABLE IF NOT EXISTS child_pairs(
  our_parent  VARCHAR(32) NOT NULL,
  our_child   VARCHAR(32) NOT NULL,
  comp_parent VARCHAR(32) NOT NULL,
  comp_child  VARCHAR(32) NOT NULL,
  active TINYINT DEFAULT 1,
  PRIMARY KEY(our_child, comp_child)
);

CREATE TABLE IF NOT EXISTS frontend_snapshots(
  child_asin VARCHAR(32) NOT NULL,
  iso_week   CHAR(8) NOT NULL,
  list_price DECIMAL(10,2), sale_price DECIMAL(10,2),
  coupon_flag TINYINT, coupon_amount DECIMAL(10,2),
  rank_main INT, rank_sub INT,
  rating DECIMAL(3,2), review_count INT,
  image_count INT, video_count INT,
  badges_json JSON, product_url TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY(child_asin, iso_week)
);

CREATE TABLE IF NOT EXISTS parent_funnel_weekly(
  parent_id VARCHAR(32) NOT NULL,
  iso_week  CHAR(8) NOT NULL,
  impr_ads BIGINT, clicks BIGINT, sessions BIGINT,
  orders BIGINT, revenue DECIMAL(12,2),
  buybox_pct DECIMAL(5,4), bsr_main INT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY(parent_id, iso_week)
);

CREATE TABLE IF NOT EXISTS child_funnel_weekly(
  child_asin VARCHAR(32) NOT NULL,
  iso_week   CHAR(8) NOT NULL,
  impr_ads BIGINT, clicks BIGINT, orders BIGINT, revenue DECIMAL(12,2),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY(child_asin, iso_week)
);

CREATE TABLE IF NOT EXISTS lead_sku_tags(
  parent_id VARCHAR(32) NOT NULL,
  child_asin VARCHAR(32) NOT NULL,
  is_traffic_driver TINYINT NOT NULL,
  PRIMARY KEY(parent_id, child_asin)
);

CREATE TABLE IF NOT EXISTS inventory_woc(
  child_asin VARCHAR(32) NOT NULL,
  iso_week   CHAR(8) NOT NULL,
  woc_fba DECIMAL(6,2), woc_local DECIMAL(6,2), woc_overseas DECIMAL(6,2),
  sla_local_days INT, transfer_leadtime_days INT,
  inbound_fba_woc_7d DECIMAL(6,2), inbound_fba_woc_14d DECIMAL(6,2),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY(child_asin, iso_week)
);

CREATE TABLE IF NOT EXISTS ads_weekly(
  child_asin VARCHAR(32) NOT NULL,
  iso_week   CHAR(8) NOT NULL,
  channel ENUM('sp','sb','sd') NOT NULL,
  spend DECIMAL(10,2), clicks BIGINT, impressions BIGINT,
  cpc DECIMAL(6,3), acos DECIMAL(6,3), roas DECIMAL(6,3),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY(child_asin, iso_week, channel)
);

CREATE TABLE IF NOT EXISTS profit_estimates(
  child_asin VARCHAR(32) NOT NULL,
  iso_week   CHAR(8) NOT NULL,
  est_gross_profit_unit DECIMAL(8,2),
  gross_profit_total DECIMAL(12,2),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY(child_asin, iso_week)
);

CREATE TABLE IF NOT EXISTS parent_features(
  parent_id VARCHAR(32) NOT NULL,
  iso_week  CHAR(8) NOT NULL,
  c_impr DECIMAL(8,4), c_ctr DECIMAL(8,4), c_cvr DECIMAL(8,4),
  contrib_json JSON,
  lead_stock_ok DECIMAL(5,2),
  lead_stock_risk_json JSON, evidence_json JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY(parent_id, iso_week)
);

CREATE TABLE IF NOT EXISTS child_features(
  child_asin VARCHAR(32) NOT NULL,
  iso_week   CHAR(8) NOT NULL,
  effective_woc DECIMAL(6,2), risk_level ENUM('NONE','LOW','HIGH'),
  gmroi_gross DECIMAL(8,3), gmroi_net_ads DECIMAL(8,3), ppad DECIMAL(8,3),
  evidence_json JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY(child_asin, iso_week)
);

CREATE TABLE IF NOT EXISTS scene_json(
  scene_id VARCHAR(64) NOT NULL,
  iso_week CHAR(8) NOT NULL,
  payload JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY(scene_id, iso_week)
);

CREATE TABLE IF NOT EXISTS competition_json(
  parent_id VARCHAR(32) NOT NULL,
  iso_week CHAR(8) NOT NULL,
  payload JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY(parent_id, iso_week)
);

CREATE TABLE IF NOT EXISTS parent_json(
  parent_id VARCHAR(32) NOT NULL,
  iso_week CHAR(8) NOT NULL,
  payload JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY(parent_id, iso_week)
);

CREATE TABLE IF NOT EXISTS child_json(
  parent_id VARCHAR(32) NOT NULL,
  iso_week CHAR(8) NOT NULL,
  payload JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY(parent_id, iso_week)
);

CREATE TABLE IF NOT EXISTS budget_plan(
  parent_id VARCHAR(32) NOT NULL,
  iso_week CHAR(8) NOT NULL,
  payload JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY(parent_id, iso_week)
);

CREATE TABLE IF NOT EXISTS weekly_report(
  parent_id VARCHAR(32) NOT NULL,
  iso_week CHAR(8) NOT NULL,
  markdown MEDIUMTEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY(parent_id, iso_week)
);
