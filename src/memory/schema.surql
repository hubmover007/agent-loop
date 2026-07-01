-- Agent-Loop Memory Pool Schema (SurrealDB v2)
-- 四层倒锥拓扑 + 语义边
-- Note: SurrealDB v2 requires IF NOT EXISTS for idempotent schema init

-- ============================================================
-- Layer 0: FACTS (锥尖 - 检索入口)
-- ============================================================
DEFINE TABLE IF NOT EXISTS fact SCHEMAFULL;

DEFINE FIELD IF NOT EXISTS fact_type ON fact TYPE string
    ASSERT $value IN ["entity", "facetpoint"];

-- entity: user:{id}, server:{ip}, tool:{name}
-- facetpoint: 精确断言，如 "nginx 监听端口 8000"
-- agent_id: namespace isolation — defaults to "shared" for backward compatibility
DEFINE FIELD IF NOT EXISTS name ON fact TYPE string;
DEFINE FIELD IF NOT EXISTS value ON fact TYPE any;
DEFINE FIELD IF NOT EXISTS agent_id ON fact TYPE string DEFAULT "shared";
DEFINE FIELD IF NOT EXISTS embedding ON fact TYPE array<float>;
DEFINE FIELD IF NOT EXISTS created_at ON fact TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at ON fact TYPE datetime DEFAULT time::now();

-- Unified schema fields (non-breaking additions)
DEFINE FIELD IF NOT EXISTS type ON fact TYPE string DEFAULT "fact";
DEFINE FIELD IF NOT EXISTS summary ON fact TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS trigger ON fact TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS action ON fact TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS outcome ON fact TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS lesson ON fact TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS confidence ON fact TYPE float DEFAULT 0.5;
DEFINE FIELD IF NOT EXISTS related_ids ON fact TYPE array DEFAULT [];
DEFINE FIELD IF NOT EXISTS related_type ON fact TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS context ON fact;
DEFINE FIELD IF NOT EXISTS steps ON fact TYPE array DEFAULT [];
DEFINE FIELD IF NOT EXISTS error ON fact TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS fix ON fact TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS tags ON fact TYPE array DEFAULT [];
DEFINE FIELD IF NOT EXISTS access_count ON fact TYPE int DEFAULT 0;

-- Composite unique: same (fact_type, name) only unique per agent namespace
DEFINE INDEX IF NOT EXISTS idx_fact_name ON fact FIELDS fact_type, name, agent_id UNIQUE;
DEFINE INDEX IF NOT EXISTS idx_fact_type ON fact FIELDS fact_type;
DEFINE INDEX IF NOT EXISTS idx_fact_agent ON fact FIELDS agent_id;
-- SurrealDB v2 MTREE: no DISTANCE COSINE TYPE F64 suffix
DEFINE INDEX IF NOT EXISTS idx_fact_mtree ON fact FIELDS embedding HNSW DIMENSION 768 TYPE F32;

-- ============================================================
-- Layer 1: FACETS (语义维度)
-- ============================================================
DEFINE TABLE IF NOT EXISTS facet SCHEMAFULL;

-- 例如: "运维:SSH", "飞书:IM", "AWS:EC2"
DEFINE FIELD IF NOT EXISTS name ON facet TYPE string;
DEFINE FIELD IF NOT EXISTS description ON facet TYPE string;
DEFINE FIELD IF NOT EXISTS embedding ON facet TYPE array<float>;
DEFINE FIELD IF NOT EXISTS created_at ON facet TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at ON facet TYPE datetime DEFAULT time::now();

DEFINE INDEX IF NOT EXISTS idx_facet_name ON facet FIELDS name UNIQUE;
DEFINE INDEX IF NOT EXISTS idx_facet_mtree ON facet FIELDS embedding HNSW DIMENSION 768 TYPE F32;

-- ============================================================
-- Layer 2: EPISODES (事件上下文 - 核心检索返回目标)
-- ============================================================
DEFINE TABLE IF NOT EXISTS episode SCHEMAFULL;

DEFINE FIELD IF NOT EXISTS title ON episode TYPE string;
DEFINE FIELD IF NOT EXISTS summary ON episode TYPE string;
DEFINE FIELD IF NOT EXISTS content ON episode TYPE string;
DEFINE FIELD IF NOT EXISTS embedding ON episode TYPE array<float>;
DEFINE FIELD IF NOT EXISTS tags ON episode TYPE array<string>;
DEFINE FIELD IF NOT EXISTS type ON episode TYPE string DEFAULT "episode";
DEFINE FIELD IF NOT EXISTS user_input ON episode TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS output ON episode TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS task_count ON episode TYPE int DEFAULT 0;
DEFINE FIELD IF NOT EXISTS session_id ON episode TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS consolidated ON episode TYPE bool DEFAULT false;
DEFINE FIELD IF NOT EXISTS consolidated_at ON episode;
DEFINE FIELD IF NOT EXISTS created_at ON episode TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at ON episode TYPE datetime DEFAULT time::now();

-- Unified schema fields (non-breaking additions; type/summary/tags already exist)
DEFINE FIELD IF NOT EXISTS project ON episode TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS trigger ON episode TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS action ON episode TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS outcome ON episode TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS lesson ON episode TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS confidence ON episode TYPE float DEFAULT 0.5;
DEFINE FIELD IF NOT EXISTS related_ids ON episode TYPE array DEFAULT [];
DEFINE FIELD IF NOT EXISTS related_type ON episode TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS steps ON episode TYPE array DEFAULT [];
DEFINE FIELD IF NOT EXISTS error ON episode TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS fix ON episode TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS access_count ON episode TYPE int DEFAULT 0;

DEFINE INDEX IF NOT EXISTS idx_episode_title ON episode FIELDS title;
DEFINE INDEX IF NOT EXISTS idx_episode_mtree ON episode FIELDS embedding HNSW DIMENSION 768 TYPE F32;

-- Full-text search analyzer (SurrealDB v2 compatible)
DEFINE ANALYZER IF NOT EXISTS search_analyzer TOKENIZERS blank,class FILTERS lowercase;
-- DEFINE INDEX IF NOT EXISTS idx_episode_ft ON episode FIELDS summary SEARCH ANALYZER search_analyzer;

-- ============================================================
-- Layer 3: PROJECTS (锥底 - 项目全景)
-- ============================================================
DEFINE TABLE IF NOT EXISTS project SCHEMAFULL;

DEFINE FIELD IF NOT EXISTS name ON project TYPE string;
DEFINE FIELD IF NOT EXISTS description ON project TYPE string;
DEFINE FIELD IF NOT EXISTS embedding ON project TYPE array<float>;
DEFINE FIELD IF NOT EXISTS created_at ON project TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at ON project TYPE datetime DEFAULT time::now();

-- Unified schema fields (non-breaking additions)
DEFINE FIELD IF NOT EXISTS type ON project TYPE string DEFAULT "project";
DEFINE FIELD IF NOT EXISTS confidence ON project TYPE float DEFAULT 0.5;
DEFINE FIELD IF NOT EXISTS tags ON project TYPE array DEFAULT [];

DEFINE INDEX IF NOT EXISTS idx_project_name ON project FIELDS name UNIQUE;
DEFINE INDEX IF NOT EXISTS idx_project_mtree ON project FIELDS embedding HNSW DIMENSION 768 TYPE F32;

-- ============================================================
-- 语义边: 节点间关系，每条边有自然语言描述 + 向量化
-- ============================================================
DEFINE TABLE IF NOT EXISTS edge SCHEMAFULL;

DEFINE FIELD IF NOT EXISTS source_type ON edge TYPE string;
DEFINE FIELD IF NOT EXISTS source_id ON edge TYPE record;
DEFINE FIELD IF NOT EXISTS target_type ON edge TYPE string;
DEFINE FIELD IF NOT EXISTS target_id ON edge TYPE record;
DEFINE FIELD IF NOT EXISTS relation ON edge TYPE string;
DEFINE FIELD IF NOT EXISTS embedding ON edge TYPE array<float>;
DEFINE FIELD IF NOT EXISTS created_at ON edge TYPE datetime DEFAULT time::now();

DEFINE INDEX IF NOT EXISTS idx_edge_source ON edge FIELDS source_id;
DEFINE INDEX IF NOT EXISTS idx_edge_target ON edge FIELDS target_id;
DEFINE INDEX IF NOT EXISTS idx_edge_mtree ON edge FIELDS embedding HNSW DIMENSION 768 TYPE F32;

-- ============================================================
-- 图关系定义 (SurrealDB RELATE)
-- ============================================================

-- fact → facet: 事实属于某维度
DEFINE TABLE IF NOT EXISTS belongs_to SCHEMAFULL TYPE RELATION IN fact OUT facet;

-- facet → episode: 维度下的事件
DEFINE TABLE IF NOT EXISTS contains SCHEMAFULL TYPE RELATION IN facet OUT episode;

-- episode → project: 事件属于项目
DEFINE TABLE IF NOT EXISTS part_of SCHEMAFULL TYPE RELATION IN episode OUT project;

-- 跨层引用 (fact 直接关联 episode)
DEFINE TABLE IF NOT EXISTS references_episode SCHEMAFULL TYPE RELATION IN fact OUT episode;

-- ============================================================
-- 系统元数据表
-- ============================================================

-- Agent 注册表
DEFINE TABLE IF NOT EXISTS agent_registry SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS agent_id ON agent_registry TYPE string;
DEFINE FIELD IF NOT EXISTS status ON agent_registry TYPE string;
DEFINE FIELD IF NOT EXISTS expert_profile ON agent_registry TYPE any;
DEFINE FIELD IF NOT EXISTS created_at ON agent_registry TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS destroyed_at ON agent_registry;
DEFINE INDEX IF NOT EXISTS idx_agent_id ON agent_registry FIELDS agent_id UNIQUE;

-- 任务表
DEFINE TABLE IF NOT EXISTS task_registry SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS task_id ON task_registry TYPE string;
DEFINE FIELD IF NOT EXISTS parent_id ON task_registry;
DEFINE FIELD IF NOT EXISTS scope ON task_registry TYPE string;
DEFINE FIELD IF NOT EXISTS status ON task_registry TYPE string;
DEFINE FIELD IF NOT EXISTS priority ON task_registry TYPE int;
DEFINE FIELD IF NOT EXISTS assigned_agent ON task_registry;
DEFINE FIELD IF NOT EXISTS branch_space ON task_registry;
DEFINE FIELD IF NOT EXISTS result ON task_registry TYPE any;
DEFINE FIELD IF NOT EXISTS created_at ON task_registry TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at ON task_registry TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_task_id ON task_registry FIELDS task_id UNIQUE;
DEFINE INDEX IF NOT EXISTS idx_task_parent ON task_registry FIELDS parent_id;
DEFINE INDEX IF NOT EXISTS idx_task_status ON task_registry FIELDS status;

-- 遗弃池
DEFINE TABLE IF NOT EXISTS discard_pool SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS agent_id ON discard_pool TYPE string;
DEFINE FIELD IF NOT EXISTS task_id ON discard_pool TYPE string;
DEFINE FIELD IF NOT EXISTS result ON discard_pool TYPE any;
DEFINE FIELD IF NOT EXISTS reason ON discard_pool TYPE string;
DEFINE FIELD IF NOT EXISTS agent_log ON discard_pool TYPE array<string>;
DEFINE FIELD IF NOT EXISTS discarded_at ON discard_pool TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_discard_agent ON discard_pool FIELDS agent_id;
DEFINE INDEX IF NOT EXISTS idx_discard_task ON discard_pool FIELDS task_id;
