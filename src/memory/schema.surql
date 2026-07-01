-- Agent-Loop Memory Pool Schema (SurrealDB)
-- 四层倒锥拓扑 + 语义边

-- ============================================================
-- Layer 0: FACTS (锥尖 - 检索入口)
-- ============================================================
DEFINE TABLE fact SCHEMAFULL;

DEFINE FIELD fact_type ON fact TYPE string
    ASSERT $value IN ["entity", "facetpoint"];

-- entity: user:{id}, server:{ip}, tool:{name}
-- facetpoint: 精确断言，如 "nginx 监听端口 8000"
-- agent_id: namespace isolation — defaults to "shared" for backward compatibility
DEFINE FIELD name ON fact TYPE string;
DEFINE FIELD value ON fact TYPE any;
DEFINE FIELD agent_id ON fact TYPE string DEFAULT "shared";
DEFINE FIELD embedding ON fact TYPE array<float>;
DEFINE FIELD created_at ON fact TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON fact TYPE datetime DEFAULT time::now();

-- Unified schema fields (non-breaking additions)
DEFINE FIELD type ON fact TYPE string DEFAULT "fact";
DEFINE FIELD summary ON fact TYPE string DEFAULT "";
DEFINE FIELD trigger ON fact TYPE string DEFAULT "";
DEFINE FIELD action ON fact TYPE string DEFAULT "";
DEFINE FIELD outcome ON fact TYPE string DEFAULT "";
DEFINE FIELD lesson ON fact TYPE string DEFAULT "";
DEFINE FIELD confidence ON fact TYPE float DEFAULT 0.5;
DEFINE FIELD related_ids ON fact TYPE array DEFAULT [];
DEFINE FIELD related_type ON fact TYPE string DEFAULT "";
DEFINE FIELD context ON fact TYPE option;
DEFINE FIELD steps ON fact TYPE array DEFAULT [];
DEFINE FIELD error ON fact TYPE string DEFAULT "";
DEFINE FIELD fix ON fact TYPE string DEFAULT "";
DEFINE FIELD tags ON fact TYPE array DEFAULT [];
DEFINE FIELD access_count ON fact TYPE int DEFAULT 0;

-- Composite unique: same (fact_type, name) only unique per agent namespace
DEFINE INDEX idx_fact_name ON fact FIELDS fact_type, name, agent_id UNIQUE;
DEFINE INDEX idx_fact_type ON fact FIELDS fact_type;
DEFINE INDEX idx_fact_agent ON fact FIELDS agent_id;
DEFINE INDEX idx_fact_mtree ON fact FIELDS embedding MTREE DIMENSION 1536 DISTANCE COSINE TYPE F64;

-- ============================================================
-- Layer 1: FACETS (语义维度)
-- ============================================================
DEFINE TABLE facet SCHEMAFULL;

-- 例如: "运维:SSH", "飞书:IM", "AWS:EC2"
DEFINE FIELD name ON facet TYPE string;
DEFINE FIELD description ON facet TYPE string;
DEFINE FIELD embedding ON facet TYPE array<float>;
DEFINE FIELD created_at ON facet TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON facet TYPE datetime DEFAULT time::now();

DEFINE INDEX idx_facet_name ON facet FIELDS name UNIQUE;
DEFINE INDEX idx_facet_mtree ON facet FIELDS embedding MTREE DIMENSION 1536 DISTANCE COSINE TYPE F64;

-- ============================================================
-- Layer 2: EPISODES (事件上下文 - 核心检索返回目标)
-- ============================================================
DEFINE TABLE episode SCHEMAFULL;

DEFINE FIELD title ON episode TYPE string;
DEFINE FIELD summary ON episode TYPE string;
DEFINE FIELD content ON episode TYPE string;
DEFINE FIELD embedding ON episode TYPE array<float>;
DEFINE FIELD tags ON episode TYPE array<string>;
DEFINE FIELD type ON episode TYPE string DEFAULT "episode";
DEFINE FIELD user_input ON episode TYPE string DEFAULT "";
DEFINE FIELD output ON episode TYPE string DEFAULT "";
DEFINE FIELD task_count ON episode TYPE int DEFAULT 0;
DEFINE FIELD session_id ON episode TYPE string DEFAULT "";
DEFINE FIELD consolidated ON episode TYPE bool DEFAULT false;
DEFINE FIELD consolidated_at ON episode TYPE option<datetime>;
DEFINE FIELD created_at ON episode TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON episode TYPE datetime DEFAULT time::now();

-- Unified schema fields (non-breaking additions; type/summary/tags already exist)
DEFINE FIELD project ON episode TYPE string DEFAULT "";
DEFINE FIELD trigger ON episode TYPE string DEFAULT "";
DEFINE FIELD action ON episode TYPE string DEFAULT "";
DEFINE FIELD outcome ON episode TYPE string DEFAULT "";
DEFINE FIELD lesson ON episode TYPE string DEFAULT "";
DEFINE FIELD confidence ON episode TYPE float DEFAULT 0.5;
DEFINE FIELD related_ids ON episode TYPE array DEFAULT [];
DEFINE FIELD related_type ON episode TYPE string DEFAULT "";
DEFINE FIELD steps ON episode TYPE array DEFAULT [];
DEFINE FIELD error ON episode TYPE string DEFAULT "";
DEFINE FIELD fix ON episode TYPE string DEFAULT "";
DEFINE FIELD access_count ON episode TYPE int DEFAULT 0;

DEFINE INDEX idx_episode_title ON episode FIELDS title;
DEFINE INDEX idx_episode_mtree ON episode FIELDS embedding MTREE DIMENSION 1536 DISTANCE COSINE TYPE F64;
DEFINE INDEX idx_episode_ft ON episode FIELDS summary SEARCH ANALYZER simple BM25;

-- ============================================================
-- Layer 3: PROJECTS (锥底 - 项目全景)
-- ============================================================
DEFINE TABLE project SCHEMAFULL;

DEFINE FIELD name ON project TYPE string;
DEFINE FIELD description ON project TYPE string;
DEFINE FIELD embedding ON project TYPE array<float>;
DEFINE FIELD created_at ON project TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON project TYPE datetime DEFAULT time::now();

-- Unified schema fields (non-breaking additions)
DEFINE FIELD type ON project TYPE string DEFAULT "project";
DEFINE FIELD confidence ON project TYPE float DEFAULT 0.5;
DEFINE FIELD tags ON project TYPE array DEFAULT [];

DEFINE INDEX idx_project_name ON project FIELDS name UNIQUE;
DEFINE INDEX idx_project_mtree ON project FIELDS embedding MTREE DIMENSION 1536 DISTANCE COSINE TYPE F64;

-- ============================================================
-- 语义边: 节点间关系，每条边有自然语言描述 + 向量化
-- ============================================================
DEFINE TABLE edge SCHEMAFULL;

DEFINE FIELD source_type ON edge TYPE string;
DEFINE FIELD source_id ON edge TYPE record;
DEFINE FIELD target_type ON edge TYPE string;
DEFINE FIELD target_id ON edge TYPE record;
DEFINE FIELD relation ON edge TYPE string;
DEFINE FIELD embedding ON edge TYPE array<float>;
DEFINE FIELD created_at ON edge TYPE datetime DEFAULT time::now();

DEFINE INDEX idx_edge_source ON edge FIELDS source_id;
DEFINE INDEX idx_edge_target ON edge FIELDS target_id;
DEFINE INDEX idx_edge_mtree ON edge FIELDS embedding MTREE DIMENSION 1536 DISTANCE COSINE TYPE F64;

-- ============================================================
-- 图关系定义 (SurrealDB RELATE)
-- ============================================================

-- fact → facet: 事实属于某维度
DEFINE TABLE belongs_to SCHEMAFULL TYPE RELATION IN fact OUT facet;

-- facet → episode: 维度下的事件
DEFINE TABLE contains SCHEMAFULL TYPE RELATION IN facet OUT episode;

-- episode → project: 事件属于项目
DEFINE TABLE part_of SCHEMAFULL TYPE RELATION IN episode OUT project;

-- 跨层引用 (fact 直接关联 episode)
DEFINE TABLE references_episode SCHEMAFULL TYPE RELATION IN fact OUT episode;

-- ============================================================
-- 系统元数据表
-- ============================================================

-- Agent 注册表
DEFINE TABLE agent_registry SCHEMAFULL;
DEFINE FIELD agent_id ON agent_registry TYPE string;
DEFINE FIELD status ON agent_registry TYPE string;
DEFINE FIELD expert_profile ON agent_registry TYPE any;
DEFINE FIELD created_at ON agent_registry TYPE datetime DEFAULT time::now();
DEFINE FIELD destroyed_at ON agent_registry TYPE option<datetime>;
DEFINE INDEX idx_agent_id ON agent_registry FIELDS agent_id UNIQUE;

-- 任务表
DEFINE TABLE task_registry SCHEMAFULL;
DEFINE FIELD task_id ON task_registry TYPE string;
DEFINE FIELD parent_id ON task_registry TYPE option<string>;
DEFINE FIELD scope ON task_registry TYPE string;
DEFINE FIELD status ON task_registry TYPE string;
DEFINE FIELD priority ON task_registry TYPE int;
DEFINE FIELD assigned_agent ON task_registry TYPE option<string>;
DEFINE FIELD branch_space ON task_registry TYPE option<string>;
DEFINE FIELD result ON task_registry TYPE any;
DEFINE FIELD created_at ON task_registry TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON task_registry TYPE datetime DEFAULT time::now();
DEFINE INDEX idx_task_id ON task_registry FIELDS task_id UNIQUE;
DEFINE INDEX idx_task_parent ON task_registry FIELDS parent_id;
DEFINE INDEX idx_task_status ON task_registry FIELDS status;

-- 遗弃池
DEFINE TABLE discard_pool SCHEMAFULL;
DEFINE FIELD agent_id ON discard_pool TYPE string;
DEFINE FIELD task_id ON discard_pool TYPE string;
DEFINE FIELD result ON discard_pool TYPE any;
DEFINE FIELD reason ON discard_pool TYPE string;
DEFINE FIELD agent_log ON discard_pool TYPE array<string>;
DEFINE FIELD discarded_at ON discard_pool TYPE datetime DEFAULT time::now();
DEFINE INDEX idx_discard_agent ON discard_pool FIELDS agent_id;
DEFINE INDEX idx_discard_task ON discard_pool FIELDS task_id;
