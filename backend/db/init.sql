-- MySQL 初始化脚本
-- Docker 容器首次启动时自动执行

CREATE DATABASE IF NOT EXISTS agentic_rag
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE agentic_rag;

-- 文档表
CREATE TABLE IF NOT EXISTS documents (
    doc_id VARCHAR(64) PRIMARY KEY,
    file_name VARCHAR(512) NOT NULL,
    doc_type VARCHAR(16) NOT NULL,       -- pdf, md, txt
    file_path VARCHAR(1024) NOT NULL,    -- 本地存储路径
    page_count INT DEFAULT 1,
    chunk_count INT DEFAULT 0,
    size_bytes BIGINT,
    status VARCHAR(16) DEFAULT 'processing',  -- processing, ready, error
    error_message TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 会话表
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id VARCHAR(64) PRIMARY KEY,
    title VARCHAR(512),                  -- 首条问题的摘要
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 问答记录表
CREATE TABLE IF NOT EXISTS qa_records (
    record_id VARCHAR(64) PRIMARY KEY,
    conversation_id VARCHAR(64) NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources_json JSON,                   -- [{chunk_id, doc_name, page, snippet, score}]
    agent_steps_json JSON,               -- Agent 各步骤的记录
    total_rounds INT,
    model VARCHAR(64),
    tokens_used INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- chunks 元数据表（模块 3.5）
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id VARCHAR(128) PRIMARY KEY,
    doc_id VARCHAR(64) NOT NULL,
    content_hash VARCHAR(64) NOT NULL,   -- SHA-256 hex digest
    char_count INT NOT NULL DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_chunks_doc_id (doc_id),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
