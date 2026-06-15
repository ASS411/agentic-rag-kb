-- Migration: add status column, relax answer NOT NULL constraint
-- Run against existing databases before deploying the new application code.

USE agentic_rag;

ALTER TABLE qa_records
    ADD COLUMN status VARCHAR(16) NOT NULL DEFAULT 'complete' AFTER question,
    MODIFY COLUMN answer TEXT NULL;
