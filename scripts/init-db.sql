-- Apex Database Initialization
-- This script runs automatically when the PostgreSQL container is first created.

-- Enable useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search optimization

-- Create application schema (optional - using public for simplicity)
-- CREATE SCHEMA IF NOT EXISTS apex;

-- Grant permissions (the default user already has permissions on public schema)
-- This is here as a placeholder for multi-user setups

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'Apex database initialized successfully';
END $$;
