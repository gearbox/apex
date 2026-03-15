# Apex API - Makefile
# Common commands for development and deployment

.PHONY: help dev prod down logs migrate shell db-shell test test-cov test-integration test-integration-local test-all lint clean migrate-reset

# Default target
help:
	@echo "Apex API - Available Commands"
	@echo ""
	@echo "Development:"
	@echo "  make dev        - Start development environment (hot reload)"
	@echo "  make down       - Stop all containers"
	@echo "  make logs       - Follow container logs"
	@echo "  make shell      - Open shell in API container"
	@echo "  make db-shell   - Open PostgreSQL shell"
	@echo ""
	@echo "Database:"
	@echo "  make migrate    - Run database migrations"
	@echo "  make migrate-new NAME=xxx - Create new migration"
	@echo ""
	@echo "Testing:"
	@echo "  make test                    - Run unit tests (in API container)"
	@echo "  make test-integration        - Run integration tests (Docker with DB)"
	@echo "  make test-integration-local  - Run integration tests locally (needs postgres-test on 5433)"
	@echo "  make test-all                - Run all tests (unit + integration) in Docker with DB"
	@echo "  make test-cov                - Run all tests with coverage in Docker with DB"
	@echo "  make lint                    - Run linter"
	@echo ""
	@echo "Production:"
	@echo "  make prod       - Start production environment"
	@echo "  make build      - Build production image"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean      - Remove containers and volumes"

# =============================================================================
# Development
# =============================================================================

dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
	@echo ""
	@echo "Development environment started!"
	@echo "  API: http://localhost:8000"
	@echo "  Docs: http://localhost:8000/docs"
	@echo "  Database: localhost:5432"
	@echo ""
	@echo "Run 'make logs' to follow logs"
	@echo "Run 'make migrate' to apply migrations"

down:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml down

logs:
	docker compose logs -f

logs-api:
	docker compose logs -f api

shell:
	docker compose exec api /bin/bash

db-shell:
	docker compose exec postgres psql -U apex -d apex

# =============================================================================
# Database
# =============================================================================

migrate:
	docker compose exec api alembic upgrade head

migrate-new:
ifndef NAME
	$(error NAME is required. Usage: make migrate-new NAME=add_users_table)
endif
	docker compose exec api alembic revision --autogenerate -m "$(NAME)"

migrate-down:
	docker compose exec api alembic downgrade -1

migrate-history:
	docker compose exec api alembic history

migrate-reset: ## Drop and recreate the local dev database, then migrate
	docker compose exec postgres psql -U apex -d postgres -c "DROP DATABASE IF EXISTS apex;" -c "CREATE DATABASE apex;"
	uv run alembic upgrade head

# =============================================================================
# Testing
# =============================================================================

test:
	docker compose exec api pytest tests/unit/ -v

test-cov:
	docker compose -f docker-compose.test.yml run --build test-runner \
		pytest tests/ --cov=src --cov-report=html --tb=short --no-header
	docker compose -f docker-compose.test.yml down -v

test-integration:
	docker compose -f docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from test-runner
	docker compose -f docker-compose.test.yml down -v

test-integration-local:
	TEST_DATABASE_URL=postgresql+asyncpg://apex_test:apex_test@localhost:5433/apex_test \
	pytest tests/integration/ -v --tb=short

test-all:
	docker compose -f docker-compose.test.yml run --build test-runner \
		pytest tests/ -v --tb=short --no-header
	docker compose -f docker-compose.test.yml down -v

lint:
	docker compose exec api ruff check src/
	docker compose exec api ruff format --check src/

format:
	docker compose exec api ruff format src/

# =============================================================================
# Production
# =============================================================================

build:
	docker build -t apex-api:latest .

prod:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo ""
	@echo "Production environment started!"
	@echo "Run migrations: docker compose exec api alembic upgrade head"

# =============================================================================
# Cleanup
# =============================================================================

clean:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml down -v --remove-orphans
	docker image prune -f

clean-all: clean
	docker volume rm apex_postgres_data 2>/dev/null || true
	docker rmi apex-api:latest 2>/dev/null || true
