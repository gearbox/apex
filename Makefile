# Apex API - Makefile
# Common commands for development and deployment

.PHONY: help dev prod down logs migrate shell db-shell test lint clean

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
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linter"
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

# =============================================================================
# Testing
# =============================================================================

test:
	docker compose exec api pytest -v

test-cov:
	docker compose exec api pytest --cov=src --cov-report=html

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
