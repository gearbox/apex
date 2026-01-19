# Apex ComfyUI Generation API

A Litestar-based REST API service for ComfyUI image generation workflows.

## Features

- **Async job processing** - Submit jobs and poll for status/results
- **Image upload** - Upload reference images for image-to-image generation
- **Batch generation** - Generate multiple images in a single request
- **Aspect ratio support** - Automatic width calculation from height and aspect ratio
- **OpenAPI docs** - Interactive API documentation at `/docs`

## Requirements

- Python 3.10+
- ComfyUI instance accessible (via SSH port forwarding for remote nodes)
- A proper workflow bundle

## Installation

```bash
# Clone and navigate to project
git clone https://github.com/gearbox/apex.git
cd apex

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key settings:
- `COMFYUI_HOST` - ComfyUI server host (default: 127.0.0.1)
- `COMFYUI_PORT` - ComfyUI server port (default: 18188)
- `DEBUG` - Enable debug mode (default: false)

## Usage

### 1. Set up SSH tunnel to the GPU node instance

```bash
ssh -L 18188:localhost:18188 root@<gpu-node-host> -p <ssh-port>
```

### 2. Start the API server

```bash
# Using uv
uv run python -m src.main

# Or directly
python -m src.main
```

The API will be available at `http://localhost:8000`

### 3. API Documentation

Open `http://localhost:8000/docs` for interactive OpenAPI documentation.

## API Endpoints

### Health Check
```
GET /health/
```
Returns service health and ComfyUI connectivity status.

### Create Generation Job
```
POST /api/v1/generate/
```

Request body:
```json
{
  "prompt": "A beautiful sunset over mountains",
  "negative_prompt": "blurry, low quality",
  "height": 1024,
  "aspect_ratio": "16:9",
  "model_type": "aisha",
  "max_images": 1,
  "steps": 12,
  "seed": 12345
}
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "name": "A beautiful sunset over mountains",
  "created_at": "2024-01-15T10:30:00Z",
  "message": "Job queued successfully"
}
```

### Create Generation with Images
```
POST /api/v1/generate/with-images
Content-Type: multipart/form-data
```

Form fields:
- `data` - JSON generation request
- `image1` - First reference image (optional)
- `image2` - Second reference image (optional)

### Get Job Status
```
GET /api/v1/jobs/{job_id}
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "name": "A beautiful sunset over mountains",
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:30:01Z",
  "completed_at": "2024-01-15T10:30:15Z",
  "progress": 100.0,
  "images": [
    "http://127.0.0.1:18188/view?filename=gen_550e8400_00001_.png&type=output"
  ],
  "error": null
}
```

### List Jobs
```
GET /api/v1/jobs/?status=completed&limit=50
```

### Upload Image
```
POST /api/v1/images/upload
Content-Type: multipart/form-data
```

## Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | auto | Task name (generated from prompt if not provided) |
| `prompt` | string | required | Text prompt for generation |
| `negative_prompt` | string | default negative | Negative prompt |
| `height` | int | 1024 | Image height (256-2048) |
| `aspect_ratio` | enum | 1:1 | Aspect ratio for width calculation |
| `model_type` | enum | aisha | Model/workflow type |
| `max_images` | int | 1 | Batch size (1-4) |
| `seed` | int | random | Generation seed |
| `steps` | int | 12 | Sampling steps (1-20) |

### Aspect Ratios

- `1:1` - Square
- `4:3` / `3:4` - Standard photo
- `16:9` / `9:16` - Widescreen/Portrait
- `2:3` / `3:2` - Classic photo
- `21:9` - Ultra-wide

## Project Structure

```
comfyui-api/
├── src/
│   ├── api/
│   │   ├── routes/          # API endpoints
│   │   ├── schemas/         # Pydantic models
│   │   ├── services/        # Business logic
│   │   ├── app.py           # Litestar application
│   │   └── dependencies.py  # DI providers
│   ├── core/
│   │   └── config.py        # Settings
│   └── main.py              # Entry point
├── config/
│   └── bundles/             # ComfyUI workflow bundles
├── pyproject.toml
└── README.md
```

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/
```

## Quick Start

### Copy and edit environment
cp .env.example .env  # Edit .env with your R2 credentials

### Start development environment
make dev

### Run migrations
make migrate

### View logs
make logs

## Available Commands

| Command | Description |
|---------|-------------|
| `make dev` | Start dev environment (hot reload) |
| `make prod` | Start production environment |
| `make down` | Stop all containers |
| `make logs` | Follow container logs |
| `make migrate` | Run database migrations |
| `make shell` | Open shell in API container |
| `make db-shell` | Open PostgreSQL shell |
| `make test` | Run tests |

## Architecture
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│  Apex API   │────▶│ PostgreSQL  │
└─────────────┘     │  :8000      │     │  :5432      │
                    └──────┬──────┘     └─────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      ┌─────────────┐           ┌─────────────┐
      │ Cloudflare  │           │  ComfyUI    │
      │     R2      │           │ (external)  │
      └─────────────┘           └─────────────┘
```
