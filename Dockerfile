# Build a reproducible CPU service and generate the small artifact during image build.
FROM ghcr.io/astral-sh/uv:0.11.7 AS uv

FROM python:3.13-slim
COPY --from=uv /uv /uvx /bin/
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY data ./data
COPY artifacts/.gitkeep ./artifacts/.gitkeep
RUN uv sync --frozen --no-dev
RUN uv run --no-dev python -m ml_model_service.cli train --backend sklearn

EXPOSE 8000
CMD ["uv", "run", "--no-dev", "uvicorn", "ml_model_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
