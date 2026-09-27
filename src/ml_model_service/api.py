"""FastAPI application for ticket classification."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status

from .config import Settings
from .protocol import predict_one
from .registry import ModelRegistry
from .schemas import (
    HealthResponse,
    ModelResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an app with an isolated registry, useful for tests and deployments."""

    runtime_settings = settings or Settings()
    registry = ModelRegistry(runtime_settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            registry.load()
            logger.info(
                "loaded %s model from %s",
                runtime_settings.model_backend,
                registry.artifact_path,
            )
        except Exception as error:  # startup remains live so readiness explains why
            registry.fail(error)
            logger.exception("model startup failed")
        application.state.model_registry = registry
        yield

    application = FastAPI(
        title="Support Ticket Model Service",
        version="0.1.0",
        description="Classify synthetic support tickets with a pre-trained artifact.",
        lifespan=lifespan,
    )
    application.state.model_registry = registry

    @application.get("/v1/health/live", response_model=HealthResponse)
    async def live() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.get("/v1/health/ready", response_model=HealthResponse)
    async def ready(request: Request) -> HealthResponse:
        current_registry: ModelRegistry = request.app.state.model_registry
        if not current_registry.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "model_not_ready",
                    "message": current_registry.error or "model is not ready",
                },
            )
        return HealthResponse(status="ready")

    @application.get("/v1/model", response_model=ModelResponse)
    async def model_info(request: Request) -> ModelResponse:
        current_registry: ModelRegistry = request.app.state.model_registry
        try:
            metadata = current_registry.metadata()
            backend = current_registry.backend
            if backend is None:
                raise RuntimeError("model is not ready")
        except RuntimeError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "model_not_ready", "message": str(error)},
            ) from error
        device_value = metadata.get("device")
        return ModelResponse(
            backend=backend.backend_name,
            model_version=backend.model_version,
            labels=list(backend.labels),
            algorithm=str(metadata.get("algorithm", "unknown")),
            artifact=str(metadata["artifact"]),
            device=device_value if isinstance(device_value, str) else None,
        )

    @application.post("/v1/predict", response_model=PredictionResponse)
    async def predict(
        payload: PredictionRequest, request: Request
    ) -> PredictionResponse:
        current_registry: ModelRegistry = request.app.state.model_registry
        backend = current_registry.backend
        if backend is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "model_not_ready", "message": "model is not ready"},
            )
        try:
            result = predict_one(backend, payload.text)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "invalid_text", "message": str(error)},
            ) from error
        except RuntimeError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "model_not_ready", "message": str(error)},
            ) from error
        return PredictionResponse(
            label=result.label,
            confidence=result.confidence,
            probabilities=result.probabilities,
            backend=backend.backend_name,
            model_version=backend.model_version,
        )

    return application


app = create_app()

__all__ = ["app", "create_app"]
