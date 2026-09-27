"""Command-line interface for offline model lifecycle operations."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .config import Settings
from .training import BackendName, evaluate_artifact, train_and_save


def _backend(value: str) -> BackendName:
    if value not in {"sklearn", "torch"}:
        raise argparse.ArgumentTypeError("backend must be sklearn or torch")
    return value  # type: ignore[return-value]


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", type=_backend, choices=("sklearn", "torch"))
    parser.add_argument("--data", type=Path, help="CSV fixture path")
    parser.add_argument("--artifact", type=Path, help="artifact output/input path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Support-ticket model lifecycle CLI")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="evaluate and save a serving artifact")
    _add_model_arguments(train)
    train.add_argument("--seed", type=int, default=42)

    evaluate = commands.add_parser("evaluate", help="evaluate a saved artifact")
    _add_model_arguments(evaluate)

    serve = commands.add_parser("serve", help="run the FastAPI service")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--log-level", default=None)

    return parser


def _settings(backend: BackendName | None, data: Path | None) -> Settings:
    current = Settings()
    values: dict[str, object] = {}
    if backend is not None:
        values["model_backend"] = backend
    if data is not None:
        values["data_path"] = data
    return Settings(**{**current.model_dump(), **values})


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        import uvicorn

        settings = Settings()
        uvicorn.run(
            "ml_model_service.api:app",
            host=args.host or settings.host,
            port=args.port or settings.port,
            log_level=args.log_level or settings.log_level,
        )
        return 0

    settings = _settings(args.backend, args.data)
    backend: BackendName = args.backend or settings.model_backend
    artifact = args.artifact or settings.resolved_artifact_path()
    if args.command == "train":
        result = train_and_save(
            backend,
            settings.data_path,
            artifact,
            seed=args.seed,
        )
        payload = {
            "backend": result.backend,
            "artifact": str(result.artifact_path),
            "labels": list(result.labels),
            "metrics": result.metrics.as_dict(),
        }
    else:
        metrics = evaluate_artifact(backend, settings.data_path, artifact)
        payload = {
            "backend": backend,
            "artifact": str(artifact),
            "metrics": metrics.as_dict(),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
