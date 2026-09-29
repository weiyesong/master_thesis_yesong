from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_manager import resolve_config, set_global_seed
from scripts.prediction_export import (
    checkpoint_sha256,
    classification_embedding_extractor,
    collect_deterministic_predictions,
    export_predictions,
    save_evaluation_artifacts,
    validate_prediction_export,
)
from scripts.run_experiments import (
    build_model,
    classification_type_from_config,
    load_config,
    make_dataloaders,
    model_display_name,
    select_experiments,
)


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-sample predictions from an existing checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--split", choices=("train", "val", "calibration", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Deprecated compatibility flag; schema v2 always exports backbone representations.",
    )
    parser.add_argument("--corruption-type", default="none")
    parser.add_argument("--corruption-severity", type=int, default=0)
    args = parser.parse_args()

    config = resolve_config(load_config(args.config), args.config, dry_run=False)
    experiment = select_experiments(config, [args.experiment])[0]
    seed = config["seed"]
    set_global_seed(seed, deterministic=bool(config.get("reproducibility", {}).get("deterministic", True)))
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    loaders = make_dataloaders(config)
    model = build_model(config, experiment["model"], num_classes=int(config["data"]["num_classes"])).to(device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state_dict)

    embedding_extractor = classification_embedding_extractor(model)
    inference_started = time.perf_counter()
    collected = collect_deterministic_predictions(
        model,
        loaders[args.split],
        device,
        limit_batches=args.limit_batches,
        embedding_extractor=embedding_extractor,
    )
    inference_seconds = time.perf_counter() - inference_started
    class_names = config["data"]["class_names"]
    model_name = model_display_name(experiment["model"])
    classification_type = classification_type_from_config(config)
    paths = export_predictions(
        args.output_dir,
        sample_ids=collected["sample_ids"],
        true_labels=collected["labels"],
        logits=collected["logits"],
        class_names=class_names,
        model_name=model_name,
        dataset=config["data"]["name"],
        adaptation_mode=experiment["model"]["adaptation_mode"],
        seed=seed,
        checkpoint=str(args.checkpoint.resolve()),
        split=args.split,
        uq_method="deterministic",
        corruption_type=args.corruption_type,
        corruption_severity=args.corruption_severity,
        expected_count=len(collected["sample_ids"]),
        embeddings=collected["embeddings"],
        classification_type=classification_type,
        multilabel_threshold=float(config.get("metrics", {}).get("multilabel_threshold", 0.5)),
        source={
            "config": str(args.config.resolve()),
            "split_manifest": str(
                Path(config["data"]["split_manifest"]).resolve()
                if config["data"].get("split_manifest")
                else "embedded_official_manifest"
            ),
            "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
            "checkpoint_run_id": checkpoint.get("run_id") if isinstance(checkpoint, dict) else None,
            "split_total_count": len(loaders[args.split].dataset),
            "limit_batches": args.limit_batches,
            "dry_run_checkpoint": bool(checkpoint.get("config", {}).get("dry_run", False)),
            "inference_seconds": inference_seconds,
            "samples_per_second": len(collected["sample_ids"]) / max(inference_seconds, 1e-12),
        },
    )
    metrics = save_evaluation_artifacts(args.output_dir, n_bins=int(config["metrics"]["n_bins"]))
    (args.output_dir / "recomputed_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = validate_prediction_export(args.output_dir, expected_count=len(collected["sample_ids"]))
    print(json.dumps({"paths": {key: str(value) for key, value in paths.items()}, "validation": report, "metrics": metrics}, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
