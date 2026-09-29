#!/usr/bin/env python3
"""Recompute uncertainty summaries from saved probability draws only.

The input must be an NPZ containing either ``[N,D,C]`` classification
probabilities or ``[N,D,C,H,W]`` segmentation probabilities, where ``D`` is
the stochastic-pass or ensemble-member axis.  This module deliberately has
no model, checkpoint, Torch, or project-code dependency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


PASSTHROUGH_KEYS = (
    "sample_id",
    "sample_ids",
    "label",
    "labels",
    "true_labels",
    "valid_mask",
    "member_seeds",
    "seeds",
    "member_run_ids",
)


def _binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    eps = np.finfo(np.float64).tiny
    values = np.clip(probabilities, eps, 1.0 - 1.0e-15)
    return -(values * np.log(values) + (1.0 - values) * np.log(1.0 - values))


def recompute(
    probabilities: np.ndarray,
    *,
    semantics: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    values = np.asarray(probabilities)
    if values.ndim not in (3, 5):
        raise ValueError(f"Expected [N,D,C] or [N,D,C,H,W], received {values.shape}")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError(f"Probabilities must be floating point, received {values.dtype}")
    if not np.isfinite(values).all():
        raise ValueError("Probability tensor contains non-finite values")
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum < -1.0e-6 or maximum > 1.0 + 1.0e-6:
        raise ValueError(f"Probability range is invalid: [{minimum}, {maximum}]")
    if semantics == "categorical":
        error = float(np.max(np.abs(values.sum(axis=2, dtype=np.float64) - 1.0)))
        if error > 1.0e-4:
            raise ValueError(f"Categorical probabilities do not sum to one; max error={error}")
    elif semantics != "independent_bernoulli":
        raise ValueError(semantics)

    n_samples = values.shape[0]
    output_shape = (n_samples, *values.shape[2:])
    entropy_shape = (n_samples, *values.shape[3:]) if values.ndim == 5 else (n_samples,)
    mean_probabilities = np.empty(output_shape, dtype=np.float32)
    predictive_variance = np.empty(output_shape, dtype=np.float32)
    predictive_entropy = np.empty(entropy_shape, dtype=np.float32)
    expected_entropy = np.empty(entropy_shape, dtype=np.float32)
    disagreement = np.empty(entropy_shape, dtype=np.float32)
    per_label_outputs: dict[str, np.ndarray] = {}
    if semantics == "independent_bernoulli":
        per_label_shape = (n_samples, values.shape[2])
        per_label_outputs = {
            "predictive_entropy_per_label": np.empty(per_label_shape, dtype=np.float32),
            "expected_predictive_entropy_per_label": np.empty(per_label_shape, dtype=np.float32),
            "mi_style_disagreement_per_label": np.empty(per_label_shape, dtype=np.float32),
        }

    eps = np.finfo(np.float64).tiny
    for start in range(0, n_samples, batch_size):
        stop = min(n_samples, start + batch_size)
        block = np.asarray(values[start:stop], dtype=np.float64)
        mean = block.mean(axis=1)
        variance = block.var(axis=1)
        mean_probabilities[start:stop] = mean.astype(np.float32)
        predictive_variance[start:stop] = variance.astype(np.float32)
        if semantics == "categorical":
            predictive = -(mean * np.log(np.clip(mean, eps, 1.0))).sum(axis=1)
            per_draw = -(block * np.log(np.clip(block, eps, 1.0))).sum(axis=2)
            expected = per_draw.mean(axis=1)
        else:
            predictive_per_label = _binary_entropy(mean)
            expected_per_label = _binary_entropy(block).mean(axis=1)
            per_label_outputs["predictive_entropy_per_label"][start:stop] = predictive_per_label
            per_label_outputs["expected_predictive_entropy_per_label"][start:stop] = expected_per_label
            per_label_outputs["mi_style_disagreement_per_label"][start:stop] = (
                predictive_per_label - expected_per_label
            )
            predictive = predictive_per_label.sum(axis=1)
            expected = expected_per_label.sum(axis=1)
        predictive_entropy[start:stop] = predictive.astype(np.float32)
        expected_entropy[start:stop] = expected.astype(np.float32)
        disagreement[start:stop] = (predictive - expected).astype(np.float32)

    return {
        "mean_probabilities": mean_probabilities,
        "predictive_entropy": predictive_entropy,
        "expected_predictive_entropy": expected_entropy,
        "mi_style_disagreement": disagreement,
        "predictive_variance": predictive_variance,
        **per_label_outputs,
    }


def _optional_array(archive: Any, key: str) -> np.ndarray | None:
    if key not in archive.files:
        return None
    try:
        return archive[key]
    except ValueError as error:
        if "Object arrays cannot be loaded" not in str(error):
            raise
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--probability-key", default="probabilities")
    parser.add_argument(
        "--semantics",
        choices=("categorical", "independent_bernoulli"),
        default="categorical",
        help="Use independent_bernoulli for TreeSatAI multilabel outputs.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--trust-input-pickle",
        action="store_true",
        help="Permit loading legacy object-dtype sample IDs from a trusted NPZ.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output paths must differ")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with np.load(args.input, allow_pickle=args.trust_input_pickle) as archive:
        if args.probability_key not in archive.files:
            raise KeyError(f"{args.probability_key!r} absent; available={archive.files}")
        probabilities = archive[args.probability_key]
        output = recompute(probabilities, semantics=args.semantics, batch_size=args.batch_size)
        for key in PASSTHROUGH_KEYS:
            value = _optional_array(archive, key)
            if value is not None:
                output[key] = value
        output["source_probability_shape"] = np.asarray(probabilities.shape, dtype=np.int64)
        output["draw_axis"] = np.asarray(1, dtype=np.int64)
        output["class_axis"] = np.asarray(2, dtype=np.int64)
        output["probability_semantics"] = np.asarray(args.semantics)

    np.savez_compressed(args.output, **output)
    print(
        json.dumps(
            {
                "input": str(args.input.resolve()),
                "output": str(args.output.resolve()),
                "input_shape": list(probabilities.shape),
                "semantics": args.semantics,
                "output_arrays": {key: list(value.shape) for key, value in output.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
