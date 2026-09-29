from __future__ import annotations

"""Export auditable sample manifests from downloaded GEO-Bench-2 tortillas."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import tacoreader


FIELDS = {
    "treesatai": {
        "tortilla:id": "sample_id",
        "tortilla:data_split": "split",
        "source_path": "source_path",
        "ts_path": "advertised_ts_path",
        "year": "static_image_year",
        "stac:time_start": "stac_time_start",
        "lon": "longitude",
        "lat": "latitude",
        "species_labels": "labels",
        "dist_labels": "label_distributions",
    },
    "cloudsen12": {
        "tortilla:id": "sample_id",
        "tortilla:data_split": "split",
        "roi_id": "roi_id",
        "old_roi_id": "old_roi_id",
        "equi_id": "equi_id",
        "s2_id": "sentinel2_product_id",
        "stac:time_start": "stac_time_start",
        "lon": "longitude",
        "lat": "latitude",
        "label_type": "label_type",
    },
    "spacenet7": {
        "tortilla:id": "sample_id",
        "tortilla:data_split": "split",
        "patch_id": "patch_id",
        "aoi": "aoi",
        "source_img_file": "source_image",
        "source_mask_file": "source_mask",
        "year": "year",
        "month": "month",
        "stac:time_start": "stac_time_start",
        "lon": "longitude",
        "lat": "latitude",
    },
}

EXPECTED_ARTIFACT_SHA256 = {
    "treesatai": "0ddb8068720242ad4f5931ea91f3459ed695ad490bbaa48905afe72dd9623aee",
    "cloudsen12": "16b3c03d7b15cf42f6ef0cee6d453b6ad8ebbe7744674c4b58657511f7f5d0c0",
    "spacenet7": "f202abe270b729f7f2651de64cb5c6b41c5f9915109ec12b6c467afa2abcb5b6",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serializable(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def export(dataset: str, data_root: Path, output_root: Path) -> dict:
    artifact = data_root / dataset / f"geobench_{dataset}.tortilla"
    actual_sha = sha256(artifact)
    if actual_sha != EXPECTED_ARTIFACT_SHA256[dataset]:
        raise ValueError(f"{dataset} artifact checksum mismatch: {actual_sha}")
    source = tacoreader.load(str(artifact))
    mapping = FIELDS[dataset]
    frame = pd.DataFrame(
        {
            target: source[source_name].map(serializable)
            for source_name, target in mapping.items()
        }
    )
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate {dataset} sample_id values")
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / f"{dataset}_actual_manifest.csv"
    frame.to_csv(csv_path, index=False)
    summary = {
        "dataset": dataset,
        "geobench2_commit": "fd9d0b664e6fb0faba54636bdff4906634debd4b",
        "artifact_path": str(artifact.resolve()),
        "artifact_size_bytes": artifact.stat().st_size,
        "artifact_sha256": actual_sha,
        "manifest_path": str(csv_path.resolve()),
        "manifest_sha256": sha256(csv_path),
        "manifest_rows": len(frame),
        "observed_split_counts": {
            str(key): int(value) for key, value in frame["split"].value_counts().items()
        },
        "sample_id_unique": bool(frame["sample_id"].is_unique),
    }
    summary_path = output_root / f"{dataset}_actual_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", choices=sorted(FIELDS))
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-root", type=Path, default=Path("reports/dataset_manifests"))
    args = parser.parse_args()
    print(json.dumps([export(name, args.data_root, args.output_root) for name in args.datasets], indent=2))


if __name__ == "__main__":
    main()
