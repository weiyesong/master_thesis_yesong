from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import rasterio


CLASS_NAMES = (
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
)
SPLITS = ("train", "val", "calibration", "test")
RATIOS = {"train": 0.70, "val": 0.10, "calibration": 0.10, "test": 0.10}
FIELDS = (
    "sample_id", "dataset_index", "file_path", "class_label", "class_index",
    "split", "official_split", "sha256", "crs", "left", "bottom", "right",
    "top", "center_x", "center_y", "spatial_group",
)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_split_map(data_root: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = data_root / f"eurosat-{split}.txt"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            sample_id = Path(line.strip()).stem
            if sample_id in result:
                raise ValueError(f"Official split overlap: {sample_id}")
            result[sample_id] = split
    return result


def discover(data_root: Path) -> List[Dict[str, Any]]:
    image_root = data_root / "ds/images/remote_sensing/otherDatasets/sentinel_2/tif"
    paths = sorted(image_root.rglob("*.tif"), key=lambda value: value.as_posix())
    if not paths:
        raise FileNotFoundError(f"No GeoTIFF files under {image_root}")
    official = official_split_map(data_root)
    records: List[Dict[str, Any]] = []
    for index, path in enumerate(paths):
        label = path.parent.name
        if label not in CLASS_NAMES:
            raise ValueError(f"Unknown class directory: {label}")
        with rasterio.open(path) as dataset:
            bounds = dataset.bounds
            crs = dataset.crs.to_string() if dataset.crs else "unknown"
        sample_id = path.stem
        records.append(
            {
                "sample_id": sample_id,
                "dataset_index": index,
                "file_path": path.relative_to(data_root).as_posix(),
                "class_label": label,
                "class_index": CLASS_NAMES.index(label),
                "official_split": official.get(sample_id, "unknown"),
                "sha256": sha256_file(path),
                "crs": crs,
                "left": bounds.left,
                "bottom": bounds.bottom,
                "right": bounds.right,
                "top": bounds.top,
                "center_x": (bounds.left + bounds.right) / 2,
                "center_y": (bounds.bottom + bounds.top) / 2,
            }
        )
        if (index + 1) % 2500 == 0:
            print(f"Scanned {index + 1}/{len(paths)} samples", flush=True)
    return records


def attach_spatial_groups(records: List[Dict[str, Any]], search_cell_m: float = 1000.0) -> int:
    """Keep overlapping or <=20 m adjacent footprints in the same component."""
    union_find = UnionFind(len(records))
    cells: Dict[tuple[str, int, int], List[int]] = defaultdict(list)
    overlap_edges = 0
    hash_owner: Dict[str, int] = {}
    for index, record in enumerate(records):
        owner = hash_owner.setdefault(record["sha256"], index)
        union_find.union(index, owner)
        x_cell = math.floor(float(record["center_x"]) / search_cell_m)
        y_cell = math.floor(float(record["center_y"]) / search_cell_m)
        for x_offset in (-1, 0, 1):
            for y_offset in (-1, 0, 1):
                for other_index in cells[(record["crs"], x_cell + x_offset, y_cell + y_offset)]:
                    other = records[other_index]
                    x_overlap = min(float(record["right"]), float(other["right"])) - max(float(record["left"]), float(other["left"]))
                    y_overlap = min(float(record["top"]), float(other["top"])) - max(float(record["bottom"]), float(other["bottom"]))
                    if x_overlap >= -20 and y_overlap >= -20:
                        union_find.union(index, other_index)
                        overlap_edges += 1
        cells[(record["crs"], x_cell, y_cell)].append(index)
    roots = {root: group for group, root in enumerate(sorted({union_find.find(i) for i in range(len(records))}))}
    for index, record in enumerate(records):
        record["spatial_group"] = f"geo_{roots[union_find.find(index)]:05d}"
    return overlap_edges


def assign_group_stratified(records: List[Dict[str, Any]], seed: int) -> None:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["spatial_group"]].append(record)
    targets = {
        split: {label: round(sum(r["class_label"] == label for r in records) * RATIOS[split]) for label in CLASS_NAMES}
        for split in SPLITS
    }
    totals = {split: round(len(records) * RATIOS[split]) for split in SPLITS}
    totals["test"] = len(records) - sum(totals[name] for name in SPLITS[:-1])
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: -len(item[1]))
    counts = {split: Counter() for split in SPLITS}
    split_totals = Counter()
    for _, group in items:
        vector = Counter(record["class_label"] for record in group)
        scores = {}
        for split in SPLITS:
            class_cost = sum(
                ((counts[split][label] + vector[label] - targets[split][label]) / max(targets[split][label], 1)) ** 2
                - ((counts[split][label] - targets[split][label]) / max(targets[split][label], 1)) ** 2
                for label in CLASS_NAMES
            )
            total_cost = (
                ((split_totals[split] + len(group) - totals[split]) / totals[split]) ** 2
                - ((split_totals[split] - totals[split]) / totals[split]) ** 2
            )
            scores[split] = class_cost + total_cost
        chosen = min(SPLITS, key=lambda split: (scores[split], split_totals[split] / totals[split], SPLITS.index(split)))
        for record in group:
            record["split"] = chosen
        counts[chosen].update(vector)
        split_totals[chosen] += len(group)


def validate(records: List[Dict[str, Any]], data_root: Path) -> Dict[str, Any]:
    ids = [record["sample_id"] for record in records]
    paths = [record["file_path"] for record in records]
    duplicate_ids = sorted(key for key, value in Counter(ids).items() if value > 1)
    duplicate_paths = sorted(key for key, value in Counter(paths).items() if value > 1)
    missing = [path for path in paths if not (data_root / path).is_file()]
    split_sets = {split: {record["sample_id"] for record in records if record["split"] == split} for split in SPLITS}
    intersections = {
        f"{left}__{right}": sorted(split_sets[left] & split_sets[right])
        for index, left in enumerate(SPLITS) for right in SPLITS[index + 1 :]
    }
    hash_splits: Dict[str, set[str]] = defaultdict(set)
    group_splits: Dict[str, set[str]] = defaultdict(set)
    for record in records:
        hash_splits[record["sha256"]].add(record["split"])
        group_splits[record["spatial_group"]].add(record["split"])
    cross_hashes = sorted(key for key, values in hash_splits.items() if len(values) > 1)
    cross_groups = sorted(key for key, values in group_splits.items() if len(values) > 1)
    distribution = {
        split: {label: sum(r["split"] == split and r["class_label"] == label for r in records) for label in CLASS_NAMES}
        for split in SPLITS
    }
    errors = []
    if duplicate_ids: errors.append("duplicate sample_id")
    if duplicate_paths: errors.append("duplicate file_path")
    if missing: errors.append("missing files")
    if any(intersections.values()): errors.append("split intersection")
    if sum(len(values) for values in split_sets.values()) != len(records): errors.append("sample assignment is not exhaustive")
    if cross_hashes: errors.append("identical file content crosses splits")
    if cross_groups: errors.append("overlapping geospatial group crosses splits")
    return {
        "valid": not errors,
        "errors": errors,
        "total_samples": len(records),
        "split_counts": {split: len(split_sets[split]) for split in SPLITS},
        "class_distribution": distribution,
        "duplicate_sample_ids": duplicate_ids,
        "duplicate_file_paths": duplicate_paths,
        "missing_files": missing,
        "split_intersections": intersections,
        "cross_split_duplicate_hashes": cross_hashes,
        "cross_split_spatial_groups": cross_groups,
    }


def write_outputs(records: List[Dict[str, Any]], report: Dict[str, Any], output_dir: Path, seed: int, overlap_edges: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    records.sort(key=lambda record: int(record["dataset_index"]))
    with (output_dir / "eurosat_splits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: record[field] for field in FIELDS} for record in records)
    payload = {"schema_version": 1, "generation_seed": seed, "ratios": RATIOS, "class_names": CLASS_NAMES, "samples": records}
    (output_dir / "eurosat_splits.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report.update({"generation_seed": seed, "ratios": RATIOS, "grouped_spatial_edges_within_20m": overlap_edges})
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / "class_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_label", *SPLITS, "total"])
        for label in CLASS_NAMES:
            values = [report["class_distribution"][split][label] for split in SPLITS]
            writer.writerow([label, *values, sum(values)])
        values = [report["split_counts"][split] for split in SPLITS]
        writer.writerow(["TOTAL", *values, sum(values)])


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or validate immutable EuroSAT split manifests.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("splits/eurosat_70_10_10_10_spatial20m"))
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        records = load_manifest(args.output_dir / "eurosat_splits.csv")
        overlap_edges = int(json.loads((args.output_dir / "validation_report.json").read_text())["grouped_spatial_edges_within_20m"])
    else:
        if args.output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite immutable split directory: {args.output_dir}")
        records = discover(args.data_root)
        overlap_edges = attach_spatial_groups(records)
        assign_group_stratified(records, args.seed)
    report = validate(records, args.data_root)
    if not args.validate_only:
        write_outputs(records, report, args.output_dir, args.seed, overlap_edges)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
