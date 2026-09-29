# GEO-Bench-2 SpaceNet7 static protocol and loader validation

Status: **OFFICIAL STATIC ARTIFACT, AOI SPLITS, AND COMMON LOADER VALIDATED**  
Date: 2026-08-10

## Pinned source and artifact

- GEO-Bench-2 0.9 commit: [`fd9d0b664e6fb0faba54636bdff4906634debd4b`](https://github.com/The-AI-Alliance/GEO-Bench-2/blob/fd9d0b664e6fb0faba54636bdff4906634debd4b/geobench_v2/datasets/spacenet7.py).
- Official artifact: [aialliance/spacenet7](https://huggingface.co/datasets/aialliance/spacenet7), `geobench_spacenet7.tortilla`.
- Local path: `/workspace/datasets/spacenet7/geobench_spacenet7.tortilla`.
- Size: 3,056,268,185 bytes.
- SHA256: `f202abe270b729f7f2651de64cb5c6b41c5f9915109ec12b6c467afa2abcb5b6`.
- Saved actual manifest: `reports/dataset_manifests/spacenet7_actual_manifest.csv`, SHA256 `ef5b1d89883d1cdee9fca81c2c37989f661d93fd21bdfda6dc333204aeefa5ce`.

Only the official benchmark artifact was used.

## Actual static protocol

| Split | Samples | AOIs |
|---|---:|---:|
| Train | 3,500 | 41 |
| Validation | 652 | 7 |
| Test | 1,152 | 12 |
| Total | 5,304 | 60 |

The official generation assigns whole AOIs to splits and then forms non-overlapping 512×512 patches. The downloaded metadata confirms that AOI, source-image, source-mask, patch-ID, and centroid identity sets are pairwise disjoint across train/validation/test. All 5,304 `tortilla:id` and `patch_id` values are unique.

Each month/patch is an independent static segmentation example. The loader does not group observations by AOI, build sequences, or perform temporal aggregation. Available observations span 2017–2020 and all calendar months, but this metadata is retained only for provenance.

## Image convention

- Bands: Planet RGB in red, green, blue order.
- Canonical benchmark wavelength metadata: 665, 560, 490 nm, from the pinned GEO-Bench generic RGB sensor registry.
- Native patch: 3×512×512.
- Observed pixel size: approximately 4.777314267 m.
- Model input: 3×224×224 using bilinear image resizing.
- Effective model-grid spacing over the same footprint: approximately 10.92 m (4.777314267 m × 512/224).
- Normalization: official GEO-Bench-2 channel-wise z-score statistics.

## Building/background mapping

The generation script rasterizes masks as binary 0=no building and 1=building. GEO-BenchV2 0.9 then adds one in `__getitem__`, emitting 1=no-building and 2=building, while declaring a three-name tuple `("background", "no-building", "building")`. Across all 5,304 source masks only raw values 0 and 1 occur; consequently official output class 0 is unreachable in the delivered artifact.

The thesis adapter makes the static binary task explicit:

- official 1 -> thesis 0, background/no-building;
- official 2 -> thesis 1, building;
- possible official 0 -> ignore index 255 (not observed in this artifact).

Mask resizing is nearest-neighbor only.

## Full integrity scan

Every one of the 5,304 image/mask pairs was checked:

- shape mismatches: 0;
- affine-transform mismatches: 0;
- CRS mismatches: 0;
- masks containing values outside raw {0,1}: 0;
- all masks contain background; 5,175 contain at least one building pixel.

Loader tests on each split produced synchronized finite `3×224×224` images and `224×224` binary masks. No training or model inference was performed for SpaceNet7.
