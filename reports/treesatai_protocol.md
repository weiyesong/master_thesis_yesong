# GEO-Bench-2 TreeSatAI protocol and import record

Status: **STATIC BENCHMARK IMPORT VALIDATED; MULTI-TIMESTAMP ARTIFACT BLOCKED UPSTREAM**  
Date: 2026-08-10

## Pinned source and artifact

- Package/source: `GeoBenchV2` 0.9 at commit [`fd9d0b664e6fb0faba54636bdff4906634debd4b`](https://github.com/The-AI-Alliance/GEO-Bench-2/tree/fd9d0b664e6fb0faba54636bdff4906634debd4b).
- Official artifact: [aialliance/treesatai](https://huggingface.co/datasets/aialliance/treesatai), `geobench_treesatai.tortilla`.
- Local path: `/workspace/datasets/treesatai/geobench_treesatai.tortilla`.
- Size: 1,771,492,601 bytes.
- SHA256: `0ddb8068720242ad4f5931ea91f3459ed695ad490bbaa48905afe72dd9623aee` (matches the pinned dataset class).
- Actual sample manifest: `reports/dataset_manifests/treesatai_actual_manifest.csv`, SHA256 `b958ada672e7aeaa29652731af814834e9dae4fbe1c53d2edc92802ff18eae54`.

No raw TreeSatAI release was downloaded or substituted.

## Actual downloaded protocol

| Item | Observed value |
|---|---|
| Train | 4,000 |
| Validation | 1,000 |
| Test | 2,000 |
| Total | 7,000 |
| Sample ID | unique outer `tortilla:id`; original identity retained as `source_path` |
| Labels | 15-dimensional multi-hot vectors |
| Label cardinality | 1: 2,611; 2: 2,794; 3: 1,275; 4: 270; 5: 43; 6: 7 |
| Static S2 raster | 12×304×304, uint16, 10 m metadata |
| Model input | T×C×H×W = 1×12×224×224, bilinear image resize |

The resize preserves the full delivered footprint, corresponding to an effective pixel spacing of about 13.57 m at 224×224 (10 m × 304/224).

The official generation code uses a spatial 10×10 balanced checkerboard assignment with random state 42, followed by split-preserving subsetting with random state 24. The final artifact does not retain checkerboard block IDs, so the generation rule is source-verifiable but cannot be reconstructed solely from the delivered manifest. Exact `sample_id`, `source_path`, `ts_path`, and centroid sets are pairwise disjoint across train/validation/test.

## Class mapping

The installed class order and emitted labels are:

0 Abies; 1 Acer; 2 Alnus; 3 Betula; 4 Cleared; 5 Fagus; 6 Fraxinus; 7 Larix; 8 Picea; 9 Pinus; 10 Populus; 11 Prunus; 12 Pseudotsuga; 13 Quercus; 14 Tilia.

This is a **multi-label** task. The official class definition contains 15 classes, although its dataset docstring says “13-class.” The downloaded metadata uses all 15 named classes. Future training must use a multi-label objective and metrics; the existing multiclass cross-entropy evaluation path must not be used unchanged.

## Bands, wavelengths, and normalization

Official S2 order: B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09.

Canonical `wavelengths_nm`: 490, 560, 665, 842, 705, 740, 783, 865, 1610, 2190, 443, 945. The resolved DOFA adapter receives 0.490, 0.560, 0.665, 0.842, 0.705, 0.740, 0.783, 0.865, 1.610, 2.190, 0.443, 0.945 µm. Panopticon receives the canonical nm values unchanged.

The loader uses the pinned GEO-Bench-2 channel-wise z-score statistics. A real two-sample batch was finite with normalized range −1.7758 to 3.1995.

## Temporal artifact finding

The official outer manifest records one static S2 raster and a `ts_path` for every sample, e.g. `sentinel-ts/..._2017.h5`. However:

- none of the 7,000 referenced HDF5 files is included under the official artifact root;
- the official Hugging Face repository publishes only the tortilla plus statistics/README files;
- the official dataset class attempts to open those external paths when `include_ts=True`;
- the tortilla itself packages only aerial, S1, and static S2 GeoTIFF modalities.

Therefore the number and timing of the advertised multi-temporal observations cannot be verified from the official benchmark artifact. The packaged static observation has one `stac:time_start` year per sample: 2011 (664), 2013 (680), 2014 (980), 2015 (630), 2016 (1,062), 2017 (426), 2018 (978), 2019 (811), and 2020 (769).

The project does not silently fetch the raw release. Its adapter represents the official packaged data as **T=1**, records `temporal_source=official_static_s2_single_timestamp`, and still uses the required common operation:

`each timestamp -> same encoder -> mean encoder representation -> classification head`

The implementation is shared by DOFA and Panopticon. It supports a temporal validity mask and has a unit test proving mean aggregation, but a genuine T>1 test remains blocked until an official GEO-Bench-2 artifact supplies the HDF5 observations.

## Smoke-test evidence

- Loader: PASS; `[2,1,12,224,224]` image batch, `[2,15]` multi-label targets, unique IDs, finite tensors.
- DOFA: PASS; real pretrained checkpoint, finite `[2,768]` features and `[2,15]` logits.
- Panopticon: PASS; official pretrained checkpoint, finite `[2,768]` features and `[2,15]` logits.
- Training: not run.

## Inconsistencies requiring an explicit thesis decision

1. The official artifact is static despite recording unavailable time-series paths. A multi-temporal TreeSatAI experiment cannot currently be claimed as validated.
2. The official dataset prose says 13 classes, but code and artifact define 15 multi-label classes.
3. The current general thesis metrics are multiclass; TreeSatAI needs a separately specified multi-label evaluation protocol before training.
