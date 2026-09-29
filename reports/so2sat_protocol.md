# So2Sat classification protocol and integration validation

Date: 2026-08-09  
Status: **PIPELINE READY; REAL DATA AND REAL BACKBONES VALIDATED; NO TRAINING RUN**

## Scope decision

This project uses the official GEO-Bench-2 **m-So2Sat optical classification task**, not a new random split and not the separate multimodal Sentinel-1 + Sentinel-2 variant. This is the GEO-Bench-2 task described as 17-class local-climate-zone classification from Sentinel-2 optical imagery with 19,992/986/986 train/validation/test samples. The official `GeoBenchSo2Sat` implementation defaults to the ten Sentinel-2 bands used here.

Primary sources:

- [GEO-Bench-2 dataset overview](https://github.com/The-AI-Alliance/GEO-Bench-2/blob/fd9d0b664e6fb0faba54636bdff4906634debd4b/docs/index.md)
- [Official GEO-Bench-2 So2Sat dataset class](https://github.com/The-AI-Alliance/GEO-Bench-2/blob/fd9d0b664e6fb0faba54636bdff4906634debd4b/geobench_v2/datasets/so2sat.py)
- [Official GEO-Bench-2 generation script](https://github.com/The-AI-Alliance/GEO-Bench-2/blob/fd9d0b664e6fb0faba54636bdff4906634debd4b/geobench_v2/generate_benchmark/so2sat.py)
- [Official redistributed artifact](https://huggingface.co/datasets/aialliance/so2sat/tree/main)
- [Original So2Sat LCZ42 protocol](https://github.com/zhu-xlab/So2Sat-LCZ42)

## Dataset definition

| Property | Validated definition |
|---|---|
| Task | Single-label local climate zone classification |
| Classes | 17 |
| Modalities used | Sentinel-2 optical only |
| Bands | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 |
| Native stored patch | 10 × 32 × 32 |
| Pixel spacing | 10 m × 10 m in the released So2Sat patches |
| Ground footprint | Approximately 320 m × 320 m |
| Model input | 10 × 224 × 224, deterministic bilinear resize |
| Raw value convention | Decimal reflectance; original DN values were divided by 10,000 |
| Benchmark partitions | train 19,992; validation 986; test 986 |
| Calibration partition | None in the official artifact |
| Official file | `datasets/so2sat/geobench_so2sat.tortilla` |
| Official file size | 1,210,751,529 bytes |
| Official SHA-256 | `2f9aa3a0cbf7f5071d2fafee24156a6041a9c732f0f979881b8db582201aa7bc` |
| Package validated | `GeoBenchV2==0.9` |

The original So2Sat documentation states that every stored band is on a 10 m grid. Bands whose Sentinel-2 native ground sampling distance is 20 m were therefore already resampled before distribution; the integration does not attempt another band-dependent resampling. The only spatial transform added by this project is the deterministic 32-to-224 model resize.

## Official split and geographic logic

No project-side split generation occurs.

The lineage is:

1. So2Sat version 2 assigns 42 cities worldwide to training.
2. Ten different cities spanning ten cultural zones are held out; their western halves form validation and eastern halves form test.
3. GEO-Bench converts those pre-existing partitions and creates the balanced `m-So2Sat` subset without moving a sample between parent partitions.
4. GEO-Bench-2 reads the resulting `default_partition.json` and writes its split membership into `tortilla:data_split`.

The downloaded GEO-Bench-2 artifact was checked directly:

| Split | Total | Per class |
|---|---:|---:|
| train | 19,992 | 1,176 |
| validation (`val` internally) | 986 | 58 |
| test | 986 | 58 |
| total | 21,964 | — |

All 21,964 `patch_id` values are unique. The shared loader also checks pairwise train/validation/test ID disjointness. The artifact does not expose city names or coordinates, so the original geographic assignment cannot be reconstructed from its IDs alone; preservation is established from the official generation lineage and embedded split tags. The integration must not replace this with an IID split.

Relevant lineage source: [GEO-Bench So2Sat converter](https://github.com/ServiceNow/geo-bench/blob/4777011901e1e4e2519383921bc415df7deb0e47/make_benchmark/dataset_converters/so2sat.py) and [balanced benchmark resampler](https://github.com/ServiceNow/geo-bench/blob/4777011901e1e4e2519383921bc415df7deb0e47/make_benchmark/create_benchmark.py).

## Label mapping

The numeric mapping follows the official GEO-Bench-2 class sequence exactly, including its spelling and capitalization:

| Index | Class |
|---:|---|
| 0 | Compact high-rise |
| 1 | Compact middle-rise |
| 2 | Compact low-rise |
| 3 | Open high-rise |
| 4 | Open middle-rise |
| 5 | Open low-rise |
| 6 | Lightweight low-rise |
| 7 | Large low-rise |
| 8 | Sparsely built |
| 9 | Heavy industry |
| 10 | Dense Trees |
| 11 | Scattered trees |
| 12 | Bush, scrub |
| 13 | Low plants |
| 14 | Bare rock or paved |
| 15 | Bare soil or sand |
| 16 | Water |

Every loaded item cross-checks its numeric inner label against the outer tortilla label string. A mismatch raises an error rather than silently changing the mapping.

## Bands, spectral metadata, and normalization

| Order | Band | Center wavelength (nm) | DOFA value (µm) | Sentinel-2 native GSD | GEO-Bench mean | GEO-Bench std |
|---:|---|---:|---:|---:|---:|---:|
| 0 | B02 | 492.9971095687 | 0.4929971096 | 10 m | 0.1295105070 | 0.0414236039 |
| 1 | B03 | 559.5987534818 | 0.5595987535 | 10 m | 0.1172439903 | 0.0519625656 |
| 2 | B04 | 664.6300422882 | 0.6646300423 | 10 m | 0.1138101816 | 0.0733252466 |
| 3 | B05 | 704.0059319834 | 0.7040059320 | 20 m | 0.1271651983 | 0.0693643764 |
| 4 | B06 | 740.5521320761 | 0.7405521321 | 20 m | 0.1706723571 | 0.0750555247 |
| 5 | B07 | 782.4190761493 | 0.7824190761 | 20 m | 0.1928136498 | 0.0855887160 |
| 6 | B08 | 827.5394062383 | 0.8275394062 | 10 m | 0.1854843795 | 0.0865049884 |
| 7 | B8A | 864.7801257644 | 0.8647801258 | 20 m | 0.2072914541 | 0.0939712226 |
| 8 | B11 | 1613.8624163477 | 1.6138624163 | 20 m | 0.1768450141 | 0.1023889408 |
| 9 | B12 | 2203.6182057820 | 2.2036182058 | 20 m | 0.1284958571 | 0.0922746733 |

Wavelength centers come from the [Panopticon/Geobreeze Sentinel-2 sensor metadata](https://github.com/geobreeze/geobreeze/blob/main/geobreeze/datasets/metadata/sensors/sentinel2.yaml). The configuration stores nanometers. The shared model factory passes nanometers to Panopticon and converts the same ordered values to micrometers for DOFA.

Normalization is channelwise `(reflectance - mean) / std`, using the statistics embedded in the official `GeoBenchSo2Sat` implementation. GEO-Bench-2's statistics code computes dataset statistics through the training dataloader. No validation/test statistics and no ImageNet RGB statistics are used. The loader refuses a config whose declared means/stds differ from the validated official values.

This choice is scientifically appropriate for the benchmark data: the raw values are decimal reflectance and the statistics match that dynamic range. A finite forward smoke test establishes interface correctness, not downstream accuracy.

## Sample identity

The outer tortilla rows contain identifiers such as:

- `tortilla:id = sample_id_0003`
- `patch_id = id_0003`

The pipeline exports `patch_id` as `sample_id` because it is the stable patch identifier shared by the nested modalities. It also preserves the split name and dataset index in every sample dictionary. IDs do not encode city or coordinates.

## Shared-pipeline integration

Implementation is in the existing classification entrypoint; no So2Sat-specific training script was created.

- `GeoBenchSo2SatClassificationDataset` wraps the official package dataset.
- `make_dataloaders` dispatches on `data.name` and returns the same `train`/`val`/`test` dictionary used by the existing training/evaluation loop.
- `run_experiment` now accepts classification on either `eurosat` or `so2sat`.
- Both backbones use the existing `build_model` factory and the same downstream head contract.
- `resolve_config` forbids local manifests, split files, split fractions, split seeds, and other random-split settings for So2Sat.
- File hash, full split counts, class/band order, normalization, label strings, sample uniqueness, and split overlap are fail-closed checks.
- `GeoBenchV2==0.9` is pinned in the experiment requirements and Docker image.
- Dataset artifacts live under ignored `datasets/`; this directory is excluded from the content-addressed code snapshot.

Canonical configuration: `configs/so2sat_classification.yaml`.

## Validation performed

### Full dataset loading test

Result: **PASS**

- Official 1.21 GB artifact downloaded from `aialliance/so2sat`.
- SHA-256 matched the published GEO-Bench-2 checksum.
- Observed split counts: 19,992 train; 986 validation; 986 test.
- All 21,964 IDs were unique and pairwise split-disjoint.
- Loaded batch: `[2, 10, 224, 224]`.
- Example IDs: `id_113293`, `id_117269`.
- Example labels: 0, 2.
- Batch values were finite; after official z-score normalization and resizing, observed range was `[-1.46098, 3.75838]` and mean was `-0.18156`.

### One-batch DOFA forward

Result: **PASS**

- Real DOFA ViT-Base and pretrained checkpoint used.
- Checkpoint SHA-256: `4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1`.
- Input: `[2, 10, 224, 224]`.
- Backbone features: `[2, 768]`, all finite.
- Classification logits: `[2, 17]`, all finite.
- Missing checkpoint keys: none.
- Expected pretraining-only unexpected keys: `mask_token`, `projector.weight`, `projector.bias`.

### One-batch Panopticon forward

Result: **PASS**

- Real official `VIT_BASE14` Panopticon weights used.
- Checkpoint SHA-256: `55024f411a7f383ed1a646d9b833b65683a4443846603fc7d37591d5afe9d26e`.
- Input: `[2, 10, 224, 224]` with ten wavelength IDs in nanometers.
- Backbone features: `[2, 768]`, all finite.
- Classification logits: `[2, 17]`, all finite.
- Missing/unexpected checkpoint keys: none.

Both smoke models used frozen backbones and inference mode. No loss backward pass, optimizer step, epoch, checkpoint write, or full training was run.

### Regression and configuration tests

Result: **PASS — 31/31 tests**

The added tests verify that the official embedded partition is accepted and local/random split definitions are rejected. Existing EuroSAT, model, provenance, and prediction-export tests remain green. Configuration validation also passed for both So2Sat experiments.

Content-addressed code snapshot after integration: `sha256:c99c1409e6a01bd4c4e2aca5208e847548b487e3ab8b84a798c796d1e16baa01` (77 included code/config files).

## Known limitation for later calibration work

GEO-Bench-2 m-So2Sat provides train, validation, and test only. It does **not** provide an official calibration split. Validation is currently reserved for checkpoint selection, and test must remain untouched until final evaluation. Temperature Scaling should therefore not be added by fitting and evaluating on validation; a separate, explicitly approved calibration protocol is required. No such alternative split was invented in this integration.
