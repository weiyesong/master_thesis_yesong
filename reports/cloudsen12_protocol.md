# GEO-Bench-2 CloudSEN12 protocol and loader validation

Status: **OFFICIAL ARTIFACT AND COMMON OPTICAL LOADER VALIDATED WITH NOTED SPLIT-SOURCE OVERLAP**  
Date: 2026-08-10

## Pinned source and artifact

- GEO-Bench-2 0.9 commit: [`fd9d0b664e6fb0faba54636bdff4906634debd4b`](https://github.com/The-AI-Alliance/GEO-Bench-2/blob/fd9d0b664e6fb0faba54636bdff4906634debd4b/geobench_v2/datasets/cloudsen12.py).
- Official artifact: [aialliance/cloudsen12](https://huggingface.co/datasets/aialliance/cloudsen12), `geobench_cloudsen12.tortilla`.
- Local path: `/workspace/datasets/cloudsen12/geobench_cloudsen12.tortilla`.
- Size: 9,944,531,349 bytes.
- SHA256: `16b3c03d7b15cf42f6ef0cee6d453b6ad8ebbe7744674c4b58657511f7f5d0c0`.
- Saved actual manifest: `reports/dataset_manifests/cloudsen12_actual_manifest.csv`, SHA256 `af395425e531b213e59f49290bb3101715557583cefb48d6efc7b02465d95f1a`.

Only the official benchmark artifact was used.

## Actual embedded split manifest

| Split | Samples |
|---|---:|
| Train | 4,000 |
| Validation | 535 |
| Test | 975 |
| Total | 5,510 |

These counts were read from the downloaded tortilla. They intentionally replace the generation script's requested target counts (4,000/1,000/2,000), which the published artifact does not reach. No expected-size constant is used.

`tortilla:id` is the saved sample ID. All 5,510 are unique. `roi_id`, `old_roi_id`, `equi_id`, and centroid groups are pairwise disjoint across splits.

## Optical input and masks

- Selected common input: Sentinel-2 optical only; B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12. B10 is absent from the benchmark sensor definition.
- Canonical `wavelengths_nm`: 443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190.
- Native benchmark raster: 12×512×512 at 10 m metadata.
- Model input: 12×224×224, bilinear image resizing.
- Effective model-grid spacing over the same footprint: approximately 22.86 m (10 m × 512/224).
- Normalization: pinned GEO-Bench-2 train statistics, channel-wise z-score.
- Class mapping: 0 clear; 1 thick cloud; 2 thin cloud; 3 cloud shadow.
- Ignore label: none in the official definition or observed masks.
- Mask resizing: nearest neighbor only.

GEO-BenchV2 0.9's CloudSEN12 class computes a normalized dictionary but returns the original unnormalized dictionary. The project adapter explicitly applies the class's already-instantiated official normalizer to correct that return-path defect without changing the statistics. Real normalized loader samples from every split were finite.

## Full integrity scan

Every one of the 5,510 image/mask pairs was checked:

- shape mismatches: 0;
- affine-transform mismatches: 0;
- CRS mismatches: 0;
- masks containing values outside {0,1,2,3}: 0;
- every mask contained clear pixels; classes 1, 2, and 3 occurred in 4,160, 1,875, and 3,772 sample masks respectively.

Loader tests on train/validation/test produced synchronized finite `12×224×224` images and `224×224` masks with only allowed values.

## Cross-split identity check

There are no duplicated sample IDs or ROI/equi-location groups across splits. Three Sentinel-2 product IDs do cross split boundaries:

- train/test: `S2A_MSIL1C_20190427T015701_N0207_R060_T53TPN_20190427T040043`;
- train/test: `S2A_MSIL1C_20190605T054641_N0207_R048_T45VWJ_20190605T093053`;
- train/validation: `S2A_MSIL1C_20190111T082311_N0207_R121_T34HCK_20190111T102302`.

The associated ROIs/centroids remain disjoint, so these are not duplicate samples, but they are same-product scene-source overlap and should be disclosed as a potential benchmark leakage concern rather than described as fully scene-disjoint.

No training or model inference was performed for CloudSEN12.
