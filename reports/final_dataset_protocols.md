# Final dataset protocols and provenance audit

Audit date: 2026-08-10 UTC  
Scope: EuroSAT, TreeSatAI, CloudSEN12, and SpaceNet7 only  
Outcome: **four final dataset artifacts and their current manifests were verified; material protocol caveats are retained below**

No data were downloaded, altered, relabelled, or repartitioned during this audit. No model training or model inference was performed. For the three GEO-Bench-2 datasets, the downloaded official benchmark artifact—not the raw upstream dataset—is the final thesis dataset.

## Audit basis and global convention

GEO-Bench-2 is pinned as `GeoBenchV2` 0.9 at Git commit `fd9d0b664e6fb0faba54636bdff4906634debd4b`; the installation and artifact policy are recorded in `environment/geobench2.lock.yaml`. The three Hugging Face download URLs use `resolve/main`, so the URL alone is not an immutable version. Exact artifact identity is instead fixed by the local byte size and SHA256 below. Each hash matches the expected hash in the pinned GEO-Bench-2 dataset class.

Optical wavelength metadata are canonicalized in nanometers in dataset/config metadata. DOFA converts nm to µm at its adapter boundary; Panopticon receives nm unchanged. The wavelength values below are center-wavelength metadata supplied by the relevant registry/config, not values inferred from image pixels.

| Task | Dataset | Final artifact status | Split status | Leakage-control verdict |
|---|---|---|---|---|
| Classification | EuroSAT | Verified TorchGeo all-band archive | Verified custom spatial 70/10/10/10 manifest | Strong local footprint control; source-scene separation unprovable |
| Classification, multi-label | TreeSatAI | Verified official GEO-Bench-2 Tortilla | Verified embedded 4,000/1,000/2,000 split | Official geographic rule and disjoint identities; block IDs absent from delivered manifest |
| Segmentation | CloudSEN12 | Verified official GEO-Bench-2 Tortilla | Verified embedded 4,000/535/975 split | ROI/location identities disjoint; three Sentinel-2 products cross splits |
| Segmentation | SpaceNet7 | Verified official GEO-Bench-2 Tortilla | Verified embedded 3,500/652/1,152 split | Whole AOIs and their monthly source images are disjoint |

## 1. EuroSAT classification

### Exact provenance

| Field | Verified value |
|---|---|
| Dataset implementation | TorchGeo 0.8.1 `EuroSAT`, all-band release |
| Pinned source URL revision | `https://hf.co/datasets/torchgeo/eurosat/resolve/1ce6f1bfb56db63fd91b6ecc466ea67f2509774c/EuroSATallBands.zip` |
| Local artifact | `/workspace/data/EuroSATallBands.zip` |
| Artifact size | 2,067,725,275 bytes |
| Artifact MD5 | `5ac12b3b2557aa56e1826e981e8e200e`, equal to TorchGeo's expected MD5 |
| Artifact SHA256 | `751f070f9bffa2eed48b24ca2dd0b02959280c08837e8c9a5532a67ba611df59` |
| Final split manifest | `splits/eurosat_70_10_10_10_spatial20m/eurosat_splits.csv` |
| Manifest SHA256 | `c5cadc7936394f0678307f7db25c55a2dc19f73ddb93891b772c16221e8abf22` |
| Equivalent JSON SHA256 | `392faea118debab19ae7dab6adbacc3a12474cd6d36e237fecccb231e20b2b8c` |
| Split validation report SHA256 | `e692e850f2bb53e56b92820229c57d683865cbd84e03ecb1ae6d3a355259e910` |
| Spatial audit report SHA256 | `32eae73cfc859f2978953fe175ecdfd03e7cc8fd26c1e8b59924db8850681abd` |

### Split manifest and identities

| Split | Samples |
|---|---:|
| Train | 18,866 |
| Validation | 2,707 |
| Calibration | 2,713 |
| Test | 2,714 |
| **Total** | **27,000** |

The final thesis split is the project-generated, class-stratified, spatial-grouped 70/10/10/10 manifest with generation seed `20260803`; it is not TorchGeo's original 60/20/20 split. The original TorchGeo split assignment is retained in the `official_split` column for provenance.

The manifest contains `sample_id`, `dataset_index`, relative file path, label/index, final and official split, per-file SHA256, CRS, bounds, center, and spatial group. All 27,000 sample IDs, paths, and content hashes are unique; there are no missing paths, no duplicate IDs, no cross-split content hashes, and no cross-split spatial groups. Sample ID is the stable image stem, for example `AnnualCrop_1`.

### Channels, wavelengths, geometry, and normalization

- Archive band order: B01, B02, B03, B04, B05, B06, B07, B08, B09, B10, B11, B12, B8A.
- Final common thesis input: RGB in B04, B03, B02 order.
- Native images: 13-band uint16 GeoTIFF, 64×64 pixels, approximately 10 m pixels. A distributed 100-file header audit confirmed this shape/type; the manifest covers ten UTM CRSs.
- Model input: the full 64×64 footprint resized to 224×224.
- Exact Panopticon/ecosystem metadata: B04 `664.6300422881802` nm, B03 `559.5987534818435` nm, B02 `492.9971095687347` nm.
- The frozen historical DOFA runs record the rounded nominal values `[665, 560, 490]` nm. These are scientifically compatible nominal centers but are not byte-identical to the more precise Panopticon metadata. Historical metadata must remain unchanged; future cross-model configs should explicitly state which registry precision they use.

Normalization is model/run-specific rather than a single property embedded in the archive:

- The six frozen final DOFA runs use project constants, in RGB order, mean `[1136.89, 1120.77, 1184.39]` and standard deviation `[965.23, 712.12, 650.20]` in raw Sentinel-2 DN. Their values are recorded in code and run configs, but the repository does **not** preserve a primary external statistical provenance or a reproducible derivation manifest for these constants. They remain part of the immutable historical protocol.
- Current Panopticon configs use statistics recomputed only from the 18,866 final train images before resizing: mean `[936.085209866211, 1031.3388562784282, 1111.4795678522003]`, population standard deviation `[589.387623763833, 388.3087839027959, 327.14241012761033]`, based on 77,275,136 pixels per channel. Validation, calibration, and test pixels were excluded.

### Label mapping

The verified index order is: 0 AnnualCrop; 1 Forest; 2 HerbaceousVegetation; 3 Highway; 4 Industrial; 5 Pasture; 6 PermanentCrop; 7 Residential; 8 River; 9 SeaLake. All ten classes occur in every final split.

### Spatial and temporal leakage controls

- No byte-identical sample, exact center, overlapping footprint, or adjacency edge within 20 m crosses the final splits.
- The audit found 754 overlapping-footprint edges and 19,272 adjacency edges within 20 m, all kept within a single split by 15,626 spatial groups.
- EuroSAT files do not expose original Sentinel-2 acquisition/product identifiers. Consequently, broader same-acquisition/source-scene leakage cannot be conclusively excluded. This is the principal remaining leakage limitation.
- This is a static classification dataset; no temporal sequences or temporal aggregation are used.

**Protocol verdict: VERIFIED with a documented source-scene limitation and a documented historical normalization/wavelength-precision difference.**

## 2. TreeSatAI multi-label classification

### Exact provenance

| Field | Verified value |
|---|---|
| Benchmark code | GEO-Bench-2 / `GeoBenchV2` 0.9 at `fd9d0b664e6fb0faba54636bdff4906634debd4b` |
| Official repository | `aialliance/treesatai` |
| Final local artifact | `/workspace/datasets/treesatai/geobench_treesatai.tortilla` |
| Artifact size | 1,771,492,601 bytes |
| Artifact SHA256 | `0ddb8068720242ad4f5931ea91f3459ed695ad490bbaa48905afe72dd9623aee` |
| Extracted final manifest | `reports/dataset_manifests/treesatai_actual_manifest.csv` |
| Manifest SHA256 | `b958ada672e7aeaa29652731af814834e9dae4fbe1c53d2edc92802ff18eae54` |

No raw TreeSatAI release was substituted.

### Split manifest and identities

| Split | Samples |
|---|---:|
| Train | 4,000 |
| Validation | 1,000 |
| Test | 2,000 |
| **Total** | **7,000** |

Counts come from the downloaded Tortilla's embedded manifest, not a hard-coded expectation. `tortilla:id` is the final sample ID (`sample_0`, etc.); all 7,000 IDs are unique. The original filename identity is retained as `source_path`, and the advertised time-series path is retained as `advertised_ts_path`; both are unique and pairwise disjoint across splits. Coordinates and labels are also saved in the extracted manifest.

The official generation code uses a balanced 10×10 geographic checkerboard with random state 42, followed by split-preserving subsetting with random state 24. The published artifact does not retain checkerboard block IDs, so that rule is source-verifiable but cannot be reconstructed solely from the final manifest.

### Channels, wavelengths, geometry, and normalization

- Official static Sentinel-2 order: B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09.
- Canonical center wavelengths in the same order: `[490, 560, 665, 842, 705, 740, 783, 865, 1610, 2190, 443, 945]` nm, from the pinned GEO-Bench-2 TreeSatAI/Sentinel-2 band registry.
- Delivered static S2 raster: 12×304×304 uint16 with 10 m metadata; the full footprint is resized bilinearly to 224×224 (effective model-grid spacing about 13.57 m).
- Normalization source: the exact channelwise z-score constants in the pinned GEO-Bench-2 class, configured as benchmark train statistics. In band order above, means are `[245.31068420410156, 387.63568115234375, 248.4667205810547, 2825.93603515625, 625.9300537109375, 2118.83740234375, 2709.37890625, 2982.208740234375, 1316.7186279296875, 594.203369140625, 265.8070068359375, 2962.182373046875]`; standard deviations are `[117.73491668701172, 130.0995635986328, 129.66375732421875, 756.8175659179688, 191.35238647460938, 517.2822265625, 691.1488037109375, 754.9419555664062, 411.339111328125, 234.48863220214844, 125.9928207397461, 674.169189453125]`.
- The constants were source-checked against the installed pinned class; this audit did not recompute them from all raw pixels.

### Label mapping

This is a **15-class multi-label** dataset whose targets are 15-dimensional multi-hot vectors: 0 Abies; 1 Acer; 2 Alnus; 3 Betula; 4 Cleared; 5 Fagus; 6 Fraxinus; 7 Larix; 8 Picea; 9 Pinus; 10 Populus; 11 Prunus; 12 Pseudotsuga; 13 Quercus; 14 Tilia. Observed label cardinality ranges from one to six labels per sample. The class code/artifact uses 15 labels even though an upstream dataset docstring says “13-class”; the artifact and emitted 15-label vectors are authoritative.

### Spatial and temporal leakage controls

- Final sample IDs, original source paths, advertised time-series paths, and centroid coordinate pairs are pairwise disjoint across train/validation/test.
- The official checkerboard rule is explicitly geographic, but its block assignments are not embedded. Therefore the artifact verifies identity/coordinate separation and the source verifies the algorithm; the exact spatial groups cannot be independently replayed from the artifact alone.
- The Tortilla packages one static Sentinel-2 observation per sample. Although all 7,000 rows advertise external HDF5 time-series paths, none of those HDF5 files is present in the official artifact, and the official repository does not provide them alongside it.
- The final thesis dataset must therefore be described as **T=1 static TreeSatAI**, not as a validated multi-temporal dataset. The retained acquisition years span 2011 and 2013–2020, but no sample is temporally aggregated from multiple observations.

**Protocol verdict: OFFICIAL STATIC ARTIFACT VERIFIED. A genuine multi-timestamp protocol is unavailable from the final artifact, and TreeSatAI requires multi-label—not multiclass—training/evaluation.**

## 3. CloudSEN12 segmentation

### Exact provenance

| Field | Verified value |
|---|---|
| Benchmark code | GEO-Bench-2 / `GeoBenchV2` 0.9 at `fd9d0b664e6fb0faba54636bdff4906634debd4b` |
| Official repository | `aialliance/cloudsen12` |
| Final local artifact | `/workspace/datasets/cloudsen12/geobench_cloudsen12.tortilla` |
| Artifact size | 9,944,531,349 bytes |
| Artifact SHA256 | `16b3c03d7b15cf42f6ef0cee6d453b6ad8ebbe7744674c4b58657511f7f5d0c0` |
| Extracted final manifest | `reports/dataset_manifests/cloudsen12_actual_manifest.csv` |
| Manifest SHA256 | `af395425e531b213e59f49290bb3101715557583cefb48d6efc7b02465d95f1a` |

Only the official downloaded benchmark artifact is in scope.

### Split manifest and identities

| Split | Samples |
|---|---:|
| Train | 4,000 |
| Validation | 535 |
| Test | 975 |
| **Total** | **5,510** |

These are the actual embedded counts; they deliberately supersede the generation script's nominal requested sizes. `tortilla:id` is the final sample ID (for example `ROI_04136__20180913T081601_20180913T082753_T36SYJ`). All 5,510 IDs are unique. The extracted manifest also records current and old ROI IDs, equi-grid ID, Sentinel-2 product ID, acquisition time, coordinate, and label type.

### Channels, wavelengths, geometry, normalization, and labels

- Common input is Sentinel-2 optical only in B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12 order; B10 is not part of the benchmark sensor definition.
- Canonical center wavelengths: `[443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190]` nm from the pinned GEO-Bench-2 registry.
- Native image/mask: 12×512×512 at 10 m metadata. Images are resized bilinearly and masks by nearest neighbor to 224×224, preserving the full footprint (effective grid spacing about 22.86 m).
- Class mapping: 0 clear; 1 thick cloud; 2 thin cloud; 3 cloud shadow. There is no ignore label in the official definition or the observed masks.
- Normalization source: pinned GEO-Bench-2 channelwise z-score constants, in the band order above. Means are `[2030.244384765625, 2074.817138671875, 2209.807373046875, 2247.927490234375, 2589.593505859375, 3103.521240234375, 3277.909423828125, 3331.6318359375, 3377.544677734375, 4038.193115234375, 2448.748046875, 1907.728515625]`; standard deviations are `[2723.43603515625, 2691.302734375, 2539.91357421875, 2538.520751953125, 2504.328369140625, 2241.74462890625, 2145.667724609375, 2176.997802734375, 2066.763671875, 3083.179931640625, 1595.065185546875, 1474.11767578125]`.
- The values were source-checked against the pinned class rather than recomputed. GEO-Bench-2 0.9 computes a normalized dictionary but returns the unnormalized dictionary; the project adapter explicitly applies that already-instantiated official normalizer.

All 5,510 image/mask pairs were previously exhaustively checked: zero shape, affine, or CRS mismatches and zero masks outside `{0,1,2,3}`.

### Spatial and temporal leakage controls

- Sample ID, `roi_id`, legacy ROI ID, `equi_id`, and centroid coordinate-pair sets are pairwise disjoint across splits.
- Three Sentinel-2 product IDs cross split boundaries despite the disjoint ROIs:
  - train/test: `S2A_MSIL1C_20190427T015701_N0207_R060_T53TPN_20190427T040043`;
  - train/test: `S2A_MSIL1C_20190605T054641_N0207_R048_T45VWJ_20190605T093053`;
  - train/validation: `S2A_MSIL1C_20190111T082311_N0207_R121_T34HCK_20190111T102302`.
- These are not duplicate samples or locations, but they are same-acquisition scene-source overlap and must be disclosed as a benchmark leakage risk.
- Each sample is a single acquisition; no temporal sequence modeling or temporal aggregation is used.

**Protocol verdict: OFFICIAL ARTIFACT, SPLITS, LABELS, AND MASK INTEGRITY VERIFIED, with three disclosed cross-split product-source overlaps.**

## 4. SpaceNet7 static building segmentation

### Exact provenance

| Field | Verified value |
|---|---|
| Benchmark code | GEO-Bench-2 / `GeoBenchV2` 0.9 at `fd9d0b664e6fb0faba54636bdff4906634debd4b` |
| Official repository | `aialliance/spacenet7` |
| Final local artifact | `/workspace/datasets/spacenet7/geobench_spacenet7.tortilla` |
| Artifact size | 3,056,268,185 bytes |
| Artifact SHA256 | `f202abe270b729f7f2651de64cb5c6b41c5f9915109ec12b6c467afa2abcb5b6` |
| Extracted final manifest | `reports/dataset_manifests/spacenet7_actual_manifest.csv` |
| Manifest SHA256 | `ef5b1d89883d1cdee9fca81c2c37989f661d93fd21bdfda6dc333204aeefa5ce` |

No raw SpaceNet7 release is substituted for this artifact.

### Split manifest and identities

| Split | Samples | AOIs |
|---|---:|---:|
| Train | 3,500 | 41 |
| Validation | 652 | 7 |
| Test | 1,152 | 12 |
| **Total** | **5,304** | **60** |

Counts and AOIs are read from the embedded artifact. The final sample ID is `tortilla:id` (`sample_0`, etc.); original `patch_id` is also retained. All 5,304 sample IDs and patch IDs are unique. The extracted manifest additionally records AOI, source image/mask, year, month, acquisition time, and coordinate.

The official generator assigns whole AOIs to splits and creates non-overlapping 512×512 patches. AOI, source-image, source-mask, patch-ID, and centroid coordinate-pair sets are pairwise disjoint across train/validation/test.

### Channels, wavelengths, geometry, normalization, and labels

- Planet RGB order: red, green, blue.
- Canonical metadata: `[665, 560, 490]` nm from GEO-Bench-2's generic RGB sensor registry. These are generic RGB center proxies, not a product-specific Planet spectral-response-function characterization; this limitation should remain explicit in model metadata.
- Native patch: 3×512×512; observed pixel size approximately 4.777314267 m. The full patch is resized bilinearly to 224×224 (effective grid spacing about 10.92 m); masks use nearest-neighbor resizing.
- Normalization source: pinned GEO-Bench-2 channelwise z-score constants. RGB means are `[116.94474029541016, 103.55889129638672, 76.77427673339844]`; standard deviations are `[61.655845642089844, 49.64897537231445, 45.88066864013672]`. They were source-checked against the pinned class, not independently recomputed.
- The generated raw mask is binary 0=no building, 1=building. GEO-Bench-2 0.9 adds one during loading, producing 1=no building and 2=building while declaring `("background", "no-building", "building")`; declared class 0 is unreachable in this artifact.
- The thesis adapter makes the binary task explicit: official 1 → thesis 0 background/no building; official 2 → thesis 1 building; any official 0 → ignore 255. No official 0 was observed.

All 5,304 image/mask pairs were previously exhaustively checked: zero shape, affine, or CRS mismatches and zero raw masks outside `{0,1}`. Of these, 5,175 contain building pixels.

### Spatial and temporal leakage controls

- Whole-AOI assignment is the primary leakage barrier. No AOI, monthly source image/mask, patch, or coordinate pair crosses splits.
- Observations span 2017–2020 and all months. Each month/patch remains an independent static sample; the thesis protocol must not group them into temporal sequences.
- Because all months from one AOI remain in one split, repeated-location temporal observations cannot leak across train/validation/test under the downloaded manifest.

**Protocol verdict: OFFICIAL STATIC ARTIFACT, AOI-DISJOINT SPLITS, BINARY LABEL REMAP, AND MASK INTEGRITY VERIFIED.**

## Material inconsistencies and limitations to retain in thesis reporting

1. **EuroSAT source-scene identity is unavailable.** Exact, overlap, and 20 m adjacency leakage are controlled, but same Sentinel-2 product/acquisition separation cannot be proven.
2. **EuroSAT preprocessing provenance differs by historical backend.** Frozen DOFA runs use repository constants without a preserved derivation record; Panopticon uses reproducible final-train-only statistics. Historical runs must not be rewritten, and comparisons must disclose this preprocessing difference.
3. **TreeSatAI is static in the official downloaded artifact.** Its external HDF5 time-series paths are unresolved, and its actual task is 15-class multi-label despite an upstream “13-class” docstring.
4. **CloudSEN12 is location-disjoint but not fully product-disjoint.** Three Sentinel-2 acquisition product IDs cross final split boundaries.
5. **SpaceNet7 requires an explicit label remap.** GEO-Bench-2's emitted labels are offset, and its class 0 is unreachable; the thesis binary mapping above is the validated contract.

These findings do not require dataset replacement or model training. They define the final protocol boundaries and the disclosures required when results are interpreted.
