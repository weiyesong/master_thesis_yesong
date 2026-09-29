# Panopticon real-model backend validation

Validation date: 2026-08-09 UTC  
Outcome: **VALIDATED for real pretrained EuroSAT RGB classification forward/backward smoke use**  
Code snapshot used for the final smoke: `sha256:b6916bec11dc47e02d6a416704a212986f34aface43a425f8e382c2c934b2553`

This is a backend validation, not an experiment result. No epoch loop, optimizer step, checkpoint creation, EuroSAT training, validation/test inference, or Temperature Scaling was run. The only model computation was a two-sample real forward/backward smoke for each adaptation mode.

## 1. Final verdict

| Requirement | Result | Evidence |
|---|---|---|
| Official pretrained weights load | PASS | TorchGeo `Panopticon_Weights.VIT_BASE14`; pinned official URL; real `panopticon_vitb14(weights=..., img_size=224)` construction completed through strict state-dict loading. |
| RGB channel IDs | PASS after metadata precision correction | Input order `B04, B03, B02`; IDs `[664.6300422881802, 559.5987534818435, 492.9971095687347]`. |
| Wavelength units | PASS | Config declares `nanometers`; wrapper passes an actual `[B,3]` float tensor in nm to `chn_ids`. No µm conversion occurs on the Panopticon path. |
| Expected resolution | PASS | Config/input/model patch embed all use `224×224`; patch size 14 gives a `16×16` patch grid. |
| Normalization/dynamic range | PASS after model-specific correction | Dataset-level channelwise standardization now uses statistics computed only from the fixed train split; no val/calibration/test pixels were used. |
| Feature dimension | PASS | Real outputs are `[2,768]`, matching `backbone.model.num_features == 768` and the official API. |
| Finite forward | PASS | Both modes produced finite features, logits, and cross-entropy loss. |
| Frozen gradient policy | PASS | All 98,120,448 backbone parameters are frozen; 0 backbone gradient tensors; head weight and bias receive non-zero gradients. |
| Full-finetune gradient policy | PASS | All 179 intended optical-path backbone tensors receive finite, non-zero gradients; only three explicitly declared SAR-only embeddings are frozen. |

## 2. Official model and weight provenance

- Backend: TorchGeo `0.8.1`, `torchgeo.models.panopticon_vitb14`.
- Weight enum: `Panopticon_Weights.VIT_BASE14`.
- Official pinned URL: `https://hf.co/lewaldm/panopticon/resolve/c8c2bb9555819e8b2bcedf5b3b00e3bf531554e7/panopticon_vitb14_teacher.pth`.
- Downloaded file: `/home/yesong/.cache/torch/hub/checkpoints/panopticon_vitb14_teacher.pth`.
- Size: 395,965,930 bytes.
- SHA-256: `55024f411a7f383ed1a646d9b833b65683a4443846603fc7d37591d5afe9d26e`.
- TorchGeo weight transform: `Identity()`. Normalization is therefore an external dataloader responsibility.
- TorchGeo's factory removes the pretraining-only checkpoint `mask_token`, resizes positional embeddings for the requested resolution, then calls strict state-dict loading and asserts no missing/unexpected keys. The real factory returned successfully for both modes; recorded missing keys `[]`, unexpected keys `[]`.

The [official Panopticon README](https://github.com/Panopticon-FM/panopticon#using-panopticon) specifies a 224×224 RGB example, nm channel IDs, a 768-dimensional image representation, and standard-normal input normalization. The pretrained-weight URL is pinned to the upstream revision embedded in TorchGeo rather than an unversioned latest file.

## 3. Spectral metadata

Final EuroSAT RGB mapping:

| Tensor channel | Sentinel-2 band | Meaning | Panopticon channel ID |
|---:|---|---|---:|
| 0 | B04 | Red | 664.6300422881802 nm |
| 1 | B03 | Green | 559.5987534818435 nm |
| 2 | B02 | Blue | 492.9971095687347 nm |

These values come from the [official Geobreeze/Panopticon Sentinel-2 sensor metadata](https://github.com/geobreeze/geobreeze/blob/main/geobreeze/datasets/metadata/sensors/sentinel2.yaml), which models the Sentinel-2A MSI spectral response functions. They also agree, to rounding, with the official Panopticon README example `[664, 559, 493]` nm. The prior `[665,560,490]` values were defensible generic Sentinel-2 nominal centers, but were replaced by the model ecosystem's exact metadata to eliminate ambiguity.

`configured_wavelengths(config, "nanometers")` returned the values above. `PanopticonClassifier.extract_features()` constructed a per-batch `chn_ids` tensor with shape `[2,3]` in the same R/G/B order and passed `{"imgs": image, "chn_ids": channel_ids}` to the real backbone.

## 4. Resolution and feature contract

- Actual input: `[2,3,224,224]`, `torch.float32`, finite.
- Model patch-embed `img_size`: `[224,224]`.
- Patch size: 14; actual configured grid: `[16,16]`.
- `backbone.model.num_features`: 768.
- Frozen real feature output: `[2,768]`, finite, observed range `[-7.232326, 7.430559]`.
- Full-finetune real feature output: `[2,768]`, finite, observed range `[-7.232326, 7.430559]` before any parameter update.
- Real classifier logits: `[2,10]`, finite in both modes.

This matches the official image-level classification contract. No mocked backbone was involved in these measurements.

## 5. Normalization and dynamic-range validation

### 5.1 Evidence and correction

Panopticon's official evaluation path normalizes each dataset using its own channel statistics before model transforms; the [official GeoBench adapter](https://github.com/Panopticon-FM/panopticon/blob/main/dinov2/data/datasets/geobench.py#L67-L72) loads dataset normalization statistics and [applies channelwise `(x-mean)/std`](https://github.com/Panopticon-FM/panopticon/blob/main/dinov2/data/datasets/geobench.py#L136-L137). The README independently recommends standard-normal inputs.

The previous Panopticon configs reused broad Sentinel-2 constants from the DOFA path. On this thesis's fixed EuroSAT train split those constants would have produced population means `[-0.2080,-0.1256,-0.1121]` and standard deviations `[0.6106,0.5453,0.5031]`, not approximately zero/one. They were therefore not retained for Panopticon.

The final statistics were computed over all 18,866 fixed-manifest **train** images, 77,275,136 pixels per channel, before resizing:

| Band order | Mean (raw DN) | Population std (raw DN) |
|---|---:|---:|
| B04 / Red | 936.085209866211 | 589.387623763833 |
| B03 / Green | 1031.3388562784282 | 388.3087839027959 |
| B02 / Blue | 1111.4795678522003 | 327.14241012761033 |

The raw train range was 0–28,000 DN for each visible band. Channelwise standardization makes the full train population exactly zero mean/unit population variance by construction; validation, calibration, and test data do not influence the transform. No clipping was added because the official pipeline standardizes without specifying clipping. The actual two-image smoke batch was finite, with overall range `[-1.620166,5.029929]`; its per-channel means/stds need not equal zero/one because it contains only two images.

Both Panopticon configs now record the method, train-only source manifest, raw units, sample count, pixel count, means, and standard deviations. `EuroSATClassificationTransform` consumes these explicit values. DOFA configs and historical DOFA preprocessing were not changed.

## 6. Real frozen adaptation smoke

Configuration: `configs/eurosat_panopticon_frozen_baseline.yaml`; batch size 2; real normalized EuroSAT train samples; one forward and one loss backward; no optimizer and no update.

| Check | Observed |
|---|---:|
| Backbone parameters | 98,120,448 |
| Backbone trainable parameters | 0 |
| Backbone parameter tensors | 182 |
| Backbone tensors with gradients | 0 |
| Head parameters | 7,690 |
| Head trainable tensors with non-zero gradients | `1.weight`, `1.bias` |
| Feature shape / finite | `[2,768]` / yes |
| Logit shape / finite | `[2,10]` / yes |
| Cross-entropy finite | yes (`2.471732`) |
| Peak allocated GPU memory | 510,270,464 bytes |

The wrapper was in training mode for the head, while the frozen backbone was explicitly held in eval mode. Thus frozen adaptation neither requests nor accumulates backbone gradients and does not update backbone train/eval-sensitive state.

## 7. Real full-finetune adaptation smoke

Configuration: `configs/eurosat_panopticon_full_finetune.yaml`; same two real samples; one forward and one loss backward; no optimizer and no update.

| Check | Observed |
|---|---:|
| Backbone parameters | 98,120,448 |
| Intended trainable backbone parameters | 98,115,840 |
| Backbone parameter tensors | 182 |
| Intended trainable backbone tensors | 179 |
| Tensors with present gradients | 179/179 |
| Tensors with finite gradients | 179/179 |
| Tensors with non-zero gradients | 179/179 |
| Missing / non-finite / zero intended gradients | `[] / [] / []` |
| Head trainable tensors with non-zero gradients | `1.weight`, `1.bias` |
| Feature shape / finite | `[2,768]` / yes |
| Logit shape / finite | `[2,10]` / yes |
| Cross-entropy finite | yes (`2.581976`) |
| Peak allocated GPU memory | 919,853,568 bytes |

The first real backward showed that three SAR-only embeddings receive structurally present but identically zero gradients for positive optical wavelength IDs:

- `model.patch_embed.chnfus.chnemb.embed_transmit`
- `model.patch_embed.chnfus.chnemb.embed_receive`
- `model.patch_embed.chnfus.chnemb.embed_orbit`

Panopticon selects these embeddings only for negative SAR channel IDs. They are not part of the EuroSAT RGB computation graph and cannot be learned from this dataset. The full-finetune config now explicitly lists them as expected frozen parameters, and the factory validates those names before freezing them. All remaining backbone parameters—including patchification, optical channel embedding path, attention blocks, MLPs, layer scales, and final normalization—remain trainable and received finite, non-zero gradients in the final smoke.

## 8. Code and checks changed

- Preserved the existing `PanopticonClassifier` input contract and official TorchGeo factory.
- Added explicit, validated per-dataset normalization mean/std support to `EuroSATClassificationTransform`.
- Added config validation requiring positive, channel-aligned, train-split-only normalization statistics.
- Updated both Panopticon EuroSAT configs with exact official Sentinel-2 wavelength metadata and train-only normalization statistics.
- Added explicit freezing/validation of configured inactive Panopticon parameters for optical-only full fine-tuning.
- Added unit coverage for explicit normalization, leakage guard, and inactive-parameter freezing.
- Unit result: 23 targeted tests passed (`test_experiment_manager`, `test_manifest_dataloader`, `test_experiment_configuration`).

## 9. Boundary of this validation

The backend is now genuinely validated for real pretrained model construction and the requested tiny gradient smoke. This does **not** establish EuroSAT accuracy, calibration, convergence, multi-seed reproducibility, final batch-size memory requirements, or training stability. Those claims require future experiments, which were intentionally not launched here.
