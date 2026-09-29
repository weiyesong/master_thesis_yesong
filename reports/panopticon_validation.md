# Real Panopticon model validation

Validation date: 2026-08-10  
Outcome: **VALIDATED for frozen and full-finetune classification backends**  
Scope: real official checkpoint, real EuroSAT and So2Sat inputs, tiny forward/backward smoke tests only. No optimizer step, epoch, or training run was performed.

## Official model and checkpoint

- Factory: TorchGeo 0.8.1 `torchgeo.models.panopticon_vitb14`.
- Weights enum: `Panopticon_Weights.VIT_BASE14`.
- Official checkpoint URL: [Panopticon teacher weights](https://hf.co/lewaldm/panopticon/resolve/c8c2bb9555819e8b2bcedf5b3b00e3bf531554e7/panopticon_vitb14_teacher.pth).
- Cache path: `/home/yesong/.cache/torch/hub/checkpoints/panopticon_vitb14_teacher.pth`.
- Size: 395,965,930 bytes.
- SHA256: `55024f411a7f383ed1a646d9b833b65683a4443846603fc7d37591d5afe9d26e`.
- Load result: strict upstream load succeeded; missing keys `[]`, unexpected keys `[]`.
- Official implementation/reference: [TorchGeo 0.8.1 Panopticon source](https://github.com/torchgeo/torchgeo/blob/v0.8.1/torchgeo/models/panopticon.py) and [Panopticon project](https://github.com/Panopticon-FM/panopticon).

## Input contract

| Check | Evidence | Result |
|---|---|---|
| Channel IDs | TorchGeo documents positive optical IDs as wavelengths in **nanometers**. The adapter passed canonical nm values unchanged. | PASS |
| RGB order | EuroSAT input was B04/B03/B02 with IDs 664.6300/559.5988/492.9971 nm. | PASS |
| Multispectral order | So2Sat input was B02/B03/B04/B05/B06/B07/B08/B8A/B11/B12 with the corresponding 10 nm IDs from config. | PASS |
| Spatial size | Real inputs were 224×224; initialized patch grid was 16×16 for patch size 14. The published model recommends 224. | PASS |
| Normalization | EuroSAT uses fixed channel-wise z-score statistics computed from its training split in raw Sentinel-2 L2A DN. So2Sat uses the pinned GEO-Bench-2 training statistics. This matches the Panopticon project's standard-normal input convention and prevents test leakage. | PASS |
| Dynamic range | Real normalized RGB batch: finite, min −1.6202, max 5.0299, mean 0.6271, std 1.0028. Real normalized 10-band sample: finite, min −2.0665, max −0.3889, mean −1.4136, std 0.5841. Values are plausible z-scores; no clipping or unscientific 0–255/0–1 mixing was observed. | PASS |
| Feature dimension | Backbone returned `[B, 768]`, matching ViT-Base `num_features=768`. | PASS |

The multispectral test used official So2Sat sample `id_376488`. Its feature tensor was `[1, 768]`, logits were `[1, 17]`, and both were entirely finite. The RGB test used two real EuroSAT training samples; features were `[2, 768]`, logits were `[2, 10]`, and both were entirely finite.

## Adaptation smoke tests

Executed on an NVIDIA GeForce RTX 3090 with PyTorch 2.5.1+cu124. Each mode performed one real forward pass, one cross-entropy backward pass, and no parameter update.

### Frozen

- Backbone parameters: 98,120,448.
- Trainable backbone parameters: **0**.
- Backbone gradient tensors after backward: **0**.
- Trainable head parameters: 7,690; both trainable head tensors received finite gradients.
- Result: **PASS — the backbone is fully frozen and receives no gradients.**

### Full fine-tuning

- Trainable backbone parameters: **98,115,840** across 179 tensors.
- Trainable backbone tensors receiving finite gradients: **179/179**.
- Trainable backbone tensors without gradients: `[]`.
- The three deliberately frozen SAR-only embeddings were `embed_transmit`, `embed_receive`, and `embed_orbit`; none received a gradient. They are outside the positive optical-wavelength computation path and are explicitly recorded in the config.
- Both trainable head tensors received gradients.
- Result: **PASS — every intended trainable backbone tensor receives a finite gradient.**

## Conclusion

The real official Panopticon ViT-Base/14 checkpoint loads cleanly, the project adapter supplies correct channel metadata and units, the configured preprocessing matches the upstream input convention, feature dimensions and values are valid, and frozen/full-finetune gradient behavior is correct. Panopticon is therefore a genuinely validated classification backend. This validation does not constitute a trained EuroSAT result.
