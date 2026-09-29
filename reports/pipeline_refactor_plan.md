# Minimal classification pipeline refactor plan

## Inspection result

The canonical EuroSAT classification loop is already shared across experiments. Dataset loading, optimization, checkpoint selection, metrics, prediction export, and seed application do not need to be duplicated for another backbone.

Current configurability is uneven:

- **Dataset:** configurable through `data.*`; the fixed manifest path, RGB bands, image size, normalization, and loaders are already independent of the model.
- **Adaptation:** already normalized by `resolve_config()` from `model.adaptation_mode` to `model.freeze_backbone`, and the common optimizer/audit code works with any wrapper exposing `backbone`, `head`, and `freeze_backbone`.
- **Seed:** used consistently by the RNGs, workers, loaders, run IDs, and metadata. However, validation occurs late and coercion with `int(...)` can silently accept non-integer values such as floats.
- **Model:** `build_model()` is the remaining bottleneck. It constructs DOFA inline, embeds DOFA-specific wavelength handling, and the optional embedding exporter assumes DOFA's `forward_features` API. There is no Panopticon branch.
- **Output safety:** per-run directories are unique and refuse reuse, but the root-level `summary.json` is overwritten on each invocation.

The installed TorchGeo 0.8.1 provides the official `panopticon_vitb14` implementation and `Panopticon_Weights.VIT_BASE14`. Its forward input is `{"imgs": [B,C,H,W], "chn_ids": [B,C]}`; optical channel IDs are wavelengths in nanometres. This differs from DOFA, which receives wavelengths in micrometres through `forward_features(..., wave_list=...)`.

## Minimal changes

1. **Keep one training loop and introduce a common classifier contract.**
   - Retain `DOFARGBLinearProbe` for checkpoint compatibility.
   - Add a small `PanopticonClassifier` wrapper with the same public attributes: `backbone`, `head`, `freeze_backbone`, and `extract_features(images)`.
   - Extract the existing configurable classification-head construction into one helper used by both wrappers.
   - Add `extract_features()` to the DOFA wrapper, leaving its state-dict parameter names unchanged.

2. **Make spectral metadata explicit and unit-safe.**
   - Read `data.wavelengths` plus `data.wavelength_units` from YAML.
   - Convert configured wavelengths to micrometres for DOFA and nanometres for Panopticon.
   - Preserve the existing Sentinel-2 RGB values as a compatibility fallback for old configs/checkpoints, while adding explicit values to active DOFA configs and the Panopticon template.

3. **Add Panopticon to the existing model factory.**
   - Support `model.name: panopticon`, `model.size: base`, `model.pretrained`, and a configurable official weights enum (`model.weights: VIT_BASE14`).
   - Fail clearly for unsupported sizes, weight identifiers, missing TorchGeo support, or invalid channel metadata.
   - Do not download/load Panopticon during config-only validation or unit tests.

4. **Strengthen configuration validation without changing old protocols.**
   - Validate model names (`dofa`, `panopticon`, while retaining the existing `resnet18` compatibility path).
   - Validate `adaptation_mode` as `frozen` or `full_finetune`.
   - Require `seed`/`seeds` values to be actual integers, reject booleans/duplicates, and retain arbitrary integer values rather than prescribing 42/43/44.
   - Validate wavelength length against configured channels and supported units.

5. **Remove remaining model-specific evaluation branching.**
   - Add a generic classification embedding extractor that calls the wrapper contract.
   - Keep the old DOFA extractor name as a compatibility alias.
   - Centralize the model display name used in results and prediction manifests.

6. **Add configuration templates, not experiment constants in code.**
   - Add Panopticon frozen and full-finetune EuroSAT YAML files with unique output roots.
   - Keep optimizer/LR/epoch/head choices in YAML; the Python runner must not choose them based on model or adaptation mode.

7. **Guarantee new invocations do not overwrite old summaries.**
   - Continue using immutable unique run directories.
   - Write invocation summaries under a unique `summaries/` filename instead of replacing root `summary.json`.
   - Do not migrate, rename, or alter any existing outputs/checkpoints.

8. **Verification without full training.**
   - Add unit tests using fake backbones for both wrapper APIs, both adaptation modes, arbitrary seeds, validation failures, and unique summary paths.
   - Run configuration-only validation for DOFA and Panopticon configs.
   - Run the focused unit test suite only; do not instantiate pretrained Panopticon weights or launch a training epoch.

## Files expected to change

- `scripts/run_experiments.py`
- `scripts/experiment_manager.py`
- `scripts/prediction_export.py`
- `scripts/export_checkpoint_predictions.py`
- `tests/test_experiment_configuration.py`
- `tests/test_experiment_manager.py`
- active DOFA YAML configs (explicit spectral metadata only)
- new Panopticon YAML configs
- this plan

No existing experiment output, result, or checkpoint is to be modified.
