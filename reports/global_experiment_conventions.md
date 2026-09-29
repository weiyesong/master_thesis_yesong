# Global experiment conventions

Status: **PINNED** on 2026-08-10. These rules apply to future experiments. Historical resolved configs and metadata are not rewritten.

## Benchmark source

- Package: `GeoBenchV2` 0.9.
- Repository: [The AI Alliance / GEO-Bench-2](https://github.com/The-AI-Alliance/GEO-Bench-2/tree/fd9d0b664e6fb0faba54636bdff4906634debd4b).
- Pinned commit: `fd9d0b664e6fb0faba54636bdff4906634debd4b`.
- Environment lock: `environment/geobench2.lock.yaml`.
- The installed distribution's `direct_url.json` records the same requested revision and resolved commit.
- Thesis datasets must use official GEO-Bench-2 artifacts. The upstream project states that benchmark submissions use the official versions hosted on Hugging Face; raw original releases are not interchangeable with those artifacts.
- Split sizes are observations from downloaded embedded manifests. They are never protocol constants in configs or code.

## Spectral invariant

The only canonical dataset field is `data.wavelengths_nm`. It contains optical center wavelengths in nanometers and must match `data.channels` in number and order.

| Consumer | Conversion | Saved units | Asserted optical scale |
|---|---:|---|---:|
| Dataset/config | none | nm | 300–2500 nm |
| DOFA adapter | divide by 1000 | µm | 0.3–2.5 µm |
| Panopticon adapter | identity | nm | 300–2500 nm |

For every new resolved run config, `data.wavelengths_nm` preserves the canonical dataset metadata while each DOFA/Panopticon experiment records `model.actual_wavelengths` and `model.actual_wavelength_units`. Resolution rejects legacy ambiguous fields, channel-count mismatches, and micrometer-scale values mistakenly placed in `wavelengths_nm`. Model construction rechecks that saved adapter values agree with the canonical values.

## Verification

- All active DOFA/Panopticon EuroSAT and So2Sat configs resolve under the invariant.
- The So2Sat documented-size assertion was removed. The loader verifies embedded split labels, unique IDs, and pairwise split disjointness from the installed artifact.
- `python -m unittest discover -s tests -v`: 35/35 passed after pinning.
- Final verification after dataset integration: 36/36 passed. The content-addressed snapshot mechanism resolved the current future-run source state to `sha256:6ea9fa40d97d1ae8159906f11a0695b577436afef186bc706da8895e4ba4bd95` across 83 included source/config/environment files. New-run `environment.json` records that code hash, the benchmark lock, and the installed package's Git `direct_url` revision.
