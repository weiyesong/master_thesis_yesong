# C6 reporting corrections — direct-evidence acceptance

Completed 2026-09-22 UTC. This closes reporting defects I02/C05 and I03/C09 for the new publication package. The historical C6 report, figures, results, checkpoints, and previous audit remain unchanged.

## Implemented changes

- `scripts/c6_build_final_results.py` accepts `--output-root`, refuses existing destinations, stages a complete package, and atomically publishes the new directory. This completion used `reports/core_rq_completion_20260921/published/`.
- Core TS applicability is read per dataset/model/adaptation cell from `reports/thesis_master_results.csv`. All four EuroSAT TS cells remain complete. All four TreeSatAI and eight segmentation TS cells are N/A with blank metrics. The existing master/freeze explanation is preserved as reporting policy; it is not recast as proof that validation reuse invalidates held-out evaluation.
- All 12 historical TreeSatAI TS seed results, including adverse NLL/Brier/ECE effects, remain visible in a separate CSV and a delta table in the report. The report explicitly records: results available 2026-08-24; exclusion freeze 2026-08-25; stale COMPLETE C6 2026-08-26; master/RQ3 N/A 2026-08-27. No exclusion motive is inferred, and the TS reporting decision is not claimed to precede all TS results.
- Classification decision-reliability curves now use saved deterministic probabilities directly. TreeSatAI uses flattened binary decision confidence `max(p,1-p)` and correctness at threshold 0.5. A separate pooled-positive-label diagnostic uses `p(label=1)` versus the binary label; it is not substituted for the table's decision ECE.
- All main reliability plots use 15 right-closed bins `(lo,hi]`, with 0 included in the first bin. The PNGs show every bin count, identify seed 42 and ensemble members 42/43/44, state fractional units, and retain empty bins with count 0/undefined means in matching CSVs. Segmentation uses the supplied valid mask and pooled valid pixels; confidence is not transformed into binary decision confidence.
- TreeSatAI uses Macro-F1 in both the performance–calibration plot and the three-seed MC robustness report. EuroSAT uses accuracy; segmentation uses mIoU. Exact-match TreeSatAI accuracy remains disclosed separately in the full classification table.
- The report links the completion report, RQ results section, and 24-row same-checkpoint three-way supplement. The core C6 tables remain the original frozen MC-versus-deterministic scope; the supplement isolates MC-versus-dropout-off effects. It also links class diagnostics and 10/15/30-bin evidence from the previous audit.

## Verification from original artifacts

`published/tables/package_validation.json` records **64** cell checks, **568** mean/SD/metric values with **zero difference** from the authoritative master, and **52** main reliability curves independently rebuilt from saved predictions. Maximum plotted ECE difference from its matching seed-level metric is **2.737400972563364e-08**. All **12** N/A cells and **12** historical TreeSatAI TS seed rows are preserved. The complete 136-row master is copied into the package.

`reporting_tests.log` records **9/9 tests passing** (`tests.test_c6_final_results` and `tests.test_c3_calibration_ensembles`). New regression tests cover exact bin edges/empty bins, decision ECE versus positive-label diagnostics, segmentation confidence below 0.5, and authoritative TS scope with duplicate/missing-cell rejection.

The classification and segmentation main reliability PNGs and the separate TreeSatAI diagnostic were visually inspected: count axes, legends, seed labels, units, and subplot labels are readable. Links to already-existing plots/diagnostics resolve. The three completion links intentionally target root-authored outputs being assembled in parallel.

`reporting_hash_verification.json` independently re-read and verified **74 source SHA256 values** and **19 generated artifact SHA256 values**, with **zero mismatches**. This includes the actual raw predictions used in plots, not another agent's conclusions. The generation manifest uses package-relative output paths and pins the generator source.

Generator SHA256: `9c831f3b6e904536e376ca5e8f9ae11c362b89f2fe535b84183dc12d17cbfeb1`.

Package manifest SHA256: `86f839fcc75db3ab3b78711d12604596e90eae8c0e064e241ee4957469929255`.

The final small report correction (TreeSatAI Macro-F1 robustness display and matching alt text) reused the already-generated tables to rerender the report. Its historical appendix was preserved verbatim; source provenance and all affected generated hashes were refreshed. No figures or numeric metric CSVs were changed by this final correction. The generator itself contains the same correction, so a future full rebuild produces the same reporting behavior.

## Remaining scope

This change does not resolve historical DOFA-EuroSAT normalization/code-version provenance, does not claim a review of an unavailable full thesis, and does not claim new training. Root's fixed-weight dropout-off supplemental evaluation and Claude Code's independent review are separate acceptance evidence. Methods without benefit remain valid observed results.

## Final path portability check (2026-09-23)

The three completion links are computed relative to the actual output directory, so both this published layout and the default reports-level layout resolve. The existing published body is byte-for-byte unchanged. Generator/source hashes were refreshed and all 74 source and 19 generated hashes reverified by verify_reporting_portability.py; four C6 tests passed again after this path-only change.
