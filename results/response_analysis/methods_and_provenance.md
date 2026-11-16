# Response analysis: methods and provenance

This record describes the original response-analysis execution on RebelX and its local analysis, based on source `d2c8f41`. Source commits are listed in `research_findings.md`; the public checkout contains the integrated implementations. Historical results and configurations are preserved.

## Inputs (SHA-256)

| Input | SHA-256 |
|---|---|
| `synergy_artifacts_research_d2c8f41.zip` (packet; all 7 manifest entries verified locally and on RebelX) | `7d9358915e3450512a6b9673e3f8527559f8cf9f8d56a3c30afa2b4c53e1d0e5` |
| `scientific_native_full/surface_cases.parquet` (367,152 dose cases, cached native fits and outputs) | `69e2c76476c8d14aa3c289b0a98ad6703b40b7eb9ad903f9e2df2cd1d4e23528` |
| `scientific_native_full/blocks.parquet` (92,208 blocks) | `866ae92c09829e25dde86e139a83d6a9be11fe335b024b530141243d7f13f169` |
| `2026-11-06_validation/loewe_verify/cases_verified.parquet` (historical classes, θ-roots) | `384a2bd51b4c9ffeca7d4d1768237a6212bf5cf126b2c16a382af87d72acf438` |
| `results/interaction_analysis/precision/certificates.csv` (100-digit follow-up) | `5e16570fbba38129b4a8de3ebf331b09b495398b9f0a60809475957e27c917a5` |
| `data/processed/oneil_v1/observations.parquet` | `ea588405019971906ac99c3caa547d5f4fac5b1be340356f0a8fe17a6aa3ce8b` |
| SynergyFinder 2.0.12 `R/calculate_synergy_score.R` (`a09d792`, Loewe branch identical to 2.0.3) | `ebc914c816ce543c382678d047f315181545fc1f58d43b9084934369ce9aebe7` |
| SynergyFinder 2.0.12 `R/fit_dose_response_model.R` | `327b6ba1fa10ba8a9b5874b4ad1c0e9371967d0562e838396fd5c54485504643` |
| Early RebelX return `research_return_20261107.tar.gz` (ALMANAC/NCATS runs, ALMANAC prepared data) | `5809c9ea76bf820aca3ddb4ca569ed081274b26f4a7403922de5ee95a7be8e17` |
| `research_return_20261107_ncats_v2_prepared.tar.gz` (NCATS v2 prepared data from `wt_analysis` @ `689a97b`) | `7e124d18de0b80ed1c4b5413cd57b008e2de5ce3e15567f19042afe788164ccd` |
| Final RebelX return `research_return_20261107_final.tar.gz` (job 132666; all 45 ablations) | `2b75eeba78743ea15e6d72039df6b90e985ad49186192ebcd0584de174fc5ce4` |
| Transferred ablation subset `ablation_slim2.tar` (per-file hashes in `files_slim.sha256`) | `6e770c2b3b1c57f31d4e6e8dd40a054aa509bde40e0f7285aa1f63f28d27adc0` |

Prepared-data hashes for every screen are listed in `data_coverage.csv`. New root certificates are in `root_certificates/certificates.parquet` (`0bb4d2da2ce08c7f49e4511a8926e1590f42e5b6839b0d3065338126c5edeb3b`).

## Source path reconstruction (Loewe)

Single-agent observations → `FitDoseResponse` (drc LL.4 on dose; on failure L.4 on natural-log dose, fixed Emin/Emax as in source) → cached coefficients `(b, c, d, e)` per arm → the native equation `eq.*` in `calculate_synergy_score.R` → `nleqslv(..., method = "Newton")` started at max lower asymptote + 1 → accept iff `termcd < 3` → else summed-dose fallback `max(f_a(x_a+x_b), f_b(x_a+x_b))` → cap at 100 → case score = observed − reference → block score = mean over the 16 positive-dose cases → pair-cell = mean of blocks → drug pair = mean of pair-cells → ranking.

Implementation checks:

- The native LL.4/L.4/mixed equations (2.0.12 source) match the inverse used by the existing verifier and the new certifier. L.4 uses the natural log of dose, and the arm dose enters undivided.
- The summed-dose fallback reconstructed from coefficients matches the native reference in all 184,905 fallback cases, with a maximum difference of 1.2e-11.
- Observed responses mapped to cases reproduce all 92,208 native block scores to within 1.5e-14. Case arm a is the column drug (`conc_c`) and arm b the row drug (`conc_r`).
- The inverse identity D(f(x)) = x holds at 60 digits on 12,204 distinct fitted arms. The maximum relative error is 2.4e-29; two cases exceed 1e-30 because of near-saturation cancellation.
- Arm-order symmetry: 0 status mismatches in 20,000 swapped cases.
- Monotonicity of certified references along each dose axis: 0 violations in 271,240 adjacent pairs.
- Original native curve fitting was not rerun. R is not installed locally or in any RebelX conda environment. The existing native refits (R 3.6.3, drc 3.0.1, nleqslv 3.3.2; 160 surfaces) are used as reported. No reimplementation was substituted.

## Interval certificate (`synergy_artifacts/loewe_interval.py`)

Float coefficients are treated as exact binary rationals. For co-monotone arms, g(y) = Σ x_i/D_i(y) − 1 is strictly monotone on the open joint effect interval (ℓ,u). One end has a term tending to +∞, and the other has a finite limit g_end. A root therefore exists iff g_end < 0, and it is unique.

- **Certified root:** a bracket [y₀,y₁] with ℓ<y₀<y₁<u, verified exactly. mpmath interval enclosures of g(y₀) and g(y₁) at ≥50 digits must have strictly opposite signs; precision escalates to 100 and then 200 digits if needed.
- **Certified non-existence:** the enclosure of g_end is strictly positive.
- Opposite or flat curves and an empty joint interval are undefined by definition.
- Hints come from the θ-solver, the native value or the 100-digit follow-up. If a hint fails, the certifier falls back to exact bisection in a logit coordinate.
- Results: 195,984 certified roots, 117,824 certified non-existence, 49,184 opposite/flat, 4,160 empty interval, 0 unresolved, 0 invalid. The maximum bracket width is 3.3e-11 inhibition points, and the closest certified root is 9.5e-79 inside an asymptote.

## Analyses and code

| Analysis | Code | Output |
|---|---|---|
| Certificates for all cases; rounding 6/8/10/12 digits | `synergy_artifacts/loewe_interval.py`, `experiments/response_analysis/round_certs.py` | `root_certificates/` |
| Regimes and identity tests | `experiments/response_analysis/regimes_and_tests.py` | `loewe_regimes.csv`, `loewe_regime_summary.csv`, `root_certificates/identity_tests.json` |
| Decision sensitivity (preregistered) | `synergy_artifacts/decision_sensitivity.py` | `decision/ranking_stability.csv`, `decision/ranking_stability_units.csv`, `decision/guaranteed_changes.csv`, `decision/conditional_changes.csv`, `decision/top5_N_to_C1_changes.csv` |
| Post-hoc edge-convention, attribution and concentration analyses | `synergy_artifacts/decision_followup.py` | `decision/edge_exclusion_sensitivity.csv`, `decision/consistent_edge_conventions.csv`, `decision/change_attribution.csv`, `decision/change_concentration.json` |
| Perturbation stability (Phase 3.2) | `experiments/response_analysis/rounding_effects.py` | `decision/perturbation_stability.csv` |
| Independent solver (Phase 3.1) | `experiments/response_analysis/independent_solver_check.py` | `root_certificates/independent_solver_check.csv` |
| Prospective model-predicted scoring | `synergy_artifacts/prospective_scoring.py` | `decision/prospective_scoring.csv` |
| Undefined-accepted convention sensitivity | `experiments/response_analysis/undefined_accepted_sensitivity.py` | `decision/undefined_accepted_sensitivity.csv` |
| Cross-screen standardization | `synergy_artifacts/cross_screen.py`, `experiments/response_analysis/screen_diagnostics.py` | `cross_screen/` (primary seed 2026; `three_seed/` sensitivity) |
| GPU ablation contrasts | `synergy_artifacts/ablation_summary.py` (validated by `--pseudo`, which reproduces the frozen F2/F0 comparison) | `gpu_ablation_summary.csv`, `gpu/` |
| Figures | `experiments/response_analysis/figures.py` | `figures/*.png` with plotting tables `figures/*.csv` |

## Deviations from frozen protocols and from the response_analysis preregistration

1. **Packet launch.** `submit.sh` needed two environment fixes. An ignored temporary output directory was created in the clone, because pytest `--basetemp` needs its parent. `/data` was added to the clone's `.git/info/exclude`, because the script's own data symlink failed its clean-tree check. No tracked file or configuration changed. The first submission aborted before `sbatch`.
2. **Regime 4 tolerance.** The preregistration said "native value inside the certified bracket", which is impossible for a 1e-8 Newton tolerance. Before regimes were tabulated, this was amended to |native − certified root| ≤ 0.05, the existing `loewe_verification.json` "solver at root" tolerance. The observed maximum is 2.7e-6, so the amendment changes no assignment.
3. **Post-hoc analyses.** The edge-exclusion ε sweep, the consistent-edge comparison, change attribution, the undefined-accepted sensitivity and the screen diagnostics were added after the preregistered conditional table showed every change involving near-asymptote roots. They are labelled exploratory.
4. **Cross-screen seeds.** O'Neil `f2_combined` has seeds 2027/2028 only for folds 0, 3 and 4. The primary cross-screen comparison therefore uses seed 2026 for every screen (same architecture, configuration and seed). Three-seed results are a sensitivity check, with O'Neil F2 reported as NA.
5. **O'Neil Bliss baseline.** The training-pair Bliss reference is `reference_offset` from `benchmark_v1/f2_residual` (checked seed-invariant). Other screens use their `*_reference_bliss` runs.
6. **Provenance label.** NCATS prediction files carry `study = ONEIL` in a metadata column. Observation IDs and prepared data identify the correct study, so this is a stale label, not a data error.
7. **Collector scope.** `pack_return.py` omits `wt_analysis`, the only location of NCATS v2 prepared data. Those files were packed separately with a manifest. `pack_return.py` also collects only cluster outputs. The local analyses in this directory are bundled separately (see the final handoff note).
8. **Ablation exit codes.** RebelX has no Slurm accounting database. The `scontrol` logger did not capture exit codes for tasks 0–4 and 44. Completion of those tasks rests on their metadata records, prediction files and end-of-run log summaries.
9. **Confirmatory seed.** Seed 2029 was not part of the original execution. Its later controlled follow-up, including pooled and single-seed results, is reported in `results/seed_validation/README.md`.
10. **Transfer subset.** Checkpoints (`best.pt`) and `predictions_with_truth.parquet` were not transferred; the latter duplicates `predictions.parquet` plus the truth table. Both remain in the final `pack_return.py` archive on RebelX.
11. **Exploratory ablation check.** Per-seed and per-fold response changes (`gpu/ablation_seed_check.csv`) were added after the frozen contrasts were computed.
