# GPU return, 2026-11-08

Packet `106855c`, RebelX array `132733` (30 sequential GPU tasks): 30/30 tasks complete, all `ExitCode=0:0`, no failures or retries. Tasks 0–19 were not resubmitted: their 20 configurations are identical to array `132713` (`research/response_analysis-2026-11-07`, `results/seed_validation/`), 20/20 complete.

Archive `research_return_20261108_mechanisms.tar.gz` (67,511,247 B, SHA-256 `f05e479683c0c42c223d304822377eac31a27558f708a68adb73e514dad869c1`), collected with `collect_artifacts.py --checkpoints`; 270 files verified. Checkpoints are retained by the authors.

| Experiment | Primary endpoint | Change [pair; drug 95%] | Folds | Holm p | Material |
|---|---|---|---|---|---|
| Joint identity omission | Factorial response-error interaction | +2.21% [0.82, 3.59; −1.15, 5.57] | all positive | 0.38 | No |
| No drug identity, Bliss-residual objective | Reference-augmented pair-cell contrast vs `f2_residual` | −3.85% [−7.03, −0.41; −11.17, 3.47] | disagree | 0.38 | No |

Joint omission costs slightly more than the sum of single omissions, below 5% and with the drug interval spanning zero: inconclusive. Under the residual objective the no-drug contrast advantage shrinks from −7.4% (raw target, four seeds) to −3.9% with disagreeing folds, which weakens the identity-overfitting account. `seed2029_crosscheck/` reproduces every change, Holm p and classification of the independent seed-2029 analysis.
