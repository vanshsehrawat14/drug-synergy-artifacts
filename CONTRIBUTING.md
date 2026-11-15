# Contributing

Use issues or pull requests for corrections and extensions. State the affected result, implementation change and validation. Keep changes to scientific assumptions distinct from implementation fixes.

- Preserve frozen protocols, splits, targets, units and evidence. Amend a protocol before running a changed experiment.
- Use validation folds for model selection. Held-out outcomes must not enter fitting or tuning.
- Keep input data, model weights, checkpoints, credentials and machine-specific configuration out of Git.
- For model results, record configuration, seed, information regime, dataset and split hashes, validation selection, runtime and prediction coverage.
- Run `python -m pytest -q` and the numerical checks relevant to the change. Do not change a test merely to obtain a passing result.
- Keep `docs/gate2_protocol.md` byte-identical; the original analysis verifies its SHA-256.

Original frameworks and extensions must retain their attribution in [AUTHORS.md](AUTHORS.md) and the method reports.
