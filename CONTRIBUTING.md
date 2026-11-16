# Contributing

Use a focused branch and state the affected method, evidence and validation in the pull request. Merge shared changes after a collaborator's review; preserve history.

- Name paths for their scientific content, using lowercase snake_case. Keep campaign dates, temporary phases and operational handoff labels out of maintained paths.
- Update imports, configurations, commands, reports and manuscript references together when a path changes.
- Preserve protocols, targets, folds, units, thresholds and numerical evidence. Amend a scientific protocol before running a changed experiment.
- Select models on validation folds only. Keep input datasets, predictions, checkpoints and machine-specific settings out of Git.
- Retain attribution for original frameworks and extensions; see [AUTHORS.md](AUTHORS.md).
- Run the complete tests and numerical checker; rebuild affected documents and inspect the diff.
- Use short, plain commit messages describing the change, such as `clean up`. Avoid promotional wording and elaborate summaries.

The frozen Gate 2 protocol must remain byte-identical.
