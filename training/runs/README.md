Run records from `training/pipeline.py`.

Each run writes `<run_id>.json` here - git-tracked, since this is the actual
comparable evidence (hyperparameters, feature set, repeated-CV metrics,
calibration) for every training run, replacing the earlier pattern where
results like the infarct-feature ablation only existed in an ephemeral
background-task log.

`<run_id>_model.pkl` and `<run_id>_calibration.png` are **not** git-tracked
(see `.gitignore`) - they're reproducible on demand from the JSON record's
`git_commit`, `hyperparameters`, and `feature_columns`, and committing a
multi-MB model file per run would bloat history fast. Only the actively
promoted model lives long-term, at the repo root (`best_prognostic_model.pkl`,
tracked via Git LFS) - see `training/promote_model.py`.

A run record's `"promoted"` field is `true` for exactly one run at a time:
the one currently live in `best_prognostic_model.pkl`. `promote_model.py`
flips the old one back to `false` when promoting a new one.
