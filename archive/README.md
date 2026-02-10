# Archive

Files moved here during the 2026-02-09 repository reorganization. They are no
longer actively used but are preserved for reference.

| File                               | Original location | Reason                                                                                                                                    |
| ---------------------------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `fitting-rl-algorithms-backup.qmd` | `notebooks/`      | 100% commented-out code; preserved snapshot of the abandoned brms-based Bayesian fitting approach before switching to MLE                 |
| `models.R`                         | `R/`              | 8 model-fitting wrapper functions; no notebook currently sources this file. May be useful if revisiting hierarchical model specifications |
| `results.R`                        | `R/`              | `get_subject_parameters()` function with TODO; unused anywhere                                                                            |

Scripts and code shared by collaborators, or legacy files, preserved for consultation only.
These files are **not executed** as part of this project and likely cannot be executed without edits.

| File                                | Author         | Purpose                                                   |
| ----------------------------------- | -------------- | --------------------------------------------------------- |
| `ko-recovery-sdm.R`                 | Klaus Oberauer | SDM parameter recovery simulation & fitting               |
| `inspect_ko_sdm_recovery_results.R` | Ven Popov      | Inspect recovery results saved in `output/SDMrecovery.h5` |
| `SDM_v7_simulator.py`               | Wanke Pan      | Python SDM simulator (numpy/numba)                        |
