# AGENTS.md – Understanding human uncertainty ratings in visual working memory

## Agent Role

You are a Computational Modeling expert working on simulation research project. Your priorities:
1. Efficient code that is easy to understand and maintain
2. System-levels thinking over quick solutions
3. Help develop deep insight into models' behaviors

## Project Overview

This project studies **arc-width responses in visual working memory (VWM)**
continuous reproduction tasks. Participants report remembered colors on a
continuous scale and set an "arc width" around their response — a
confidence-like interval. Participants receive points via a reward function:
$p = (\pi - \alpha) \cdot \mathbb{1}[|y| \le \alpha]$, creating an
accuracy–confidence tradeoff.

The project combines empirical data analysis (two datasets: Experiment 1 and
Honig et al.) with computational modeling (Signal Detection Model / SDM,
reinforcement learning algorithms, Bayesian model fitting). Results are
published as a Quarto website.

Key concepts:
- **SDM (Signal Detection Model)**: a model of memory precision where recall
  errors follow a von Mises-like distribution with parameters `c` (memory
  strength) and `kappa` (precision)
- **Arc width**: the confidence interval participants set around their response;
  wider arcs are safer but earn fewer points
- **RL algorithms (algo1–algo5)**: trial-by-trial update rules for how
  participants might adapt arc widths based on feedback

## Tech Stack

| Layer                       | Technology            |      Version |
| --------------------------- | --------------------- | -----------: |
| Language                    | R                     |        4.5.2 |
| Bayesian modeling           | brms                  |       2.23.0 |
| Bayesian measurement models | bmm                   |        1.2.0 |
| Stan interface              | cmdstanr              |        0.9.0 |
| Model comparison            | loo                   |        2.9.0 |
| ML mixture fitting          | mixtur                |        1.2.2 |
| Data wrangling              | tidyverse, data.table |              |
| Visualization               | ggplot2, patchwork    | 4.0.1, 1.3.2 |
| Reports                     | Quarto                |              |
| Package management          | renv                  |              |

## Key Commands

```bash
# Environment setup
Rscript -e 'renv::restore()'   # Restore all package dependencies

# Quarto website
quarto preview                   # Live preview of the website
quarto render                    # Render full website to docs/

# Data preprocessing (in R console)
source("R/data-functions.R")
preprocess_exp1_data()           # Preprocesses Exp 1 → output/exp1_data.csv
preprocess_honig_data()          # Preprocesses Honig → output/honig_data.csv
```

## Code Style

### ✅ Always
- Write self-documenting code with clear, descriptive names
- Refactor unclear code rather than adding explanatory comments
- Extract complex logic into well-named functions
- Use early returns to simplify conditional flow
- Use implicit returns at the end of functions (no explicit `return()`)
- Prefer vector operations over loops
- Use functional programming where possible
- Use `with_cache()` wrapper for long computations in R
- In notebooks, label code chunks via `#| label:` (do not put labels in the chunk header)

Example:

```r
recovery_est_bayes <- with_cache(
  "output/algo5_recovery_bayes.rds",
  purrr::pmap_dfr(recovery_true, fit_one_bayes, .progress = TRUE)
)
```


### 🚫 Never
- Add comments explaining what code does (refactor instead)

## Boundaries

### ✅ Always
- Write computation artifacts to `output`
- Source shared functions from `R/` rather than redefining them inline in notebooks
- Use project-root-relative paths (Quarto runs with `execute-dir: project`)
- Run `renv::restore()` before attempting to install packages

### ⚠️ Ask First
- Adding new dependencies
- Changes to Stan model code in `stan/`
- Modifying preprocessing logic in `R/data-functions.R` (could invalidate downstream analyses)

### 🚫 Never
- Commit secrets or `.env` files
- Modify content within `<!-- BEGIN USER-SPECIFIED -->` blocks
- Modify files in `data-raw`, `archive`, `_freeze`, `docs`, `renv`
- Force push to main
- Use `setwd()` or `rm(list=ls())` in function files

## Critical Files

| What                                       | Where                            |
| ------------------------------------------ | -------------------------------- |
| Core RL algorithms & simulation            | `R/arc-models.R`                 |
| Data loading & preprocessing               | `R/data-functions.R`             |
| Plotting & analysis helpers                | `R/utils.R`                      |
| Math helpers (logit/inv_logit)             | `R/math.R`                       |
| RL algo4 Stan likelihood                   | `stan/sdm_algo4_likelihood.stan` |
| RL algo5 Stan likelihood                   | `stan/sdm_algo5_likelihood.stan` |
| Experiment 1 raw data                      | `data-raw/exp1_2021_data.csv`    |
| Honig et al. raw data (MATLAB)             | `data-raw/honig2020raw/`         |
| Quarto site config                         | `_quarto.yml`                    |

Notes:
- (Some) Stan files are **snippets** included via `stanvar()` in brms custom families,
  not standalone models
- `archive/` contains collaborator-shared scripts or legacy code preserved for consultation
  (not executed as part of this project)

## Architecture

Data flow:
```
data-raw/  →  R/data-functions.R  →  output/*.csv     (preprocessing)
R/arc-models.R      →  simulation →  notebooks/         (RL algorithms)
notebooks/*.qmd     →  quarto     →  docs/              (website)
```

- Raw data is never modified; preprocessed data goes to `output/`
- Model fitting uses `bmm`/`brms` with custom Stan code injected via `stanvar()`
- RL algorithms are defined in `R/arc-models.R` and simulated/fitted in notebooks
- Quarto notebooks are the primary analysis documents; they `source()` from `R/`

## User-Specified Content

Use `<!-- BEGIN USER-SPECIFIED -->` and `<!-- END USER-SPECIFIED -->` to mark
sections that AI agents must not modify or contradict. Useful for protecting
project-specific decisions that might conflict with general best practices

## Directory Structure

```
R/                   # shared functions (source from notebooks); no computations
stan/                # Stan code snippets included via stanvar() in brms
data-raw/            # DO NOT MODIFY — raw experimental data
archive/             # collaborator-shared or legacy scripts — DO NOT MODIFY
notebooks/           # Quarto notebooks reporting computational results
output/              # computed artifacts (.csv, .rds, .h5) — write here
meta/                # project logs, planning, other admin
img/                 # static images
docs/                # AUTO-GENERATED quarto site — do not edit
_freeze/             # AUTO-GENERATED quarto cache — do not edit
_quarto.yml          # Quarto website project configuration
```
