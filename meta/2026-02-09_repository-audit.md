# Repository Audit & Improvement Plan

**Date**: 2026-02-09  
**Commit**: [`601528d`](https://github.com/venpopov/arc-width-vwm/commit/601528d) (main) — *"Rewrite RL model fitting: MLE parameter recovery for noisy-gradient algo5"*

---

## Executive Summary

The repository contains high-quality core analyses, but its presentation as a Quarto website has grown organically and lacks navigational structure, explanatory scaffolding, and housekeeping. A newcomer landing on the site would see an alphabetically ordered list of eight notebooks with no indication of reading order, topic grouping, or dependencies. Several notebooks are stubs or dead code, some R source files are orphaned, and duplicated logic is scattered across files. This document diagnoses these issues and proposes a concrete improvement plan.

---

## 1. Current State: Diagnosis

### 1.1 Website Navigation

The sidebar uses `contents: notebooks/*` (auto-glob), so notebooks appear in **alphabetical filename order**:

1. exp1-by-subject
2. exp1-eda
3. fitting-rl-algorithms
4. fitting-rl-algorithms-backup
5. honig-explore
6. optimal-arc-response
7. reinforcement-learning-arc
8. sequential-dependencies

This ordering is confusing. The *logical* reading order is roughly:

1. **exp1-eda** — What is the data? What are the basic empirical results?
2. **exp1-by-subject** — Per-subject diagnostic plots
3. **honig-explore** — Second dataset exploration
4. **optimal-arc-response** — Theoretical derivation of optimal behavior
5. **reinforcement-learning-arc** — RL algorithms and simulation
6. **fitting-rl-algorithms** — MLE fitting and parameter recovery
7. **sequential-dependencies** — Empirical test of RL predictions
8. ~~fitting-rl-algorithms-backup~~ — dead code, should not be in the site

### 1.2 Landing Page (index.qmd)

- Only 22 lines. Says "will be updated as we go" — WIP language.
- Embeds a YouTube video but gives no project summary, no explanation of what arc width is, no description of the research question, no outline of the analyses.
- A newcomer has no map of the project.

### 1.3 Notebook Quality Spectrum

| Notebook                     | Lines | Quality   | Status                           |
| ---------------------------- | :---: | --------- | -------------------------------- |
| honig-explore                |  35   | Stub      | Barely started                   |
| exp1-by-subject              |  49   | Sparse    | Diagnostic only, no narrative    |
| fitting-rl-algorithms-backup |  215  | Dead      | 100% commented-out code          |
| exp1-eda                     |  269  | Moderate  | Functional, needs more narrative |
| sequential-dependencies      |  363  | Very good | Has code duplication             |
| fitting-rl-algorithms        |  378  | Excellent | Clean MLE + recovery             |
| optimal-arc-response         |  458  | Excellent | Core theoretical analysis        |
| reinforcement-learning-arc   |  624  | Excellent | Core RL exploration              |

The three "core" notebooks (optimal-arc-response, reinforcement-learning-arc, fitting-rl-algorithms) are well-written. The EDA and sequential-dependencies notebooks are functional. The remaining three are stubs or dead code.

### 1.4 Code Duplication

| Duplicated Code                        | Canonical Location                  | Also Found In                                    |
| -------------------------------------- | ----------------------------------- | ------------------------------------------------ |
| `optimum_equation`, `find_optimum_sdm` | `R/arc-models.R`                    | `optimal-arc-response.qmd` (redefined inline)    |
| `run_arc_algorithm`, `algo4`, `algo5`  | `R/arc-models.R`                    | `sequential-dependencies.qmd` (redefined inline) |
| `algo5_loglik`                         | Only in `fitting-rl-algorithms.qmd` | Not extracted to `R/`                            |

These duplications risk silent divergence and make refactoring fragile.

### 1.5 Orphaned R Source Files

| File          | Issue                                                                                                                   |
| ------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `R/models.R`  | Contains 8 model-fitting wrappers. **No notebook sources this file.** `optimal-arc-response.qmd` fits its model inline. |
| `R/results.R` | Contains `get_subject_parameters()` with a TODO comment. **Unused anywhere.**                                           |

### 1.6 Orphaned Stan Files

All four Stan files (`sdm_algo4_likelihood.stan`, `sdm_algo5_likelihood.stan`, `sdm_simple_normalization.stan`, `sdm_simple_tdata.stan`) are only referenced from the 100%-commented-out code in `fitting-rl-algorithms-backup.qmd`. They represent the abandoned Bayesian approach. `sdm_simple_funs.stan` is the core SDM implementation used by the `bmm` package but its direct inclusion here appears to be a local copy rather than a package dependency.

### 1.7 Stale & Leftover Files

- `notebooks/.ipynb_checkpoints/` — two old Jupyter `.ipynb` artifacts (pre-Quarto)
- `README.md` references old project name "guilty-goose"
- `fitting-rl-algorithms-backup.qmd` — entirely commented-out dead code

### 1.8 Minor Code Quality Issues

- `R/arc-models.R`: `force(sd)` calls in `algo4`/`algo5` reference a nonexistent parameter (leftover from refactoring)
- `R/utils.R`: `library()` inside `theme_Publication` function body (should use `::` or require at top)
- `R/utils.R`: Comment on `spaghetti_single_runs` — "ugly overly-complicated function written by ChatGPT which I'm too lazy to simplify"
- `R/results.R`: Has TODO marker

---

## 2. Improvement Plan

### 2.1 Restructure the Quarto Website Navigation

**Goal**: Replace the auto-glob sidebar with explicit, logically ordered sections.

Proposed `_quarto.yml` sidebar structure:

```yaml
sidebar:
  style: docked
  search: true
  contents:
    - text: "Home"
      href: index.qmd
    - section: "Data Exploration"
      contents:
        - text: "Exp 1: Exploratory Analysis"
          href: notebooks/exp1-eda.qmd
        - text: "Exp 1: By-Subject Diagnostics"
          href: notebooks/exp1-by-subject.qmd
        - text: "Honig et al.: Data Exploration"
          href: notebooks/honig-explore.qmd
    - section: "Theory & Modeling"
      contents:
        - text: "Optimal Arc Width (SDM)"
          href: notebooks/optimal-arc-response.qmd
        - text: "RL Algorithms: Exploration"
          href: notebooks/reinforcement-learning-arc.qmd
        - text: "Model Fitting: MLE & Recovery"
          href: notebooks/fitting-rl-algorithms.qmd
    - section: "Empirical Tests"
      contents:
        - text: "Sequential Dependencies"
          href: notebooks/sequential-dependencies.qmd
```

This groups notebooks into three conceptual sections with a natural reading progression and removes the dead backup notebook from the sidebar.

### 2.2 Rewrite the Landing Page (index.qmd)

Replace the current minimal landing page with a substantive project overview:

1. **Project title and one-paragraph summary** — What is arc width? What is the research question?
2. **Task diagram or schematic** — How the arc-width continuous reproduction task works (the reward function, tradeoff)
3. **Guided reading order** — Numbered list of notebooks with one-sentence descriptions and links, grouped by section
4. **Key concepts glossary** — Brief definitions of SDM, arc width, RL algorithms (algo1–5), Chebyshev quadrature
5. **Data sources** — Brief description of Experiment 1 and Honig et al.
6. **Repository structure overview** — What's in `R/`, `stan/`, `output/`, etc.
7. Move the YouTube video to a "Presentations" subsection rather than being the centerpiece

### 2.3 Add Narrative Scaffolding to Thin Notebooks

**exp1-by-subject.qmd**: Add 2–3 paragraphs explaining what to look for in the per-subject plots, mention how many subjects, summarize takeaways (e.g., individual differences in precision, outlier subjects).

**honig-explore.qmd**: Either flesh out with real analysis (observation counts, arc-width distributions by condition, comparison with Exp 1) or mark explicitly as a "data preview" stub with a note about its limited scope.

**exp1-eda.qmd**: Add a brief introductory section explaining the experimental design before diving into plots. Add transition text between sections.

### 2.4 Eliminate Code Duplication

1. **Move `algo5_loglik` to `R/arc-models.R`** — it's the log-likelihood for the noisy-gradient model, logically belongs with the other algorithm definitions.
2. **Remove inline redefinitions** in `optimal-arc-response.qmd` — replace with `source("R/arc-models.R")`.
3. **Remove inline redefinitions** in `sequential-dependencies.qmd` — replace with `source("R/arc-models.R")`.
4. **Audit for parameter mismatches** — ensure the `R/arc-models.R` versions match what each notebook expects, since the inline versions may have diverged.

### 2.5 Clean Up Orphaned and Dead Code

1. **`fitting-rl-algorithms-backup.qmd`**: Move to a new `archive/` directory (or delete). Remove from the rendered website. If preserving for reference, add a note explaining its history.
2. **`R/models.R`**: Decide whether to:
   - (a) Refactor `optimal-arc-response.qmd` to use these wrappers (preferred if they are still useful), or
   - (b) Move to `archive/` if the model specifications have evolved past what these wrappers encode.
3. **`R/results.R`**: Same decision — integrate into a notebook or archive.
4. **Stan files for algo4/algo5**: If the Bayesian approach is truly abandoned, move to `archive/`. If it might be revisited, keep but add a README note.

### 2.6 Housekeeping

1. **Delete `notebooks/.ipynb_checkpoints/`** — stale Jupyter artifacts.
2. **Update `README.md`** — Remove "guilty-goose" references. Add a proper project description, link to the live site, list key dependencies, and describe the directory structure.
3. **Fix `force(sd)` bug** in `R/arc-models.R` — remove the errant `force(sd)` calls in `algo4` and `algo5`.
4. **Fix `library()` in function body** — `R/utils.R::theme_Publication` should not call `library(grid)` and `library(ggthemes)` inside the function. Either use namespace-qualified calls (`grid::unit()`, `ggthemes::scale_colour_Publication`) or move imports to the top of the file.
5. **Clean up self-deprecating comment** in `spaghetti_single_runs` — either simplify the function or remove the comment.
6. **Add `.ipynb_checkpoints` to `.gitignore`** if not already present.

### 2.7 Cross-Notebook Links and Dependency Arrows

Each notebook that depends on outputs from another should state this clearly at the top:

- **optimal-arc-response.qmd** → produces `output/exp1_sdm_bmm_by_subj_setsize.rds`, `output/subj_optimum_posterior.rds`
- **fitting-rl-algorithms.qmd** → produces `output/algo5_recovery_est.rds`
- **sequential-dependencies.qmd** → consumes outputs from the above

Add a "Prerequisites" callout at the top of each notebook that depends on prior outputs:

```markdown
::: {.callout-note}
## Prerequisites
This notebook uses the SDM model fit from [Optimal Arc Width](optimal-arc-response.qmd).
Make sure that notebook has been rendered first.
:::
```

### 2.8 Consider Adding a Project Narrative Document

Beyond the landing page, consider adding a **"Project Guide"** (`guide.qmd` or similar) that walks through the entire research story in ~500 words without code:

1. The task and reward function
2. The empirical observation that people are near-optimal
3. The question of *how* they achieve optimality
4. The RL algorithm approach
5. Model fitting and recovery
6. Sequential dependency predictions

This gives readers the "forest" before they enter the "trees" of individual notebooks.

---

## 3. Priority Ordering

| Priority | Action                                                            | Effort  | Impact |
| :------: | ----------------------------------------------------------------- | :-----: | :----: |
|    1     | Restructure sidebar with explicit ordering and sections           |   Low   |  High  |
|    2     | Rewrite index.qmd with project overview and reading guide         | Medium  |  High  |
|    3     | Eliminate code duplication (extract to R/, source from notebooks) |   Low   | Medium |
|    4     | Remove/archive dead code (backup notebook, orphaned R files)      |   Low   | Medium |
|    5     | Delete stale files (.ipynb checkpoints)                           | Trivial |  Low   |
|    6     | Add narrative scaffolding to thin notebooks                       | Medium  | Medium |
|    7     | Fix minor code quality issues                                     |   Low   |  Low   |
|    8     | Add cross-notebook dependency callouts                            |   Low   | Medium |
|    9     | Update README.md                                                  |   Low   | Medium |
|    10    | Add project narrative guide document                              | Medium  |  High  |

---

## 4. Verification

After implementing changes:

1. `quarto render` — Ensure all notebooks render without errors
2. `quarto preview` — Manually verify sidebar ordering, section grouping, and all links
3. Check that no notebook has inline-redefined functions that exist in `R/`
4. Grep for `source(` calls to confirm all notebooks use shared functions
5. Confirm `fitting-rl-algorithms-backup.qmd` is excluded from the rendered site
6. Verify the landing page reads clearly to someone unfamiliar with the project
