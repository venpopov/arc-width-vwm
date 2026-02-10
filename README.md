# Arc width measurement in visual working memory

<!-- badges: start -->

[![Project Status: WIP - Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

<!-- badges: end -->

This repository contains data, code, and analyses for a project studying **uncertainty estimation in visual working memory**. In the arc-width procedure, participants reproduce a remembered color on a color wheel and then indicate their uncertainty by drawing a symmetric arc around their response. Points are awarded as $p = (\pi - \alpha) \cdot \mathbf{1}[|y| \le \alpha]$, creating an accuracy–confidence tradeoff.

## Reports

Rendered notebooks are available at **[venpopov.github.io/arc-width-vwm/](https://venpopov.github.io/arc-width-vwm/)**.

## How to download and replicate

1.  Clone the repository or download the [.zip](https://github.com/venpopov/arc-width-vwm/archive/refs/heads/main.zip) archive.
2.  Open the `guilty-goose.Rproj` file in RStudio.
3.  Restore package dependencies:

``` r
renv::restore()
```

`renv` creates a reproducible environment for R projects. It will install the packages listed in the `renv.lock` file into a project-local library.

## Repository structure

```
R/                   Shared functions (sourced by notebooks)
stan/                Stan code snippets (included via stanvar() in brms)
notebooks/           Quarto notebooks — the main analysis documents
data-raw/            Raw experimental data (do not modify)
output/              Computed artifacts (.csv, .rds, .h5)
archive/             Preserved but inactive code
meta/                Project management and audit documents
docs/                Auto-generated Quarto site (do not edit)
_freeze/             Auto-generated Quarto cache (do not edit)
reference/           Collaborator-shared or legacy scripts
```
