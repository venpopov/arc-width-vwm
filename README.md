# guilty-goose

<!-- badges: start -->

[![Project Status: WIP - Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

<!-- badges: end -->

This repo contains data and code for the Arc-width project.

## How to download and replicate

TODO: Add instructions for the targets pipeline

1.  You can clone the current repository or download the [.zip](https://github.com/venpopov/guilty-goose/archive/refs/heads/master.zip) archive.
2.  Open the `guilty-goose.Rproj` file in RStudio.
3.  Run the following code to install the required packages:

``` r
renv::restore()
```

`renv` is a package that creates a reproduction environment for R projects. It will install the packages listed in the `renv.lock` file, which are the packages used in the project. The `renv::restore()` command will install the packages in a separate library, so it will not interfere with your global library.
