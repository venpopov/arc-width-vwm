Sys.setenv(RENV_CONFIG_SANDBOX_ENABLED = FALSE)
options(dplyr.summarise.inform = FALSE, tidyverse.quiet = TRUE)

source("renv/activate.R")
if (Sys.info()["sysname"] == "Linux") {
  Sys.setenv("RENV_CONFIG_REPOS_OVERRIDE" = "https://packagemanager.posit.co/cran/__linux__/noble/latest")
  options(pkgType = "both")
} else if (Sys.info()["sysname"] == "Darwin") {
  Sys.setenv(RENV_CONFIG_RSPM_ENABLED = FALSE)
}
