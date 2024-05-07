library(targets)
library(tarchetypes)
library(crew)


# Set target options:
tar_option_set(
  error = "null", packages = c("dplyr", "cmdstanr"),
  controller = crew_controller_local(workers = 2, local_log_directory = "output/logs")
)

# Run the R scripts in the R/ folder with your custom functions:
tar_source("R/")

# Set other options
options(
  mc.cores = 4,
  brms.backend = "cmdstanr",
  tidyverse.quiet = TRUE,
  dplyr.summarise.inform = FALSE,
  bmm.sort_data = TRUE
)

# Targetst pipeline
tar_plan(
  # process the raw data
  tar_target(honig_data_folder, "data-raw/honig2020raw/", format = "file"),
  tar_target(honig_data, preprocess_honig_data(honig_data_folder, "output/honig_data.csv")),
  tar_target(honig_data_uniform, dplyr::filter(honig_data, session %in% c(1, 4))),
  tar_target(exp1_data_file, "data-raw/exp1_2021_data.csv", format = "file"),
  tar_target(exp1_data, preprocess_exp1_data(exp1_data_file, "output/exp1_data.csv")),

  # fit sdm to honig data
  tar_target(honig_bmm_fit_by_subject, fit_bmm1(honig_data_uniform)),
  tar_target(honig_bmm_fit_by_subject_and_session, fit_bmm2(honig_data)),
  tar_target(honig_bmm_fit_kappa_by_subject_c_by_session, fit_bmm3(honig_data)),

  # fit mixture2p ML to exp1 data
  tar_target(
    exp1_2p_ml,
    fit_2p_ml(exp1_data,
      by = c("subject", "exp_type", "setsize", "encodingtime", "delay"),
      response_var = "responseColor",
      target_var = "presentedColor"
    )
  ),

  # fit bmm mixture2p to exp1 data
  tar_target(exp1_2p_ss_bmm, fit_2p_ss_bmm1(exp1_data)),
  tar_target(exp1_2p_time_bmm, fit_2p_time_bmm1(exp1_data)),

  # fit bmm sdm to exp1 data
  tar_target(exp1_sdm_ss_ss_bmm, fit_sdm_ss_ss_bmm1(exp1_data)),
  tar_target(exp1_sdm_time_time_bmm, fit_sdm_time_time_bmm1(exp1_data))
  # tar_target(exp1_sdm_ss_time_bmm, fit_sdm_ss_time_bmm1(exp1_data)),
  # tar_target(exp1_sdm_time_ss_bmm, fit_sdm_time_ss_bmm1(exp1_data)),
)
