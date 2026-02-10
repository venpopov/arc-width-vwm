fit_bmm1 <- function(data) {
  # fit the model
  formula <- bmm::bmf(c ~ 1 + (1 | subject), kappa ~ 1 + (1 | subject))
  model <- bmm::sdm("resperr")
  bmm::bmm(formula, data, model, backend = "cmdstanr", cores = 4)
}

fit_bmm2 <- function(data) {
  # fit the model
  data$session <- as.factor(data$session)
  formula <- bmm::bmf(
    c ~ 0 + session + (0 + session || subject),
    kappa ~ 0 + session + (0 + session || subject)
  )
  model <- bmm::sdm("resperr")
  bmm::bmm(formula, data, model, backend = "cmdstanr", cores = 4)
}


fit_bmm3 <- function(data) {
  # fit the model
  data$session <- as.factor(data$session)
  formula <- bmm::bmf(
    c ~ 0 + session + (0 + session || subject),
    kappa ~ 1 + (1 | subject)
  )
  model <- bmm::sdm("resperr")
  bmm::bmm(formula, data, model, backend = "cmdstanr", cores = 4)
}

fit_2p_ss_bmm1 <- function(data) {
  data <- data |> 
    dplyr::mutate(setsize = as.factor(setsize)) |> 
    dplyr::filter(exp_type == "SetS")

  formula <- bmm::bmf(
    thetat ~ 0 + setsize + (0 + setsize || subject),
    kappa ~ 0 + setsize + (0 + setsize || subject)
  )
  model <- bmm::mixture2p("resperr")
  bmm::bmm(formula, data, model, backend = "cmdstanr", cores = 4)
}

fit_2p_time_bmm1 <- function(data) {
  data <- data |> 
    dplyr::mutate(encodingtime = as.factor(encodingtime),
                  delay = as.factor(delay)) |>
    dplyr::filter(exp_type == "Time")

  formula <- bmm::bmf(
    thetat ~ 0 + encodingtime:delay + (0 + encodingtime:delay || subject),
    kappa ~ 0 + encodingtime:delay + (0 + encodingtime:delay || subject)
  )
  model <- bmm::mixture2p("resperr")
  bmm::bmm(formula, data, model, backend = "cmdstanr", cores = 4)
}

fit_2p_ml <- function(data, by = NULL, ...) {
  withr::local_package("dplyr")
  data |> 
    group_by(across(all_of(by))) |> 
    mutate(id_var = cur_group_id()) |> 
    do({
      mixtur::fit_mixtur(
        data = .,
        model = "2_component",
        unit = "radians",
        id_var = "id_var",
        ...
      )
    }) |>
    select(-id) |> 
    rename(pmem = p_t) |>
    mutate(sd = bmm::k2sd(kappa)) |>
    ungroup()
}


fit_sdm_ss_ss_bmm1 <- function(data) {
  data <- data |> 
    dplyr::mutate(setsize = as.factor(setsize)) |> 
    dplyr::filter(part1_type == "SetS" & part2_type == "SetS")

  formula <- bmm::bmf(
    c ~ 0 + experimentorder:setsize + (0 + experimentorder:setsize || subject),
    kappa ~ 0 + experimentorder:setsize + (0 + experimentorder:setsize || subject)
  )
  model <- bmm::sdm("resperr")
  bmm::bmm(formula, data, model, backend = "cmdstanr", cores = 4, sort_data = TRUE)
}

fit_sdm_time_time_bmm1 <- function(data) {
  data <- data |> 
    dplyr::mutate(encodingtime = as.factor(encodingtime),
                  delay = as.factor(delay)) |>
    dplyr::filter(part1_type == "Time" & part2_type == "Time")

  formula <- bmm::bmf(
    c ~ 0 + experimentorder:encodingtime:experimentorder:delay + (0 + experimentorder:encodingtime:experimentorder:delay || subject),
    kappa ~ 0 + experimentorder:encodingtime:experimentorder:delay + (0 + experimentorder:encodingtime:experimentorder:delay || subject)
  )
  model <- bmm::sdm("resperr")
  bmm::bmm(formula, data, model, backend = "cmdstanr", cores = 4, sort_data = TRUE)
}