theme_Publication <- function(base_size = 14) {
  library(grid)
  library(ggthemes)
  (theme_foundation(base_size = base_size)
  + theme(
      plot.title = element_text(
        face = "bold",
        size = rel(1.2)
      ),
      text = element_text(),
      panel.background = element_rect(fill = "white"),
      plot.background = element_rect(fill = "white"),
      panel.border = element_rect(colour = NA),
      axis.title = element_text(face = "bold", size = rel(1)),
      axis.title.y = element_text(angle = 90, vjust = 2),
      axis.title.x = element_text(vjust = -0.2),
      axis.text = element_text(),
      axis.line = element_line(colour = "black"),
      axis.ticks = element_line(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      legend.key = element_rect(colour = NA),
      legend.key.size = unit(0.2, "cm"),
      legend.margin = unit(0, "cm"),
      legend.title = element_text(face = "italic"),
      plot.margin = unit(c(10, 5, 5, 5), "mm"),
      strip.background = element_rect(colour = "white", fill = "white"),
      strip.text = element_text(face = "bold")
    ))
}

#' Shift a vector by a specified number of positions
#'
#' @param x A vector to be shifted
#' @param by Integer specifying the number of positions to shift. Positive values
#'   shift right (add NAs at the beginning), negative values shift left (add NAs
#'   at the end). Default is 1.
#'
#' @return A vector of the same length as \code{x} with values shifted by \code{by}
#'   positions and NAs filling the empty positions.
#'
#' @examples
#' shift_vector(1:5, by = 2) # Returns: NA, NA, 1, 2, 3
#' shift_vector(1:5, by = -2) # Returns: 3, 4, 5, NA, NA
shift_vector <- function(x, by = 1) {
  nas <- rep(NA, abs(by))
  if (by > 0) {
    len <- length(x)
    c(nas, x[-((len - by + 1):len)])
  } else {
    c(x[-(1:abs(by))], nas)
  }
}


#' Calculate Likelihood Ratio for Expanding vs Contracting Arcs
#'
#' This function calculates the likelihood ratio of expanding arcs
#'   versus contracting arcs based on response accuracy data.
#'
#' @param dat A data frame containing columns 'arc' (full width of the arc) and
#'   'resperr' (response error).
#' @param count_by <[`tidy-select`][dplyr::dplyr_tidy_select]> Optional
#'   column(s) to group the counts by. Can be a single column, multiple columns
#'   using `c()`, or tidyselect helpers like `starts_with()`. Default is `NULL`,
#'   which uses existing grouping variables if the data is grouped.
#' @return A data frame with the likelihood ratio of expanding to contracting
#'   arcs. If `count_by` is specified or the data is grouped, results are
#'   returned for each combination of the grouping variables.
#'
#' @examples
#' # Basic usage
#' calculate_expand_likelihood_ratio(trial_data)
#'
#' # Group by a single column
#' calculate_expand_likelihood_ratio(trial_data, count_by = subject_id)
#'
#' # Group by multiple columns
#' calculate_expand_likelihood_ratio(trial_data, count_by = c(subject_id, condition))
#'
#' # Use existing grouping
#' trial_data |>
#'   group_by(subject_id) |>
#'   calculate_expand_likelihood_ratio()
#'
calculate_expand_likelihood_ratio <- function(dat, count_by = NULL) {
  # Capture the count_by expression
  count_by_quo <- rlang::enquo(count_by)

  dat <- dat |>
    mutate(
      prior_arc = shift_vector(arc, by = 1),
      prior_resperr = shift_vector(resperr, by = 1),
      resp_type = if_else(prior_arc >= 2 * abs(prior_resperr), "hit", "miss"),
      change_type = if_else(arc > prior_arc, "expanding", "contracting")
    )

  # Get the column names for count_by, or use grouping variables if NULL

  if (!rlang::quo_is_null(count_by_quo)) {
    count_by_names <- names(tidyselect::eval_select(count_by_quo, dat))
  } else {
    count_by_names <- dplyr::group_vars(dat)
  }
  count_by_syms <- rlang::syms(count_by_names)

  dat |>
    ungroup() |>
    count(resp_type, change_type, !!!count_by_syms) |>
    filter(!is.na(change_type)) |>
    complete(resp_type, change_type, !!!count_by_syms, fill = list(n = 0)) |>
    group_by(resp_type) |>
    mutate(n = n / sum(n)) |>
    spread(change_type, n) |>
    mutate(likelihood_ratio_expand = expanding / contracting) |>
    group_by(!!!dplyr::group_vars(dat))
}

plot_likelihood_ratio <- function(data) {
  data |>
    ggplot(aes(resp_type, likelihood_ratio_expand)) +
    geom_point(position = position_jitter(0.1), alpha = 0.3) +
    stat_summary(fun.data = \(x) mean_se(x, 2), color = "red") +
    geom_abline(intercept = 1, slope = 0, linetype = "dashed")
}
