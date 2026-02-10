# TODO: generalize this function to work with any model
get_subject_parameters <- function(fit, exp = TRUE) {
  ranefs <- brms::ranef(fit)$subject[, 1, ]
  cols <- colnames(ranefs)
  fixefs <- brms::fixef(fit)[cols, 1]
  out <- lapply(cols, function(i) ranefs[, i] + fixefs[i])
  names(out) <- cols
  out <- as.data.frame(out)
  if (exp) out <- exp(out)
  out
}
