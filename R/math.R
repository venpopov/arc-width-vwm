logit <- function(p) {
  log(p) - log1p(-p)
}

inv_logit <- function(x) {
  1 / (1 + exp(-x))
}
