optimum_equation <- function(alpha, cdf, pdf) {
  (pi - alpha) * pdf(alpha) - cdf(alpha) + 0.5
}

# returns a data.frame with one row and two columns with the optimal alpha and phit
find_optimum_sdm <- function(c, kappa) {
  solution <- uniroot(
    optimum_equation,
    cdf = \(x) psdm(x, c = c, kappa = kappa),
    pdf = \(x) dsdm(x, c = c, kappa = kappa),
    lower = 0, upper = pi - 0.0001
  )
  data.frame(alpha = solution$root, phit = 2 * psdm(solution$root, c = c, kappa = kappa) - 1)
}

run_arc_algorithm <- function(arc_update_fun, c = 4, kappa = 3, init_arc = pi / 2, n_trials = 1e5) {
  y <- rsdm(n_trials, 0, c = c, kappa = kappa)

  a <- rep(0, n_trials)
  a[1] <- init_arc

  for (i in seq_len(n_trials - 1)) {
    hit <- sign((abs(y[i]) <= a[i]) - 0.5) # 1 if hit, -1 if miss
    a[i + 1] <- arc_update_fun(a = a[i], hit = hit, y = y[i], i = i)
  }
  list(phit = mean(abs(y) <= a), y = y, a = a)
}

# a convenience function to show how the algorithm performs on a single run
plot_algorithm_run <- function(out) {
  par(mfrow = c(1, 2))
  plot(out$a, type = "l", xlab = "trial", ylab = "Arc half-width")
  plot(cummean(abs(out$y) <= out$a), type = "l", xlab = "trial", ylab = "Cummulative hit rate")
  par(mfrow = c(1, 1))
}

algo1 <- function(delta) {
  force(delta)
  \(a, hit, ...) {
    a * (1 - hit * delta)
  }
}

algo2 <- function(delta) {
  force(delta)
  \(a, hit, ...) a * (1 - delta)^hit
}

algo3 <- function(delta) {
  force(delta)
  \(a, hit, y, ...) a + delta * (abs(y) - a)
}

algo4 <- function(delta) {
  force(delta)

  \(a, y, ...) {
    a * (1 - delta) + (abs(y) > a) * pi * delta
  }
}

algo5 <- function(tau = 0.025, delta0 = 0.1, delta_pow = 0.5) {
  force(tau)
  force(delta0)
  force(delta_pow)

  \(a, y, i, ...) {
    # learning rate schedule
    delta <- delta0 / (i + 1)^delta_pow

    z <- (a - abs(y)) / tau
    s <- 1 / (1 + exp(-z))

    grad <- -s + (pi - a) * s * (1 - s) / tau
    a + delta * grad
  }
}

algo5_noisy_grad <- function(tau = 0.05, delta0 = 0.1, delta_pow = 0.5, sigma_g = 1) {
  force(tau)
  force(delta0)
  force(delta_pow)
  force(sigma_g)

  \(a, y, i, ...) {
    delta <- delta0 / (i + 1)^delta_pow
    z <- (a - abs(y)) / tau
    s <- 1 / (1 + exp(-z))
    grad <- -s + (pi - a) * s * (1 - s) / tau

    x <- logit(a / pi)
    x_new <- x + delta * (grad + rnorm(1, sd = sigma_g))
    inv_logit(x_new) * pi
  }
}

# --- Algo4 with logit-scale noise ---

algo4_logit_noise <- function(delta = 0.1, sigma_e = 0.2) {
  force(delta)
  force(sigma_e)

  \(a, y, ...) {
    a_det <- a * (1 - delta) + (abs(y) > a) * pi * delta
    x_det <- logit(a_det / pi)
    x_new <- x_det + rnorm(1, sd = sigma_e)
    inv_logit(x_new) * pi
  }
}

algo4_loglik <- function(y, a, c, kappa, delta, sigma_e) {
  T_ <- length(y)
  x <- logit(a / pi)

  ll_y <- sum(dsdm(y, mu = 0, c = c, kappa = kappa, log = TRUE))

  idx <- seq_len(T_ - 1)
  a_det <- a[idx] * (1 - delta) + (abs(y[idx]) > a[idx]) * pi * delta
  x_det <- logit(a_det / pi)

  ll_a <- sum(dnorm(x[idx + 1], mean = x_det, sd = sigma_e, log = TRUE))

  ll_y + ll_a
}

par_to_unconstrained_algo4 <- function(pars) {
  c(log(pars$c), log(pars$kappa), logit(pars$delta), log(pars$sigma_e))
}

par_from_unconstrained_algo4 <- function(theta) {
  list(
    c = exp(theta[1]),
    kappa = exp(theta[2]),
    delta = inv_logit(theta[3]),
    sigma_e = exp(theta[4])
  )
}

neg_loglik_algo4 <- function(theta, y, a) {
  pars <- par_from_unconstrained_algo4(theta)
  ll <- tryCatch(
    algo4_loglik(y, a,
      c = pars$c, kappa = pars$kappa,
      delta = pars$delta, sigma_e = pars$sigma_e
    ),
    error = \(e) -Inf,
    warning = \(w) -Inf
  )
  if (!is.finite(ll)) {
    return(1e10)
  }
  -ll
}

# --- Log-likelihood for the noisy-gradient model (algo5) ---

algo5_loglik <- function(y, a, c, kappa, tau, delta0, delta_pow = 0.5, sigma_g) {
  T_ <- length(y)
  x <- logit(a / pi)

  ll_y <- sum(dsdm(y, mu = 0, c = c, kappa = kappa, log = TRUE))

  idx <- seq_len(T_ - 1)
  delta_t <- delta0 / (idx + 1)^delta_pow

  z <- (a[idx] - abs(y[idx])) / tau
  s <- 1 / (1 + exp(-z))
  grad <- -s + (pi - a[idx]) * s * (1 - s) / tau

  mu_x <- x[idx] + delta_t * grad
  sd_x <- delta_t * sigma_g

  ll_a <- sum(dnorm(x[idx + 1], mean = mu_x, sd = sd_x, log = TRUE))

  ll_y + ll_a
}

par_to_unconstrained <- function(pars) {
  log(unlist(pars[c("c", "kappa", "tau", "delta0", "sigma_g")]))
}

par_from_unconstrained <- function(theta) {
  vals <- exp(theta)
  list(
    c = vals[1], kappa = vals[2], tau = vals[3],
    delta0 = vals[4], sigma_g = vals[5]
  )
}

neg_loglik <- function(theta, y, a) {
  pars <- par_from_unconstrained(theta)
  ll <- tryCatch(
    algo5_loglik(y, a,
      c = pars$c, kappa = pars$kappa, tau = pars$tau,
      delta0 = pars$delta0, delta_pow = 0.5,
      sigma_g = pars$sigma_g
    ),
    error = \(e) -Inf,
    warning = \(w) -Inf
  )
  if (!is.finite(ll)) {
    return(1e10)
  }
  -ll
}
