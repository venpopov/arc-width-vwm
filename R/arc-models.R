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

algo4 <- function(delta, sd = 0) {
  force(delta)
  force(sd)

  \(a, y, ...) {
    a_star <- a * (1 - delta) + (abs(y) > a) * pi * delta
    out <- brms::logit_scaled(a_star / pi) + rnorm(1, sd = sd)
    brms::inv_logit_scaled(out) * pi
  }
}

algo5 <- function(tau = 0.05, delta0 = 0.1, delta_pow = 0.5, sd = 0) {
  force(tau)
  force(delta0)
  force(delta_pow)
  force(sd)

  \(a, y, i, ...) {
    # learning rate schedule
    delta <- delta0 / (i + 1)^delta_pow

    z <- (a - abs(y)) / tau
    s <- 1 / (1 + exp(-z))

    grad <- -s + (pi - a) * s * (1 - s) / tau
    a_star <- a + delta * grad
    out <- brms::logit_scaled(a_star / pi) + rnorm(1, sd = sd)
    brms::inv_logit_scaled(out) * pi
  }
}

algo5_meas <- function(tau = 0.05, delta0 = 0.1, delta_pow = 0.5, sd_meas = 0) {
  force(tau)
  force(delta0)
  force(delta_pow)
  force(sd_meas)

  # returns TWO values: latent a_star and observed a_obs
  function(a_latent, y, i, ...) {
    delta <- delta0 / (i + 1)^delta_pow
    z <- (a_latent - abs(y)) / tau
    s <- 1 / (1 + exp(-z))
    grad <- -s + (pi - a_latent) * s * (1 - s) / tau
    a_star <- a_latent + delta * grad
    a_star <- max(0, min(pi, a_star))

    # noisy observation
    eta <- logit(a_star / pi) + rnorm(1, sd = sd_meas)
    a_obs <- brms::inv_logit_scaled(eta) * pi

    list(a_latent = a_star, a_obs = a_obs)
  }
}
