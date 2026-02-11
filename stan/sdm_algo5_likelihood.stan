real sdm_algo5_lpdf(vector y, vector mu, vector c, vector kappa, vector tau, vector delta, vector sigmag, array[] real arc, array[] real arc_next, array[] real trial) {
  // SDM part for response errors
  real out = sdm_simple_lpdf(y | mu, c, kappa);

  int N = size(y);
  // arc_next[i] holds the observed arc on the next trial; it is set to -1
  // for the last trial of each subject so the transition term is skipped,
  // preventing arc predictions from bleeding across subject boundaries.
  // trial[i] is the within-subject trial index used for learning rate decay.
  for (i in 1:N) {
    if (arc_next[i] > 0) {
      real a_t = arc[i];
      real y_t = y[i];
      real x_t = logit(a_t / pi());

      real d_t = delta[i] / sqrt(trial[i] + 1.0);

      real z_val = (a_t - abs(y_t)) / tau[i];
      real s_t = inv_logit(z_val);
      real g_t = -s_t + (pi() - a_t) * s_t * (1.0 - s_t) / tau[i];

      real mu_next = x_t + d_t * g_t;
      real sigma_next = d_t * sigmag[i];

      real x_next = logit(arc_next[i] / pi());
      out += normal_lpdf(x_next | mu_next, sigma_next);
    }
  }

  return out;
}
