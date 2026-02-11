real sdm_algo4_lpdf(vector y, vector mu, vector c, vector kappa, vector delta, vector asigma, array[] real arc, array[] real arc_next) {
  real out = sdm_simple_lpdf(y | mu, c, kappa);

  int N = size(y);
  // arc_next[i] holds the observed arc on the next trial; it is set to -1
  // for the last trial of each subject so the transition term is skipped,
  // preventing arc predictions from bleeding across subject boundaries
  for (i in 1:N) {
    if (arc_next[i] > 0) {
      real p_i = arc[i] / pi();
      real exceeded = abs(y[i]) > arc[i] ? 1.0 : 0.0;
      real p_det_i = p_i * (1 - delta[i]) + exceeded * delta[i];
      real x_next = logit(arc_next[i] / pi());
      out += normal_lpdf(x_next | logit(p_det_i), asigma[i]);
    }
  }
  return out;
}
