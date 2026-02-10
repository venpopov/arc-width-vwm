real sdm_algo4_lpdf(vector y, vector mu, vector c, vector kappa, vector delta, vector asigma, array[] real arc) {
  real out = sdm_simple_lpdf(y | mu, c, kappa);

  int N = size(y);
  vector[N] p = to_vector(arc) / pi();
  vector[N] x = logit(p);

  vector[N-1] p_det;
  for (i in 1:(N-1)) {
    real exceeded = abs(y[i]) > arc[i] ? 1.0 : 0.0;
    p_det[i] = p[i] * (1 - delta[i]) + exceeded * delta[i];
  }

  vector[N-1] z = (x[2:N] - logit(p_det)) ./ asigma[1:N-1];
  out += std_normal_lpdf(z) - sum(log(asigma[1:N-1]));
  return out;
}
