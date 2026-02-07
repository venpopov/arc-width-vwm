
real sdm_algo4_lpdf(vector y, vector mu, vector c, vector kappa, vector delta, vector asigma, vector theta, array[] real arc) {
  real out = sdm_simple_lpdf(y | mu, c, kappa);
  
  int N = size(y);
  vector[N-1] arc_next;
  for (i in 1:(N-1)) {
    real exceeded = abs(y[i]) > arc[i] ? 1.0 : 0.0;
    arc_next[i] = arc[i] * (1 - delta[i]) + exceeded * pi() * delta[i];
  }

  // out += theta[1] .* normal_lpdf(logit(to_vector(arc[2:]) / pi()) - logit(arc_next / pi()) | 0, asigma[2:]);
  // out += (1-theta[1]) .* uniform_lpdf(arc[2:] | 0.0, pi());

    // likelihood of the mixture model
  for (n in 1:(N-1)) {
    array[2] real ps;
    real log_sum_exp_theta = log(exp(theta[n]) + 1);
    ps[1] = theta[n] - log_sum_exp_theta + normal_lpdf(logit(arc[n+1] / pi()) - logit(arc_next[n] / pi()) | 0, asigma[n+1]);
    ps[2] = - log_sum_exp_theta + uniform_lpdf(arc[n+1] | 0.0, pi());
    out += log_sum_exp(ps);
  }
  return(out);
}
