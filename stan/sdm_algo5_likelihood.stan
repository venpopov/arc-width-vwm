
real sdm_algo5_lpdf(vector y, vector mu, vector c, vector kappa, vector delta, vector asigma, vector tau, array[] real arc) {
  real out = sdm_simple_lpdf(y | mu, c, kappa);
  
  int N = size(y);
  vector[N-1] arc_next;
  for (i in 1:(N-1)) {
    real d = delta[i] / (i)^0.5;
    real z = (arc[i] - abs(y[i])) / tau[i];
    real s = inv_logit(z);
    real grad = -s + (pi() - arc[i]) * s * (1 - s) / tau[i];
    arc_next[i] = arc[i] + d * grad;
  }

  out += normal_lpdf(logit(to_vector(arc[2:]) / pi()) - logit(arc_next / pi()) | 0, asigma[2:]);
  return(out);
}
