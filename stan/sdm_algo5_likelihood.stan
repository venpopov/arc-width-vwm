real sdm_algo5_lpdf(vector y, vector mu, vector c, vector kappa, vector tau, vector delta, vector sigmag, array[] real arc) {
  // SDM part for response errors
  real out = sdm_simple_lpdf(y | mu, c, kappa);

  int N = size(y);
  vector[N] p = to_vector(arc) / pi();
  vector[N] x = logit(p); 

  vector[N-1] mu_next;
  vector[N-1] sigma_next;

  for (i in 1:(N-1)) {
    real a_t = arc[i];
    real y_t = y[i];
    real x_t = x[i];
    
    // Learning rate decays with sqrt(trial number + 1)
    // Note: i corresponds to trial index t
    real d_t = delta[i] / sqrt(i + 1.0);
    
    // Gradient calculation
    real z_val = (a_t - abs(y_t)) / tau[i];
    real s_t = inv_logit(z_val); // equivalent to 1 / (1 + exp(-z))
    real g_t = -s_t + (pi() - a_t) * s_t * (1.0 - s_t) / tau[i];
    
    mu_next[i] = x_t + d_t * g_t;
    sigma_next[i] = d_t * sigmag[i];
  }

  // Transition probabilities
  // x[2:N] are the observed arcs for next trials
  out += normal_lpdf(x[2:N] | mu_next, sigma_next);
  
  return out;
}
