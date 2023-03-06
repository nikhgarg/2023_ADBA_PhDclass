// Stan code to fit normal distribution with improper prior

data {
  int<lower=0> N;
  int<lower=0> y[N];
  matrix[N,2] X;
}
parameters {
  real intercept;
  vector[2] alpha_beta;

}
model {
  intercept ~ normal(0, 2);
  alpha_beta ~ uniform(-1, 1);
  target += poisson_log_glm_lpmf(y | X, intercept, alpha_beta);
}

generated quantities {
  # posterior predictive distribution 

  array[N] int yposterior_pred;
  for (i in 1:N)
    yposterior_pred[i] = poisson_log_rng(intercept + dot_product(X[i], alpha_beta));
}