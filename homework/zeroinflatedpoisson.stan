// Stan code to fit normal distribution with improper prior

data {
  int<lower=0> N;
  array[N] int<lower=0> y;
  array[N] real X1;
  array[N] real X2;
}
parameters {
  real<lower=-1, upper=1> alpha;
  real<lower=-1, upper=1> beta;
}
model {
  for (i in 1:N) {
    if (y[i] == 0) {
      target += 
        log_sum_exp(bernoulli_logit_lpmf(1 | alpha * X1[i])
        ,
          bernoulli_logit_lpmf(0 | alpha * X1[i]) + poisson_lpmf(y[i] | exp(X2[i] * beta)));
    }
    else {
      target += bernoulli_logit_lpmf(0 | alpha * X1[i]) + poisson_lpmf(y[i] | exp(X2[i] * beta));
    } 
  }
}

generated quantities { 
  // //For posterior predictive check 
    array[N] int yposterior_pred;
    array[N] int<lower=0> coin_flips;
    for (i in 1:N) {
      coin_flips[i] = bernoulli_logit_rng(alpha * X1[i]);
      if (coin_flips[i] == 0) {
        yposterior_pred[i] = poisson_rng(exp(X2[i] * beta));
      } 
      else {
        yposterior_pred[i] = 0;
      }
  }
} 
