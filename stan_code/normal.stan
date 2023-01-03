// Stan code to fit normal distribution with improper prior

data {
  int<lower=0> N;
  real y[N];
}
parameters {
  real mu;
  real sigma;

}
model {
    y ~ normal(mu, sigma);
    // alternative way to write the above line
    // target += normal_lpdf(y | mu, sigma);
}
