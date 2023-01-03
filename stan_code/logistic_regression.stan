// see following for vectorized logistic regression help
// https://mc-stan.org/docs/stan-users-guide/vectorization.html
data {
    int<lower=0> N;
    int<lower=1> Nattr; // number of covariates
    matrix[N,Nattr] X;  
    // array[N] vector[Nattr] X;

    array[N] int<lower=0> y;
    
    real<lower=0> prior_width;
}

parameters {
    vector[Nattr] beta;  // attribute effects 
}
model {
    beta ~ normal(0, prior_width);
    y ~ bernoulli_logit(X*beta);
}