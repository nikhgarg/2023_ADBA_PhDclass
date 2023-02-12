data {
    int<lower=0> N;
    int<lower=0> N_test;
    int<lower=1> Nattr; // number of covariates
    matrix[N,Nattr] X;  
    matrix[N_test,Nattr] X_test;  
    array[N] real y;
    }

parameters {
    vector[Nattr] beta;  // attribute effects
    real <lower=0> sigma; // standard deviation of the noise
}
model {
    // beta ~ normal(0, 10);
    sigma ~ normal(0, 10);
    y ~ normal(X*beta, sigma);
}

generated quantities {
    # this is drawing samples from the posterior predictive distribution, so that we can check the fit
    array[N] real yposterior_pred;
    yposterior_pred = normal_rng(X*beta, sigma);

    # this is the exact predictions for the test set
    vector[N_test] y_testset_pred;
    y_testset_pred = X_test*beta;
}