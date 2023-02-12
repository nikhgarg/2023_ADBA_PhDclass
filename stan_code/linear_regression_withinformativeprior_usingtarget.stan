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
    target += normal_lpdf(beta | 0, 10); #equiv to the above line

    // sigma ~ normal(0, 10);
    target += normal_lpdf(sigma | 0, 10); #equiv to the above line

    // y ~ normal(X*beta, sigma);
    target += normal_lpdf(y | X*beta, sigma); #equiv to the above line

    // The target syntax directly adds the log density to the target, which is used for the MCMC sampling.
    // Note: you can use the target syntax to construct fancier models, like you will need for the zero-inflated model in the homework. 
    // See https://mc-stan.org/docs/stan-users-guide/zero-inflated.html for more examples. 
}

generated quantities {
    # this is drawing samples from the posterior predictive distribution, so that we can check the fit
    array[N] real yposterior_pred;
    yposterior_pred = normal_rng(X*beta, sigma);

    # this is the exact predictions for the test set
    vector[N_test] y_testset_pred;
    y_testset_pred = X_test*beta;
}