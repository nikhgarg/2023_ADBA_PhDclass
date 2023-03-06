data {
    int<lower=0> N;
    int<lower=1> Nattr; // number of covariates
    matrix[N,Nattr] X;  
    array[N] real y;

    int<lower=0> N_test;
    matrix[N_test,Nattr] X_test;  
}

transformed data {
   /* ... declarations ... statements ... */
}

parameters {
    vector[Nattr] beta;  // attribute effects
    real <lower=0> sigma; // standard deviation of the noise
}

transformed parameters {
    // real <lower=0> sigmasquared = sigma*sigma; // variance of the noise
   /* ... declarations ... statements ... */
}

model {
    // Priors
    // beta ~ normal(0, 10);
    sigma ~ normal(0, 10);

    // Likelihood
    y ~ normal(X*beta, sigma);
}

generated quantities {
    # this is drawing samples from the posterior predictive distribution, so that we can check the fit
    array[N] real yposterior_pred;
    yposterior_pred = normal_rng(X*beta, sigma);
    // this will produce at the end a matrix of size N x #_of_iterations, that are samples of y, called posterior predictive distribution

    # this is the exact predictions for the test set
    vector[N_test] y_testset_pred;
    y_testset_pred = X_test*beta;
    // produce a matrix of size N_test x #_of_iterations, that are samples of y, that is the test set predictions at each iteration
}