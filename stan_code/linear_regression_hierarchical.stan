data {
    int<lower=0> N;
    int<lower=0> N_test;
    int<lower=0> N_types;

    int<lower=1> Nattr; // number of covariates
    matrix[N,Nattr] X;  
    matrix[N_test,Nattr] X_test;  
    array[N] real y;
    array[N] int types; 
    array[N_test] int types_test; 
    }

parameters {
    vector[Nattr] beta;  // attribute effects

    // attribute effects per type
    array[N_types] vector[Nattr] beta_types;

    real <lower=0> sigma; // standard deviation of the noise
}
model {
    beta ~ normal(0, 10);

    #hierarchical prior on beta_types
    for (i in 1:N_types) {
        beta_types[i] ~ normal(beta, 10);
    }

    sigma ~ normal(0, 10);

    # likelihood for each observation
    for (i in 1:N) {
        y[i] ~ normal(X[i]*beta_types[types[i]], sigma);
    }
}

generated quantities {
    # this is drawing samples from the posterior predictive distribution, so that we can check the fit
    array[N] real yposterior_pred;
    for (i in 1:N) {
        yposterior_pred[i] = normal_rng(X[i]*beta_types[types[i]], sigma);
    }

    # this is the exact predictions for the test set
    vector[N_test] y_testset_pred;
    for (i in 1:N_test) {
        y_testset_pred[i] = X_test[i]*beta_types[types_test[i]];
    }
}