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
            # expert note: this is *not quite* equivalent. the sampling statement with ~ drops the normalizing constants (e.g. 1/sqrt(2*pi) for a normal distribution), which is not a problem for the MCMC sampling/HMC since it works in unnormalized space, but is a problem for extracting log likelihoods from the model. The "actual" equivalent statement to the ~ statement is `normal_lupdf(beta | 0, 10)`, where the u stands for unnormalized. See: https://discourse.mc-stan.org/t/request-for-final-feedback-user-controlled-unnormalized-propto-distribution-syntax/16029/3 for more details. 

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