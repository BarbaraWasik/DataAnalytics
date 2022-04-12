data {
    int<lower=1> N;
    int<lower=1> M;

    matrix[N, M] X;
    int<lower=0, upper = 1> y[N];
}

transformed data {
   vector[N] ones_N = rep_vector(1,N);
}

parameters {
   vector[M] beta;
   real alpha;
}

model {
   beta ~ normal(0,2); 
   alpha ~ normal(0,2);
   y ~ bernoulli_logit(X*beta+alpha);
}

generated quantities {
   int y_ppc[N] = bernoulli_logit_rng(X*beta+ones_N*alpha);
   vector[N] prob_ppc = inv_logit(X*beta+ones_N*alpha);
}