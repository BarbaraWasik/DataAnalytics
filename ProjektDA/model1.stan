data {
  int M; //number of games analyzed
  int y[M]; //number of openings
}
parameters {
  real<lower=0>lambda;
}
model {
  lambda ~ normal(0,120);
  for (k in 1:M) {
    y[k] ~ poisson(lambda);
  }
}
generated quantities {
  vector[M] log_lik;
  int y_sim[M];
  for (k in 1:M) {
    log_lik[k]= poisson_lpmf(y[k] | lambda);
    y_sim[k] = poisson_rng(lambda);
  }
}