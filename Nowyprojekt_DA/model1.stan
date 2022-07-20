data {
  int G; //number of games
  int <lower = 1, upper = 365> N[G]; //openning
  int y[G]; 
}
parameters {
  vector[365] alpha;
}

transformed parameters {
  array [G] real theta;
  for (k in 1:G) {
    theta[k] = inv_logit(alpha[N[k]]);
  }
}

model {
  alpha ~ normal(0,1);
  for (k in 1:G) {
    y[k] ~ bernoulli(theta[k]);
  }
}

generated quantities {
  vector[G] y_sim;
  vector[G] log_lik;
  for (k in 1:G) {
    y_sim[k] = bernoulli_rng(theta[k]);
    log_lik[k]= bernoulli_lpmf(y[k]|theta[k]);
  }
}