data {
  int G; //number of games
  int <lower = 1, upper = 365> N[G]; //opening
}

generated quantities {
  vector[365] alpha;
  int y[G];

  for (k in 1:365) {
    alpha[k] = normal_rng(0,0.5);
  }

  for (k in 1:G) {
    y[k] = bernoulli_rng(inv_logit(alpha[N[k]]));
  }
}