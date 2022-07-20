data {
  int G; //number of games
  int <lower = 1, upper = 365> N[G]; //opening
  int P[G]; // winner pawn
}

generated quantities {
  vector[365] alpha;
  vector[2] beta;
  int y[G];

  for (k in 1:365) {
    alpha[k] = normal_rng(0,1);
  }

  for (i in 1:2) {
    beta[i] = normal_rng(0,1);
  }

  for (k in 1:G) {
    y[k] = bernoulli_rng(inv_logit(alpha[N[k]] + beta[P[k]]));
  }
}