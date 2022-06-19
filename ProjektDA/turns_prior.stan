data {
    int N; // number of games
    vector[N] t; //number of turns for every game
}

generated quantities {
    real lambda = fabs(normal_rng(0,120));
    int y[N];
    for (k in 1:N) {
        y[k] = poisson_rng(lambda*t[k]);
    }
}