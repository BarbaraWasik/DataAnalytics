data {
    int N; // number of games
}

generated quantities {
    real lambda = fabs(normal_rng(0,120));
    int y[N];
    for (k in 1:N) {
        y[k] = poisson_rng(lambda);
    }
}