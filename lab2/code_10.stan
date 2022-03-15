data {
    int<lower=1> N;
}

parameters {
    real y[N];
    real theta;
}

model {
    y ~ normal(0,1);
    theta ~ normal(0,1);
}

generated quantities {
   real mean_y =mean(y);
}