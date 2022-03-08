import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cmdstanpy import cmdstan_path, CmdStanModel
import random
import statistics

# Excercise 1
df = pd.read_csv('D:\Studia\DataAnalytics\lab1\Data1.csv')
df.head()

new_df = df.set_index(df.columns[0])

new_df.plot(subplots=True, layout=(6,1),title="All columns as time series")

new_df.plot.hist(bins=120,subplots=True, layout=(6,1),title="Histograms")

new_df.plot.kde(title="Kernel Denisty Estimators")

#2018
mask = (df["Unnamed: 0"] >= '2018-01-01') & (df["Unnamed: 0"] <= '2018-31-12')
df_2018 = df[mask] 
df_2018 = df_2018[['Unnamed: 0','theta_1','theta_2','theta_3','theta_4']]

print(df_2018)

# analysis for 2018
new_df_2018 = df_2018.set_index(df.columns[0])

new_df_2018.plot(subplots=True, layout=(6,1),title="All columns as time series - 2018")

new_df_2018.plot.hist(bins=120,subplots=True, layout=(6,1),title="Histograms - 2018")

new_df_2018.plot.kde(title="Kernel Denisty Estimators - 2018")

plt.show()

#Excercise 2

F = 7
L = 5

N = F + L

ones_list = [1]*L
zeros_list = [0]*F

y = ones_list + zeros_list


random.shuffle(y)
#print(y)

dataset = {'N' : N, 'y': y}

stan_path = 'D:\Studia\DataAnalytics\lab1\\bern_1.stan'
model = CmdStanModel(stan_file=stan_path)

#print(model)
sample_mod = model.sample(dataset)
extract = sample_mod.stan_variable('theta')
plt.hist(extract,bins=10)

mean_theta = extract.mean()
plt.axvline(mean_theta,color='r')

median_theta = statistics.median(extract)
plt.axvline(median_theta,color='b')

summarize_theta = sample_mod.summary()
theta95 = summarize_theta['95%']['theta']
plt.axvline(theta95,color='g')

theta5 = summarize_theta['5%']['theta']
plt.axvline(theta5,color='m')

plt.show()





