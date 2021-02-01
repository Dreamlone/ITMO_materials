import os
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import seaborn as sns
from pylab import rcParams
rcParams['figure.figsize'] = 12, 6
import plotly.express as px
from scipy import interpolate
import scipy


file_1 = 'D:/ITMO/ITMO_materials/Ts_conference/reports/Results_daily_composer.csv'
file_2 = 'D:/ITMO/ITMO_materials/Ts_conference/reports/Results_day.csv'

df_1 = pd.read_csv(file_1)
df_2 = pd.read_csv(file_2)

################################################################################
#              Датафрейм с данными для композитной модели                      #
################################################################################
print('============================================')
print(f"MAE для композитной модели - {df_1['MAE'].mean()}")
print(f"MAPE для композитной модели - {df_1['MAPE'].mean()}\n")

best_mae = 5000
best_mape = 5000
opt_mae_size = 10
opt_mape_size = 10
for window_size in df_1['Size'].unique():
    df_local = df_1[df_1['Size'] == window_size]

    print(f"MAE для композитной модели -> {window_size} - {df_local['MAE'].mean()}")
    print(f"MAPE для композитной модели -> {window_size} - {df_local['MAPE'].mean()}")

    if df_local['MAE'].mean() < best_mae:
        best_mae = df_local['MAE'].mean()
        opt_mae_size = window_size
    if df_local['MAPE'].mean() < best_mape:
        best_mape = df_local['MAPE'].mean()
        opt_mape_size = window_size

print(f'Лучший результат по метрике MAE у окна {opt_mae_size} - {best_mae:.3f}')
print(f'Лучший результат по метрике MAPE у окна {opt_mape_size} - {best_mape:.2f}')
print('\n')
################################################################################
#              Датафрейм с данными для композитной модели                      #
################################################################################

for model in df_2['Chain'].unique():
    df_local = df_2[df_2['Chain'] == model]

    print('============================================')
    print(f"MAE для {model} - {df_local['MAE'].mean()}")
    print(f"MAPE для {model} - {df_local['MAPE'].mean()}\n")

    best_mae = 5000
    best_mape = 5000
    opt_mae_size = 10
    opt_mape_size = 10
    for window_size in df_local['Size'].unique():
        df_local_local = df_local[df_local['Size'] == window_size]

        print(f"MAE для {model} -> {window_size} - {df_local_local['MAE'].mean()}")
        print(f"MAPE для {model} -> {window_size} - {df_local_local['MAPE'].mean()}")

        if df_local_local['MAE'].mean() < best_mae:
            best_mae = df_local_local['MAE'].mean()
            opt_mae_size = window_size
        if df_local_local['MAPE'].mean() < best_mape:
            best_mape = df_local_local['MAPE'].mean()
            opt_mape_size = window_size

    print(f'Лучший результат по метрике MAE у окна {opt_mae_size} - {best_mae:.3f}')
    print(f'Лучший результат по метрике MAPE у окна {opt_mape_size} - {best_mape:.2f}')
    print('\n')