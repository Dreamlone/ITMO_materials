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


file = 'D:/ITMO/ITMO_materials/Ts_conference/reports/Results_daily_autots.csv'
df = pd.read_csv(file)

print('============================================')
print(f"MAE для AutoTs - {df['MAE'].mean()}")
print(f"MAPE для AutoTs - {df['MAPE'].mean()}\n")


for len_forecast in df['Len_forecast'].unique():
    df_local = df[df['Len_forecast'] == len_forecast]

    print(f"MAE для AutoTs -> {len_forecast} - {df_local['MAE'].mean()}")
    print(f"MAPE для AutoTs -> {len_forecast} - {df_local['MAPE'].mean()}")
    print(list(df_local['MAPE']), '\n')
