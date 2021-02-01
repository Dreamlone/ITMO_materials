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


def calculate_depth_clusters(depths):
    """
    Функция для расчета кластеров для глубины

    :param depths: массив со значениями глубины
    :return: массив с двумя "кластерами"
    """

    new_arr = []
    for i in depths:
        if i == 1:
            new_arr.append('1')
        else:
            new_arr.append('>1')

    return new_arr


def calculate_linearity(structures):
    """
    Функция рассчитывает "линейность" выбранного решения. Если в структуре цепочки
    есть хотя бы одна нелинейная модель, то считается, что цепочка моделирует
    нелинейные зависимости (или аппроксимирует линейные при помощи нелинейных)

    :param structures: список с моделями
    :return: массив с присвоенными лейблами
    """
    nl_models = ['svr', 'dtreg', 'knnreg', 'rfr']

    codes = []
    for i in structures:
        non_linear_models = 0
        for model in nl_models:
            if model in i:
                non_linear_models += 1

        if non_linear_models > 0:
            codes.append('Есть')
        else:
            codes.append('Нет')

    return codes


def plot_heat(df, len_col, window_col, target_col, cmap='jet', improve_neg=False):
    """
    Функция для отрисовки heat map

    :param df: датафрейм для процессинга
    :param len_col: длина прогноза
    :param window_col: размер скользящего окна
    :param target_col: отклик (или интересующая колонка)
    """

    len_arr = np.array(df[len_col].unique())
    window_arr = np.array(df[window_col].unique())

    # Пустая матрица
    empty_matrix = np.zeros((max(len_arr)+1, max(window_arr)+1))

    for i in len_arr:
        df_i = df[df[len_col] == i]
        for j in window_arr:
            try:
                df_ij = df_i[df_i[window_col] == j]
                target_value = np.array(df_ij[target_col])
                target_value = float(target_value[0])
                empty_matrix[i][j] = target_value
            except Exception:
                pass

    # Интерполяция
    masked_array = np.ma.masked_where(empty_matrix == 0.0, empty_matrix)
    x = np.arange(0, len(empty_matrix[0]))
    y = np.arange(0, len(empty_matrix))
    xx, yy = np.meshgrid(x, y)
    x1 = xx[~masked_array.mask]
    y1 = yy[~masked_array.mask]
    new_arr = masked_array[~masked_array.mask]
    int_matrix = interpolate.griddata((x1, y1), new_arr.ravel(), (xx, yy),
                                      method='nearest')

    # Отрисовка матрицы
    cmap = cm.get_cmap(cmap)
    plt.imshow(int_matrix, interpolation='bicubic', cmap=cmap)
    plt.colorbar(label=target_col)
    plt.ylabel(len_col, fontsize=15)
    plt.xlabel(window_col, fontsize=15)
    for i in len_arr:
        for j in window_arr:
            plt.scatter(j, i, c='black', s=2)
    plt.show()

    # Части, которые не были проинтерполированны - дозаполняются
    # интерполяцией методом ближайшего соседа
    ids = np.argwhere(empty_matrix == 0.0)
    for id in ids:
        interpolated_value = int_matrix[id[0], id[1]]
        if np.isnan(interpolated_value):
            pass
        else:
            empty_matrix[id[0], id[1]] = interpolated_value
    masked_array = np.ma.masked_where(empty_matrix == 0.0, empty_matrix)
    x = np.arange(0, len(empty_matrix[0]))
    y = np.arange(0, len(empty_matrix))
    xx, yy = np.meshgrid(x, y)
    x1 = xx[~masked_array.mask]
    y1 = yy[~masked_array.mask]
    new_arr = masked_array[~masked_array.mask]
    int_nearest = interpolate.griddata((x1, y1), new_arr.ravel(), (xx, yy),
                                       method='nearest')
    ids = np.argwhere(np.isnan(int_matrix))
    for id in ids:
        int_matrix[id[0], id[1]] = int_nearest[id[0], id[1]]

    # Удаление артефактов после интерполяции, если это требуется
    if improve_neg:
        # Границы, в которых осуществлялся поиск - от 1 до 4х
        int_matrix[int_matrix < 1.0] = 1.0
        int_matrix[int_matrix > 4.0] = 4.0
        int_matrix = int_matrix.round(0)

    # Отрисовка матрицы
    cmap = cm.get_cmap(cmap)
    plt.imshow(int_matrix, interpolation='bicubic', cmap=cmap)
    plt.colorbar(label=target_col)
    plt.ylabel(len_col, fontsize=15)
    plt.xlabel(window_col, fontsize=15)
    for i in len_arr:
        for j in window_arr:
            plt.scatter(j, i, c='black', s=2)
    plt.show()


file = 'D:/ITMO/ITMO_materials/Ts_conference/reports/Results_hour_composer.csv'
df = pd.read_csv(file)
print(f'Длина датафрема - {len(df)}')
df['Количество моделей в цепочке'] = calculate_depth_clusters(df['Depth'])
df['Присутствие нелинейных моделей'] = calculate_linearity(df['Chain'])
df = df.rename(columns={'Size': 'Размер скользящего окна',
                        'Len_forecast': 'Длина прогноза',
                        'Depth': 'Глубина цепочки'})
print(list(df.columns))

# Heat map №1
# plot_heat(df, len_col = 'Длина прогноза', window_col = 'Размер скользящего окна',
#           target_col = 'Глубина цепочки', cmap='coolwarm', improve_neg = True)

# Heat map №2
plot_heat(df, len_col = 'Длина прогноза', window_col = 'Размер скользящего окна',
          target_col = 'MAPE', cmap='coolwarm')


fig = px.scatter_3d(df, x='Длина прогноза',
                    y='Размер скользящего окна',
                    z='MAPE',
                    color='Присутствие нелинейных моделей')
fig.show()