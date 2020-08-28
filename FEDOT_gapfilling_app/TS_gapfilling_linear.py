from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.composer.node import PrimaryNode, SecondaryNode
from core.composer.ts_chain import TsForecastingChain
from core.models.data import InputData, train_test_data_setup
from core.repository.dataset_types import DataTypesEnum
from core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from pylab import rcParams
rcParams['figure.figsize'] = 18, 7

# Проверка точности восстановления исходного ряда
### Input:
# parameter (str)        --- название столбца в датафрейме data, параметр, из которого сотавляется временной ряд
# mask (str)             --- название столбца в датафрейме data, который содержит бинарный код маски пропусков
# data (pd DataFrame)    --- датафрейм, в котором содержится вся необходимая информация
# withoutgap_arr (array) --- массив без пропусков
### Output:
# Функция выводит на экран значения трех метрик: MAE, RMSE, MedianAE
def validate(parameter, mask, data, withoutgap_arr, bad_value = -100.0):
    arr_parameter = np.array(data[parameter])
    arr_mask = np.array(data[mask])
    # В каких элементах присутствуют пропуски
    ids = np.ravel(np.argwhere(arr_mask == 0))

    true_values = arr_parameter[ids]
    predicted_values = withoutgap_arr[ids]
    print('Совокупный размер пропусков:', len(true_values))
    min_value = min(true_values)
    max_value = max(true_values)
    print('Минимальное значение в пропуске - ', min_value)
    print('Максимальное значение в пропуске- ', max_value)

    # Выводим на экран метрики
    MAE = mean_absolute_error(true_values, predicted_values)
    print('Mean absolute error -', round(MAE, 2))

    RMSE = (mean_squared_error(true_values, predicted_values)) ** 0.5
    print('RMSE -', round(RMSE, 2))

    MedianAE = median_absolute_error(true_values, predicted_values)
    print('Median absolute error -', round(MedianAE, 2), '\n')

    # Совмещение предсказанных значений с пропуском
    arr_parameter_modified = np.copy(arr_parameter)
    arr_parameter[arr_mask == 1] = bad_value # Изначальные значения в пропусках
    withoutgap_arr[arr_mask == 1] = bad_value # Заполненные значения в пропусках

    arr_parameter_modified[arr_mask == 0] = bad_value
    masked_array_1 = np.ma.masked_where(arr_parameter_modified == bad_value, arr_parameter_modified)
    masked_array_2 = np.ma.masked_where(arr_parameter == bad_value, arr_parameter)
    masked_array_3 = np.ma.masked_where(withoutgap_arr == bad_value, withoutgap_arr)

    if parameter == 'Mean Tmp':
        name = 'Среднесуточная температура воздуха, ℃'
    else:
        name = 'Количество осадков, мм'

    plt.plot(data['Date'], masked_array_1, c='blue', alpha=0.5)
    plt.plot(data['Date'], masked_array_2, c='green', alpha=0.5, label = 'Действительные значения')
    plt.plot(data['Date'], masked_array_3, c='red', alpha=0.5, label = 'Предсказанные значения')
    plt.ylabel(name, fontsize=15)
    plt.xlabel('Дата', fontsize=15)
    if mask != 'Small_mask':
        plt.xlim(data['Date'][ids[0]-100], data['Date'][ids[-1]+100])
    plt.grid()
    plt.legend(fontsize = 15)
    plt.show()


# Функция-обертка для заполнения пропусков во временных рядах с помощью фреймворка FEDOT
### Input:
# data (np array)       --- одномерный массив (временной ряд), в котором требуется заполнить пропуски
# max_window_size (int) --- размер скользящего окна
# gap_value (float)     --- флаг пропуска в массиве
### Output:
# timeseries (np array) --- временной ряд без пропусков
def fill_gaps(data, max_window_size = 100, gap_value = -100.0):

    # Поиск значений пропусков
    gap_list = np.ravel(np.argwhere(data == gap_value))

    # Думаю, этот фрагмент стоит значительно переделать
    # Нахождение пропусков по интервалам
    new_gap_list = []
    local_gaps = []
    for index, gap in enumerate(gap_list):
        if index == 0:
            local_gaps.append(gap_list[index])
        else:
            prev_gap = gap_list[index-1]
            if gap - prev_gap > 1:
                # Имеется разрыв между пропусками
                local_gaps.append(gap)
                new_gap_list.append(local_gaps)
                local_gaps = []
            else:
                local_gaps.append(gap)
    # Итеративно заполняем пропуски во временном ряду
    for gap in new_gap_list:

        # Для обучения используется весь временной ряд для пропуска
        timeseries_train_part = data[:gap[0]]

        # Адаптивная длина интервала прогноза
        len_gap = len(gap)
        forecast_length = len_gap

        task = Task(TaskTypesEnum.ts_forecasting,
                    TsForecastingParams(forecast_length=forecast_length,
                                        max_window_size=max_window_size,
                                        return_all_steps=True,
                                        make_future_prediction=True))

        x1 = np.arange(0, len(timeseries_train_part)) / 10
        x2 = np.arange(0, len(timeseries_train_part)) + 1
        exog_features = np.asarray([x1, x2]).T

        input_data = InputData(idx=np.arange(0, len(timeseries_train_part)),
                               features=exog_features,
                               target=timeseries_train_part,
                               task=task,
                               data_type=DataTypesEnum.ts)

        # Строим предсказания для пропущенной части во временном ряду
        chain = TsForecastingChain(PrimaryNode('ridge'))
        chain.fit_from_scratch(input_data)

        # "Тестовые данные" для того, чтобы делать предсказание на определенную длину
        test_data_imit = np.ones(len_gap)
        x1 = np.arange(0, len(test_data_imit)) / 10
        x2 = np.arange(0, len(test_data_imit)) + 1
        exog_features = np.asarray([x1, x2]).T
        test_data = InputData(idx=np.arange(0, len(test_data_imit)),
                              features=exog_features,
                              target=test_data_imit,
                              task=task,
                              data_type=DataTypesEnum.ts)

        predicted_values = chain.forecast(initial_data=input_data, supplementary_data=test_data).predict

        # Заменяем пропуски в массиве предсказанными значениями
        data[gap] = predicted_values

    return(data)


#                               ----- ПРИМЕНЕНИЕ АЛГОРИТМА -----                                     #
if __name__ == '__main__':
    # Для каких значений будет производится запуск модели
    mask_name = 'Small_mask'
    parameter = 'Mean Tmp'

    # Загружаем датасет с данными
    data = pd.read_csv('D:/FEDOT_timeseries/Spb_meteodata.csv', sep = ';')
    data['Date'] = pd.to_datetime(data['Date'])
    print(data.head(5), '\n')

    # Генерация пропуска в выбранном столбце
    arr_parameter = np.array(data[parameter])
    mask = np.array(data[mask_name])
    arr_parameter[mask == 0] = -100.0

    # Запускаем заполнение пропусков
    without_gaps = fill_gaps(arr_parameter, max_window_size = 50, gap_value = -100.0)

    # Проверка точности
    validate(parameter=parameter, mask=mask_name, data=data, withoutgap_arr=without_gaps)

