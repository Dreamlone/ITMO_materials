import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    median_absolute_error
from matplotlib import pyplot as plt
import timeit
from pylab import rcParams

rcParams['figure.figsize'] = 18, 7
import warnings

warnings.filterwarnings('ignore')

from autots import AutoTS


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    # У представленной ниже формулы есть недостаток, - если в массиве y_true есть хотя бы одно значение 0.0,
    # то по формуле np.mean(np.abs((y_true - y_pred) / y_true)) * 100 мы получаем inf, поэтому
    zero_indexes = np.argwhere(y_true == 0.0)
    for index in zero_indexes:
        y_true[index] = 0.01
    value = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return (value)


def plot_results(actual_time_series, predicted_values, len_train_data,
                 y_name='Parameter'):
    """
    Function for drawing plot with predictions

    :param actual_time_series: the entire array with one-dimensional data
    :param predicted_values: array with predicted values
    :param len_train_data: number of elements in the training sample
    :param y_name: name of the y axis
    """

    plt.plot(np.arange(0, len(actual_time_series)),
             actual_time_series, label='Actual values', c='green')
    plt.plot(np.arange(len_train_data, len_train_data + len(predicted_values)),
             predicted_values, label='Predicted', c='blue')
    # Plot black line which divide our array into train and test
    plt.plot([len_train_data, len_train_data],
             [min(actual_time_series), max(actual_time_series)], c='black',
             linewidth=1)
    plt.ylabel(y_name, fontsize=15)
    plt.xlabel('Time index', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid()
    plt.show()


def make_forecast(df, len_forecast: int):
    """
    Function for predicting values in a time series

    """

    model_name = 'AutoTs model'

    model = AutoTS(forecast_length=len_forecast,
                   frequency='infer',
                   prediction_interval=0.9,
                   ensemble='all',
                   model_list="superfast",
                   max_generations=15,
                   num_validations=2,
                   validation_method="backwards")

    model = model.fit(df,
                      date_col='Date',
                      value_col='Value')

    prediction = model.predict()
    # point forecasts dataframe
    forecasts_df = prediction.forecast

    predicted_values = np.array(forecasts_df['Value'])
    return predicted_values, model_name


# Читаем датафрейм с данными
df = pd.read_csv('D:/ITMO/ITMO_materials/Ts_conference/data/Daily_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
true_values = np.array(df['Level'])
dates = df['Date']
print(len(true_values))

# df = pd.read_csv('D:/ITMO/ITMO_materials/Ts_conference/data/Hour_data.csv')
# df['Date'] = pd.to_datetime(df['Date'])
# true_values = np.array(df['Height'])
# true_values = true_values[-3784:]
# dates = df['Date'][-3784:]


# Длина прогноза в элементах
l_forecasts = np.array([10, 50, 100, 150, 200, 400, 500, 700])
all_maes = []
all_mapes = []
all_lens = []
all_times = []
for len_forecast in l_forecasts:

    print(f'\nРассматриваемая длина прогноза {len_forecast} элементов')

    # Got train, test parts, and the entire data
    train_dates = dates[:-len_forecast]
    train_array = true_values[:-len_forecast]
    test_array = true_values[-len_forecast:]

    dataframe_process = pd.DataFrame({'Date': train_dates,
                                      'Value': train_array})
    dataframe_process['Date'] = pd.to_datetime(dataframe_process['Date'])

    # Для каждого размера окна
    sizes = np.array([10, 50, 100, 150, 200, 400, 500, 700])

    for _ in sizes:
        start = timeit.default_timer()
        predicted_values, model_name = make_forecast(dataframe_process,
                                                     len_forecast=len_forecast)
        time_launch = timeit.default_timer() - start

        #         plot_results(actual_time_series=true_values,
        #                      predicted_values=predicted_values,
        #                      len_train_data=len(train_array),
        #                      y_name='Sea level, m')

        # Считаем метрики ошибки
        mae = mean_absolute_error(test_array, predicted_values)
        mape = mean_absolute_percentage_error(test_array, predicted_values)

        print(
            f'Цепочка {model_name}, длина прогноза {len_forecast}, mape {mape}')

        # Производим запись результатов в вектор
        all_maes.append(mae)
        all_mapes.append(mape)
        all_lens.append(len_forecast)
        all_times.append(time_launch)

result = pd.DataFrame({'MAE': all_maes,
                       'MAPE': all_mapes,
                       'Len_forecast': all_lens,
                       'Time': all_times})
result.to_csv('D:/ITMO/ITMO_materials/Ts_conference/reports/Results_daily_autots.csv',
              index=False)