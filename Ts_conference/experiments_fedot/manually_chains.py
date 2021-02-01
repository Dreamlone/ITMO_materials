import os
import pandas as pd
import numpy as np
from scipy import interpolate
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from matplotlib import pyplot as plt
import timeit
from pylab import rcParams
rcParams['figure.figsize'] = 18, 7
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
import statsmodels.api as sm
import pylab

from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.ts_chain import TsForecastingChain
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams



# Расчет метрики - cредняя абсолютная процентная ошибка
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    # У представленной ниже формулы есть недостаток, - если в массиве y_true есть хотя бы одно значение 0.0,
    # то по формуле np.mean(np.abs((y_true - y_pred) / y_true)) * 100 мы получаем inf, поэтому
    zero_indexes = np.argwhere(y_true == 0.0)
    for index in zero_indexes:
        y_true[index] = 0.01
    value = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return(value)


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


def make_forecast(chain, train_data, len_forecast: int, max_window_size: int):
    """
    Function for predicting values in a time series

    :param chain: TsForecastingChain object
    :param train_data: one-dimensional numpy array to train chain
    :param len_forecast: amount of values for predictions
    :param max_window_size: moving window size

    :return predicted_values: numpy array, forecast of model
    """

    # Here we define which task should we use, here we also define two main
    # hyperparameters: forecast_length and max_window_size
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast,
                                    max_window_size=max_window_size,
                                    return_all_steps=False,
                                    make_future_prediction=True))

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(train_data)),
                            features=None,
                            target=train_data,
                            task=task,
                            data_type=DataTypesEnum.ts)

    # Make a "blank", here we need just help FEDOT understand that the
    # forecast should be made exactly the "len_forecast" length
    predict_input = InputData(idx=np.arange(0, len_forecast),
                              features=None,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

    # Fit it
    chain.fit_from_scratch(train_input)

    # Predict
    predicted_values = chain.forecast(initial_data=train_input,
                                      supplementary_data=predict_input).predict

    return predicted_values


# Читаем датафрейм с данными
# df = pd.read_csv('D:/ITMO/ITMO_materials/Ts_conference/data/Daily_data.csv')
# df['Date'] = pd.to_datetime(df['Date'])
# true_values = np.array(df['Level'])
# print(len(true_values))

df = pd.read_csv('D:/ITMO/ITMO_materials/Ts_conference/data/Hour_data.csv')
true_values = np.array(df['Height'])
true_values = true_values[-3784:]

if __name__ == '__main__':

    #################################
    #   ###  #  #    #    #  # ##   #
    #   #    ####   ###   #  ## #   #
    #   ###  #  #  #   #  #  #  #   #
    #################################
    chain_1 = TsForecastingChain(PrimaryNode('ridge'))

    #################################
    #   ###  #  #    #    #  # ##   #
    #   #    ####   ###   #  ## #   #
    #   ###  #  #  #   #  #  #  #   #
    #################################
    node_1 = PrimaryNode('ridge')
    node_2 = PrimaryNode('xgbreg')
    node_3 = SecondaryNode('ridge', nodes_from=[node_1, node_2])
    chain_2 = TsForecastingChain(node_3)

    #################################
    #   ###  #  #    #    #  # ##   #
    #   #    ####   ###   #  ## #   #
    #   ###  #  #  #   #  #  #  #   #
    #################################
    node_first = PrimaryNode('trend_data_model')
    node_second = PrimaryNode('residual_data_model')

    # Define SecondaryNode models - its second level models
    node_trend_model = SecondaryNode('ridge', nodes_from=[node_first])
    node_residual_model = SecondaryNode('ridge', nodes_from=[node_second])

    # Root node - make final prediction
    node_final = SecondaryNode('svr', nodes_from=[node_trend_model,
                                                  node_residual_model])
    chain_3 = TsForecastingChain(node_final)

    # Длина прогноза в элементах
    l_forecasts = np.array([10, 50, 100, 150, 200, 400, 500, 700])
    all_chains = []
    all_sizes = []
    all_maes = []
    all_mapes = []
    all_lens = []
    all_times = []
    for len_forecast in l_forecasts:

        print(f'\nРассматриваемая длина прогноза {len_forecast} элементов')

        # Got train, test parts, and the entire data
        train_array = true_values[:-len_forecast]
        test_array = true_values[-len_forecast:]
        for considered_chain, chain_name in zip([chain_1,chain_2,chain_3],
                                                ['ridge', '3-node', '5-node']):
            # Для каждого размера окна
            sizes = np.array([10, 50, 100, 150, 200, 400, 500, 700])

            for window_size in sizes:

                start = timeit.default_timer()
                predicted_values = make_forecast(chain=considered_chain,
                                                 train_data=true_values,
                                                 len_forecast=len_forecast,
                                                 max_window_size=window_size)
                time_launch = timeit.default_timer() - start

                # plot_results(actual_time_series=true_values,
                #              predicted_values=predicted_values,
                #              len_train_data=len(train_array),
                #              y_name='Sea level, m')

                # Считаем метрики ошибки
                mae = mean_absolute_error(test_array, predicted_values)
                mape = mean_absolute_percentage_error(test_array, predicted_values)

                print(f'Цепочка {chain_name}, окно {window_size}, mape {mape}')

                # Производим запись результатов в вектор
                all_chains.append(chain_name)
                all_sizes.append(window_size)
                all_maes.append(mae)
                all_mapes.append(mape)
                all_lens.append(len_forecast)
                all_times.append(time_launch)


result = pd.DataFrame({'Chain':all_chains,
                       'Size':all_sizes,
                       'MAE':all_maes,
                       'MAPE':all_mapes,
                       'Len_forecast': all_lens,
                       'Time': all_times})
result.to_csv('D:/ITMO/ITMO_materials/Ts_conference/reports/Results_hour.csv',index=False)