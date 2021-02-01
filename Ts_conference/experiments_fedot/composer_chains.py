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
import datetime

from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.ts_chain import TsForecastingChain
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

from fedot.core.composer.gp_composer.gp_composer import \
    GPComposerBuilder, GPComposerRequirements
from fedot.core.repository.quality_metrics_repository import \
    MetricsRepository, RegressionMetricsEnum


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    zero_indexes = np.argwhere(y_true == 0.0)
    for index in zero_indexes:
        y_true[index] = 0.01
    value = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return value


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

    available_model_types_primary = ['linear', 'ridge', 'lasso',
                                     'dtreg', 'knnreg']

    available_model_types_secondary = ['linear', 'ridge', 'lasso', 'rfr',
                                       'dtreg', 'knnreg', 'svr']

    composer_requirements = GPComposerRequirements(
        primary=available_model_types_primary,
        secondary=available_model_types_secondary, max_arity=5,
        max_depth=3, pop_size=10, num_of_generations=12,
        crossover_prob=0.8, mutation_prob=0.8,
        max_lead_time=datetime.timedelta(minutes=5),
        add_single_model_chains=True)

    metric_function = MetricsRepository().metric_by_id(
        RegressionMetricsEnum.RMSE)
    builder = GPComposerBuilder(task=task).with_requirements(
        composer_requirements).with_metrics(metric_function).with_initial_chain(
        chain)
    composer = builder.build()

    obtained_chain = composer.compose_chain(data=train_input,
                                            is_visualise=False)
    obtained_chain.__class__ = TsForecastingChain

    print('Obtained chain')
    obtained_models = []
    for node in obtained_chain.nodes:
        print(str(node))
        obtained_models.append(str(node))
    depth = int(obtained_chain.depth)
    print(f'Глубина цепочки {depth}')

    # Fit it
    obtained_chain.fit_from_scratch(train_input)

    # Predict
    predicted_values = obtained_chain.forecast(initial_data=train_input,
                                               supplementary_data=predict_input).predict

    return predicted_values, obtained_models, depth


# Читаем датафрейм с данными
df = pd.read_csv('D:/ITMO/ITMO_materials/Ts_conference/data/Daily_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
true_values = np.array(df['Level'])
print(len(true_values))

# df = pd.read_csv('D:/ITMO/ITMO_materials/Ts_conference/data/Hour_data.csv')
# true_values = np.array(df['Height'])
# true_values = true_values[-3784:]

if __name__ == '__main__':

    #################################
    #   ###  #  #    #    #  # ##   #
    #   #    ####   ###   #  ## #   #
    #   ###  #  #  #   #  #  #  #   #
    #################################
    considered_chain = TsForecastingChain()
    node_1_first = PrimaryNode('ridge')
    node_1_second = PrimaryNode('ridge')

    node_2_first = SecondaryNode('linear', nodes_from=[node_1_first])
    node_2_second = SecondaryNode('linear', nodes_from=[node_1_second])
    node_final = SecondaryNode('svr', nodes_from=[node_2_first,
                                                  node_2_second])
    considered_chain.add_node(node_final)

    print('Init chain')
    for node in considered_chain.nodes:
        print(str(node))

    # Длина прогноза в элементах
    l_forecasts = np.array([10, 50, 100, 150, 200, 400, 500, 700])
    all_chains = []
    all_sizes = []
    all_maes = []
    all_mapes = []
    all_lens = []
    all_times = []
    all_depths = []
    for len_forecast in l_forecasts:

        print(f'\nРассматриваемая длина прогноза {len_forecast} элементов')

        # Got train, test parts, and the entire data
        train_array = true_values[:-len_forecast]
        test_array = true_values[-len_forecast:]

        # Для каждого размера окна
        sizes = np.array([10, 50, 100, 150, 200, 400, 500, 700])
        for window_size in sizes:

            start = timeit.default_timer()
            predicted_values, m, d = make_forecast(chain=considered_chain,
                                                   train_data=true_values,
                                                   len_forecast=len_forecast,
                                                   max_window_size=window_size)
            time_launch = timeit.default_timer() - start

            chain_name = m
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
            all_depths.append(d)


result = pd.DataFrame({'Chain':all_chains,
                       'Size':all_sizes,
                       'MAE':all_maes,
                       'MAPE':all_mapes,
                       'Len_forecast': all_lens,
                       'Time': all_times,
                       'Depth': all_depths})
result.to_csv('D:/ITMO/ITMO_materials/Ts_conference/reports/Results_daily_composer.csv', index=False)