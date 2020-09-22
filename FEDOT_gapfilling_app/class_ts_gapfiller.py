import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

from core.composer.ts_chain import TsForecastingChain
from core.composer.visualisation import ComposerVisualiser
from core.composer.chain import Chain
from core.composer.gp_composer.fixed_structure_composer import FixedStructureComposer
from core.composer.gp_composer.gp_composer import GPComposerRequirements
from core.composer.node import PrimaryNode, SecondaryNode
from core.models.data import InputData, OutputData
from core.repository.dataset_types import DataTypesEnum
from core.repository.quality_metrics_repository import MetricsRepository, RegressionMetricsEnum
from core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


class Gapfiller:
    """
    Base class used for filling in the gaps in time series with simple methods.
    Methods from the Gapfiller class can be used for comparison with more complex models in class TSGapfiller

    :param gap_value: value, which identify gap elements in array
    """

    def __init__(self, gap_value: float = -100.0):
        self.gap_value = gap_value

    def _parse_gap_ids(self, gap_list: list) -> list:
        """
        Method allows to parse source array with gaps indexes

        :param gap_list: array with indexes of gaps in array
        :return: a list with separated gaps in continuous intervals
        """

        new_gap_list = []
        local_gaps = []
        for index, gap in enumerate(gap_list):
            if index == 0:
                local_gaps.append(gap)
            else:
                prev_gap = gap_list[index - 1]
                if gap - prev_gap > 1:
                    # There is a "gap" between gaps
                    new_gap_list.append(local_gaps)

                    local_gaps = []
                    local_gaps.append(gap)
                else:
                    local_gaps.append(gap)
        new_gap_list.append(local_gaps)

        return (new_gap_list)

    def linear_interpolation(self, input_data):
        """
        Method allows to restore missing values in an array using linear interpolation

        :param input_data: array with gaps
        :return: array without gaps
        """

        try:
            output_data = np.array(input_data)
        except Exception:
            raise ValueError('input data should be one-dimensional array')

        # The indices of the known elements
        non_nan = np.ravel(np.argwhere(output_data != self.gap_value))
        # All known elements in the array
        masked_array = output_data[non_nan]
        f_interploate = interpolate.interp1d(non_nan, masked_array)
        x = np.arange(0, len(output_data))
        output_data = f_interploate(x)
        return (output_data)

    def local_poly_approximation(self, input_data, degree: int = 2, n_neighbors: int = 5):
        """
        Method allows to restore missing values in an array using Savitzky-Golay filter

        :param input_data: array with gaps
        :param degree: degree of a polynomial function
        :param n_neighbors: the number of neighboring known elements of the time series that the approximation is based on
        :return: array without gaps
        """

        try:
            output_data = np.array(input_data)
        except Exception:
            raise ValueError('input data should be one-dimensional array')

        i_gaps = np.ravel(np.argwhere(output_data == self.gap_value))

        # Iterately fill in the gaps in the time series
        for gap_index in i_gaps:
            # Indexes of known elements (updated at each iteration)
            i_known = np.argwhere(output_data != self.gap_value)
            i_known = np.ravel(i_known)

            # Based on the indexes we calculate how far from the gap the known values are located
            id_distances = np.abs(i_known - gap_index)

            # Now we know the indices of the smallest values in the array, so sort indexes
            sorted_idx = np.argsort(id_distances)
            nearest_values = []
            nearest_indices = []
            for i in sorted_idx[:n_neighbors]:
                time_index = i_known[i]
                nearest_values.append(output_data[time_index])
                nearest_indices.append(time_index)
            nearest_values = np.array(nearest_values)
            nearest_indices = np.array(nearest_indices)

            local_coefs = np.polyfit(nearest_indices, nearest_values, degree)
            est_value = np.polyval(local_coefs, gap_index)
            output_data[gap_index] = est_value

        return (output_data)

    def batch_poly_approximation(self, input_data, degree: int = 3, n_neighbors: int = 10):
        """
        Method allows to restore missing values in an array using batch polynomial approximations.
        Approximation is applied not for individual omissions, but for intervals of omitted values

        :param input_data: array with gaps
        :param degree: degree of a polynomial function
        :param n_neighbors: the number of neighboring known elements of the time series that the approximation is based on
        :return: array without gaps
        """

        try:
            output_data = np.array(input_data)
        except Exception:
            raise ValueError('input data should be one-dimensional array')

        # Gap indices
        gap_list = np.ravel(np.argwhere(output_data == self.gap_value))
        new_gap_list = self._parse_gap_ids(gap_list)

        # Iterately fill in the gaps in the time series
        for gap in new_gap_list:
            # Find the center point of the gap
            center_index = int((gap[0] + gap[-1]) / 2)

            # Indexes of known elements (updated at each iteration)
            i_known = np.argwhere(output_data != self.gap_value)
            i_known = np.ravel(i_known)

            # Based on the indexes we calculate how far from the gap the known values are located
            id_distances = np.abs(i_known - center_index)

            # Now we know the indices of the smallest values in the array, so sort indexes
            sorted_idx = np.argsort(id_distances)

            # Nearest known values to the gap
            nearest_values = []
            # And their indexes
            nearest_indices = []
            for i in sorted_idx[:n_neighbors]:
                # Getting the index value for the series - output_data
                time_index = i_known[i]
                # Using this index, we get the value of each of the "neighbors"
                nearest_values.append(output_data[time_index])
                nearest_indices.append(time_index)
            nearest_values = np.array(nearest_values)
            nearest_indices = np.array(nearest_indices)

            # Local approximation by an n-th degree polynomial
            local_coefs = np.polyfit(nearest_indices, nearest_values, degree)

            # Estimate our interval according to the selected coefficients
            est_value = np.polyval(local_coefs, gap)
            output_data[gap] = est_value

        return (output_data)

class TSGapfiller(Gapfiller):
    """
    Class used for filling in the gaps in time series

    :param gap_value: value, which mask gap elements in array
    """

    def inverse_ridge(self, input_data, max_window_size: int = 50):
        """
        Method fills in the gaps in the input array

        :param input_data: data with gaps to filling in the gaps in it
        :param max_window_size: window length
        :return: array without gaps
        """

        try:
            output_data = np.array(input_data)
        except Exception:
            raise ValueError('input data should be one-dimensional array')

        # Gap indices
        gap_list = np.ravel(np.argwhere(output_data == self.gap_value))
        new_gap_list = self._parse_gap_ids(gap_list)

        # Iterately fill in the gaps in the time series
        for index, gap in enumerate(new_gap_list):

            preds = []
            weights = []
            # Two predictions are generated for each gap - forward and backward
            for prediction in ['direct', 'inverse']:

                # The entire time series is used for training until the gap
                if prediction == 'direct':
                    timeseries_train_part = output_data[:gap[0]]
                elif prediction == 'inverse':
                    if index == len(new_gap_list) - 1:
                        timeseries_train_part = output_data[(gap[-1] + 1):]
                    else:
                        next_gap = new_gap_list[index + 1]
                        timeseries_train_part = output_data[(gap[-1] + 1):next_gap[0]]
                    timeseries_train_part = np.flip(timeseries_train_part)

                # Adaptive prediction interval length
                len_gap = len(gap)
                forecast_length = len_gap

                task = Task(TaskTypesEnum.ts_forecasting,
                            TsForecastingParams(forecast_length=forecast_length,
                                                max_window_size=max_window_size,
                                                return_all_steps=True,
                                                make_future_prediction=True))

                input_data = InputData(idx=np.arange(0, len(timeseries_train_part)),
                                       features=None,
                                       target=timeseries_train_part,
                                       task=task,
                                       data_type=DataTypesEnum.ts)

                # Making predictions for the missing part in the time series
                chain = TsForecastingChain(PrimaryNode('ridge'))
                chain.fit_from_scratch(input_data)

                # "Test data" for making prediction for a specific length
                test_data = InputData(idx=np.arange(0, len_gap),
                                      features=None,
                                      target=None,
                                      task=task,
                                      data_type=DataTypesEnum.ts)

                predicted_values = chain.forecast(initial_data=input_data, supplementary_data=test_data).predict

                if prediction == 'direct':
                    weights.append(np.arange(len_gap, 0, -1))
                    preds.append(predicted_values)
                elif prediction == 'inverse':
                    predicted_values = np.flip(predicted_values)
                    weights.append(np.arange(1, (len_gap + 1), 1))
                    preds.append(predicted_values)

            preds = np.array(preds)
            weights = np.array(weights)
            result = np.average(preds, axis=0, weights=weights)

            # Replace gaps in an array with predicted values
            output_data[gap] = result

        return (output_data)

    def composite_fill_gaps(self, input_data, max_window_size: int = 50):

        # Функция определения цепочки из моделей
        def get_composite_chain():
            chain = Chain()
            node_linear = PrimaryNode('linear')
            node_linear.labels = ["fixed"]
            node_rfr = PrimaryNode('rfr')
            node_rfr.labels = ["fixed"]

            node_final = SecondaryNode('lasso', nodes_from=[node_linear, node_rfr])
            node_final.labels = ["fixed"]
            chain.add_node(node_final)
            return(chain)

        try:
            output_data = np.array(input_data)
        except Exception:
            raise ValueError('input data should be one-dimensional array')

        # Gap indices
        gap_list = np.ravel(np.argwhere(output_data == self.gap_value))
        new_gap_list = self._parse_gap_ids(gap_list)

        # Iterately fill in the gaps in the time series
        for index, gap in enumerate(new_gap_list):
            # The entire time series is used for training until the gap
            timeseries_train_part = output_data[:gap[0]]

            # Adaptive prediction interval length
            len_gap = len(gap)
            forecast_length = 1

            # specify the task to solve
            task_to_solve = Task(TaskTypesEnum.ts_forecasting,
                                 TsForecastingParams(forecast_length=forecast_length,
                                                     max_window_size=max_window_size))

            train_data = InputData(idx=np.arange(0, len(timeseries_train_part)),
                                   features=None,
                                   target=timeseries_train_part,
                                   task=task_to_solve,
                                   data_type=DataTypesEnum.ts)

            # "Test data" for making prediction for a specific length
            test_data_imit = np.arange(0, len_gap)
            test_data = InputData(idx=np.arange(0, len(test_data_imit)),
                                  features=None,
                                  target=test_data_imit,
                                  task=task_to_solve,
                                  data_type=DataTypesEnum.ts)

            # Цепочка для задачи заполнения пропусков
            ref_chain = get_composite_chain()

            available_model_types_primary = ['lasso', 'ridge']

            available_model_types_secondary = ['rfr', 'lasso','ridge']
            metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)

            composer = FixedStructureComposer()

            composer_requirements = GPComposerRequirements(
                primary=available_model_types_primary,
                secondary=available_model_types_secondary,
                max_arity=2,
                max_depth=4,
                pop_size=10,
                num_of_generations=2,
                crossover_prob=0,
                mutation_prob=0.8,
                max_lead_time=datetime.timedelta(minutes=15))

            chain = composer.compose_chain(data=train_data,
                                           initial_chain=ref_chain,
                                           composer_requirements=composer_requirements,
                                           metrics=metric_function,
                                           is_visualise=False)
            chain.fit(input_data=train_data, verbose=False)

            output_values = chain.predict(test_data)
            predicted = output_values.predict[max_window_size + forecast_length:]
            print(predicted)

            # Replace gaps in an array with predicted values
            output_data[gap] = predicted

        return (output_data)


# Example of applying the algorithm
from pylab import rcParams

rcParams['figure.figsize'] = 18, 7

if __name__ == '__main__':
    # Loading an array with gaps
    data = pd.read_csv('D:/School 2020 NSS team/Meteodata.csv', sep=';')
    data['Date'] = pd.to_datetime(data['Date'])

    # Cut time series from 1992 till 1996
    data = data[data['Date'] > datetime.datetime.strptime('01.01.1992', '%d.%m.%Y')]
    data = data[data['Date'] < datetime.datetime.strptime('01.01.1996', '%d.%m.%Y')]
    additional_tmp = data['Tmp_kln']
    spb_tmp = data['100_gap']

    # Filling in gaps
    Gapfiller = TSGapfiller(gap_value=-100.0)
    withoutgap_arr = Gapfiller.inverse_ridge(spb_tmp)

    # Сравнение с простым алгоритмом заполнения пропусков - локальной аппроксимацией полиномами
    SimpleGapfill = TSGapfiller(gap_value=-100.0)
    withoutgap_arr_poly = SimpleGapfill.local_poly_approximation(spb_tmp, 4, 150)

    plt.plot(data['Date'], withoutgap_arr, c='blue', alpha=0.5)
    plt.plot(data['Date'], withoutgap_arr_poly, c='orange', alpha=0.4)
    plt.plot(data['Date'], data['Tmp_spb'], c='blue', alpha=0.2)
    plt.grid()
    plt.show()
