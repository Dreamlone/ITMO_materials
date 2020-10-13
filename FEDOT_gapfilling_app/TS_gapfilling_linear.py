import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from core.composer.node import PrimaryNode
from core.composer.ts_chain import TsForecastingChain
from core.models.data import InputData
from core.repository.dataset_types import DataTypesEnum
from core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from pylab import rcParams
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

rcParams['figure.figsize'] = 18, 7

print('Nice')
print('ts-gapfilling')
print('1')

def validate(parameter: str, mask: str, data: pd.DataFrame, withoutgap_arr: np.array,
             gap_value: float = -100.0) -> None:
    """Checking the accuracy of restoring the source timeseries

    Parameters
    ----------
    parameter : str
        The name of the column in the dataframe, the parameter from which the time series is composed
    mask : str
        The name of the column in the data frame, which contains the binary code of the gap mask
    data : pandas DataFrame
        Dataframe containing all the necessary information
    withoutgap_arr : numpy array
        Array without gaps
    gap_value : float
        Gap flag in array

    Returns
    -------
    None
        The function displays the values of three metrics: MAE, RMSE, MedianAE + plots

    """

    arr_parameter = np.array(data[parameter])
    arr_mask = np.array(data[mask])

    # Which elements have gaps
    ids = np.ravel(np.argwhere(arr_mask == 0))

    true_values = arr_parameter[ids]
    predicted_values = withoutgap_arr[ids]
    print('The total size of the gaps:', len(true_values))
    min_value = min(true_values)
    max_value = max(true_values)
    print('Minimum value in the gap - ', min_value)
    print('Maximum value in the gap - ', max_value)

    # Displaying metrics
    mae_metric = mean_absolute_error(true_values, predicted_values)
    print('Mean absolute error -', round(mae_metric, 2))

    rmse_metric = (mean_squared_error(true_values, predicted_values)) ** 0.5
    print('RMSE -', round(rmse_metric, 2))

    medianae_metric = median_absolute_error(true_values, predicted_values)
    print('Median absolute error -', round(medianae_metric, 2), '\n')

    # Matching predicted values with gaps
    arr_parameter_modified = np.copy(arr_parameter)
    arr_parameter[arr_mask == 1] = gap_value  # Initial values in gaps
    withoutgap_arr[arr_mask == 1] = gap_value  # Filled (predicted) values in gaps

    arr_parameter_modified[arr_mask == 0] = gap_value
    masked_array_1 = np.ma.masked_where(arr_parameter_modified == gap_value, arr_parameter_modified)
    masked_array_2 = np.ma.masked_where(arr_parameter == gap_value, arr_parameter)
    masked_array_3 = np.ma.masked_where(withoutgap_arr == gap_value, withoutgap_arr)

    if parameter == 'Mean Tmp':
        name = 'Среднесуточная температура воздуха, ℃'
    else:
        name = 'Количество осадков, мм'

    plt.plot(data['Date'], masked_array_1, c='blue', alpha=0.5)
    plt.plot(data['Date'], masked_array_2, c='green', alpha=0.5, label='Действительные значения')
    plt.plot(data['Date'], masked_array_3, c='red', alpha=0.5, label='Предсказанные значения')
    plt.ylabel(name, fontsize=15)
    plt.xlabel('Дата', fontsize=15)
    if mask != 'Small_mask':
        plt.xlim(data['Date'][ids[0] - 100], data['Date'][ids[-1] + 100])
    plt.grid()
    plt.legend(fontsize=15)
    plt.show()


def fill_gaps(data: np.array, max_window_size: int = 100, gap_value: float = -100.0) -> np.array:
    """Wrapper function for filling gaps in time series using FEDOT framework

    Parameters
    ----------
    data : numpy array
        One-dimensional array (time series) in which we want to fill in the gaps
    max_window_size : int
        Sliding window size
    gap_value : float
        Gap flag in array

    Returns
    -------
    numpy array
        Time series without gaps

    """

    # Gap indices
    gap_list = np.ravel(np.argwhere(data == gap_value))

    # I think this fragment should be updated
    # Finding gaps by intervals
    new_gap_list = []
    local_gaps = []
    for index, gap in enumerate(gap_list):
        if index == 0:
            local_gaps.append(gap_list[index])
        else:
            prev_gap = gap_list[index - 1]
            if gap - prev_gap > 1:
                # There is a gap between gaps
                local_gaps.append(gap)
                new_gap_list.append(local_gaps)
                local_gaps = []
            else:
                local_gaps.append(gap)

    if len(new_gap_list) == 0:
        new_gap_list.append(local_gaps)

    # Iterately fill in the gaps in the time series
    for gap in new_gap_list:
        # The entire time series is used for training until the gap
        timeseries_train_part = data[:gap[0]]

        # Adaptive prediction interval length
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

        # Making predictions for the missing part in the time series
        chain = TsForecastingChain(PrimaryNode('ridge'))
        chain.fit_from_scratch(input_data)

        # "Test data" for making prediction for a specific length
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

        # Replace gaps in an array with predicted values
        data[gap] = predicted_values

    return(data)


#                               ----- APPLICATION OF THE ALGORITHM -----                                     #
if __name__ == '__main__':

    print('Test branch')
    # For what values will the model run
    mask_name = 'Big_mask_100'
    parameter = 'Mean Tmp'

    # Loading a dataset with data
    data = pd.read_csv('D:/FEDOT_timeseries/Spb_meteodata.csv', sep=';')
    data['Date'] = pd.to_datetime(data['Date'])
    print(data.head(5), '\n')

    # Generating a gap in a selected column
    arr_parameter = np.array(data[parameter])
    mask = np.array(data[mask_name])
    arr_parameter[mask == 0] = -100.0

    # Start filling in the gaps
    without_gaps = fill_gaps(arr_parameter, max_window_size=50, gap_value=-100.0)

    # Vericfication
    validate(parameter=parameter, mask=mask_name, data=data, withoutgap_arr=without_gaps)
