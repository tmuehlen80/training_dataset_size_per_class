import math
from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.optimize import curve_fit
from sklearn import metrics
from scipy.stats.distributions import t
from typing import Tuple
from typing import Final, List

PARAMETER_INITIAL_GUESS: Final[List[float]] = [0.7, -0.05, -0.8]
PARAMETER_BOUNDS: Final[Tuple[np.ndarray, np.ndarray]] = ([0, -np.inf, -np.inf], [1, 0, 0])


def func_powerlaw(x: float, a: float, c: float) -> float:
    return a + x**(-1 / 5) * c


def func_powerlaw_inv(y: float, a: float, c: float) -> float:
    return ((y - a) / c)**(-5)


def func_powerlaw_exponent(x: float, a: float, b: float, c: float) -> float:
    return a + (x**b) * c


def func_powerlaw_exponent_inv(y: float, a: float, b: float, c: float) -> float:
    return ((y - a) / c)**(1 / b)


def predict_leave_one_out(x: np.ndarray,
                          y: np.ndarray,
                          f: callable,
                          p0=None,
                          bounds=(-np.inf, np.inf)) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Fit power law function by the leave one out principle. Perform prediction for the left out sample.

    Args:
        x: array containing the x datapoints used to fit a power law function
        y: array containing the y datapoints used to fit a power law function
        f: the model/fitting function, f(x, ...). See curve_fit for further details.
        p0: array_like, optional argument stating the nitial guess for the parameters (length N).
            See curve_fit for further details.
        bounds: 2-tuple of array_like, optional argument stating the lower and upper bounds on parameters. Defaults
            to no bounds. See curve_fit for further details.

    Returns:
        Tuple containing arrays of the ground_truths and the predictions which were calculated by the leave one out
        principle.
    '''
    test_ground_truth = []
    test_predictions = []
    size = len(x) - 1
    x = np.asarray(x)
    y = np.asarray(y)
    for i in range(size):
        train_x = np.delete(x, i)
        test_x = x[i]
        train_y = np.delete(y, i)
        test_y = y[i]
        params, _ = curve_fit(f, train_x, train_y, maxfev=20000, p0=p0, bounds=bounds)
        y_predicted = f(test_x, *params)
        test_ground_truth.append(test_y)
        test_predictions.append(y_predicted)
    return test_ground_truth, test_predictions


def calculate_confidence_bounds(data_point: float, ground_truth: np.ndarray, predictions: np.ndarray,
                                num_params: int, conf_level: float) -> Tuple[float, float]:
    '''
    Calculate lower and upper bound for a data_point based on residual standard deviation calculated from i.e. 
    ground_truth and predictions.

    Args:
        data_point: data point for which the lower and upper bound shall be calculated
        ground_truth: array containing ground truth
        predictions: array containing predictions
        num_params: number of parameters of the fitted curve. This is needed for calculating the dof degrees of freedom
        conf_level: confidence level for which the bounds shall be calculated

    Returns:
        Tuple containing the lower and upper bound for the data_point. 
    '''
    num_samples = len(ground_truth)
    assert num_samples > num_params, 'Number of samples must be greater than the number of parameters'
    sum_errs = np.sum((ground_truth - predictions)**2)
    residual_stdev = np.sqrt(1 / (num_samples - 2) * sum_errs)
    dof = max(0, (num_samples - num_params))
    tval = t.ppf(1.0 - conf_level / 2., dof)
    interval = residual_stdev * tval / (num_samples)**(1 / 2)
    lower, upper = data_point - interval, data_point + interval
    return lower, upper


def calculate_param_confidence_bounds(x_pred: np.ndarray, params: np.ndarray, params_cov: np.ndarray,
                                      n_datapoints: int, conf_level: float,
                                      prediction_type: str) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Calculate lower and upper bound for x_pred based on the covariance matric of the parameters.

    Args:
        x_pred: array containing the data points to predict
        params: parameters ralculated by fitting a powerlaw function
        params_cov: covariance matrix of the parameters
        n_datapoints: number of datappoints used to it the powerlaw function
        conf_level: confidence level for which the bounds shall be calculated
        prediction_type: predict either performance i.e. F1 score or dataset size. Depending on this we calculate 
            the bounds by using the powerlaw vs inverse powerlaw function 

    Returns:
        Tuple of arrays containing the lower and upper bound for x_pred. 
    '''
    n_params = len(params)
    dof = max(0, n_datapoints - n_params)  # number of degrees of freedom
    # student-t value for the dof and confidence level
    tval = t.ppf(1.0 - conf_level / 2., dof)
    for j, p, var in zip(range(n_datapoints), params, np.diag(params_cov)):
        sigma = var**0.5
        if prediction_type == 'performance':
            bounds_upper = func_powerlaw(x_pred, params[0] + sigma * tval, params[1] + sigma * tval)
            bounds_lower = func_powerlaw(x_pred, params[0] - sigma * tval, params[1] - sigma * tval)
        else:
            bounds_upper = func_powerlaw_inv(x_pred, params[0] + sigma * tval, params[1] + sigma * tval)
            bounds_lower = func_powerlaw_inv(x_pred, params[0] - sigma * tval, params[1] - sigma * tval)
    return bounds_lower, bounds_upper


def power_law_fit_and_plot(x: np.ndarray,
                           y: np.ndarray,
                           x_pred: np.ndarray,
                           projected_data_point: float,
                           title: str,
                           *,
                           split_point: int = None,
                           prediction_type: str = 'performance',
                           metric_name: str = 'F1 score',
                           print_conclusions: bool = True,
                           include_confidence: bool = False,
                           return_cov_matrix: bool = False,
                           conf_level: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Fit power law curve for x and y datapoints and plot results. In addition for demo purpose predict one datapoint 
    i.e. either performance or dataset_size.

    Args:
        x: array containing the x datapoints used to fit a power law function
        y: array containing the y datapoints used to fit a power law function
        x_pred: array containing the data points to predict
        projected_data_point: datapoint i.e. F1 score or dataset size to be predicted
        title: title to be used for plotting
        split_point: if not None split the x and y data into train and test/evaluate data and train only on the test
            part and predict on the test data.    
        prediction_type: predict either performance i.e. F1 score or dataset size. Depending on this we calculate 
            the bounds by using the powerlaw vs inverse powerlaw function 
        metric_name: name of the performance metric used e.g. F1 score, IoU etc. 
        include_confidence: whether or not to calculate and plot confidence bounds
        return_cov_matrix: if True return the full covariance matrix else the standard deviation of the parameters i.e.
            np.sqrt(np.diag(params_cov))
        conf_level: confidence level for which the bounds shall be calculated

    Returns:
        Array containing the fitted parameters of the power law curve and their variance.
   '''
    prediction_types = ['performance', 'dataset_size']
    if prediction_type not in prediction_types:
        raise ValueError(f"Invalid prediction type. Expected one of: {prediction_types}")

    train_x = x[:split_point]
    test_x = x[split_point:]
    train_y = y[:split_point]
    test_y = y[split_point:]
    params, params_cov = curve_fit(func_powerlaw_exponent,
                                   train_x,
                                   train_y,
                                   maxfev=20000,
                                   p0=PARAMETER_INITIAL_GUESS,
                                   bounds=PARAMETER_BOUNDS)
    y_test = func_powerlaw_exponent(test_x, params[0], params[1], params[2])

    # We want to predict either performance (e.g. F1 score) or dataset size thus we calculate the powerlaw vs inverse
    # powerlaw functions
    if prediction_type == 'performance':
        y_pred = func_powerlaw_exponent(x_pred, params[0], params[1], params[2])
    else:
        y_pred = func_powerlaw_exponent_inv(x_pred, params[0], params[1], params[2])
    print(f'Fitted parameters [a, b, c] of the power law function a + (x**b) * c are: {params}.')
    # Plot datapoints and fitted curves
    plt.plot(x_pred, y_pred, label='Fitted Curve', color='red')
    if prediction_type == 'performance':
        plt.scatter(x, y, label='Data Point')
        # only plot
        if split_point:
            plt.scatter(test_x, y_test, label='Predicted Data Point', color='orange')
        predicted_f1 = round(func_powerlaw_exponent(projected_data_point, params[0], params[1], params[2]), 5)
        plt.plot(projected_data_point,
                 predicted_f1,
                 marker='x',
                 markersize=25,
                 color='y',
                 label='Predicted Data Point')

        if include_confidence:
            bounds_lower = []
            bounds_upper = []
            test_ground_truth, test_predictions = predict_leave_one_out(train_x,
                                                                        train_y,
                                                                        func_powerlaw_exponent,
                                                                        p0=PARAMETER_INITIAL_GUESS,
                                                                        bounds=PARAMETER_BOUNDS)
            for j in range(len(y_pred)):
                lower, upper = calculate_confidence_bounds(y_pred[j],
                                                           np.asarray(test_predictions),
                                                           np.asarray(test_ground_truth),
                                                           num_params=len(params),
                                                           conf_level=conf_level)
                bounds_lower.append(lower)
                bounds_upper.append(upper)

            plt.fill_between(x_pred,
                             bounds_lower,
                             bounds_upper,
                             color='black',
                             alpha=0.15,
                             label=f'{round((1-conf_level)*100)}% upper and lower bound')
        images = int(projected_data_point - max(train_x))
        max_test_f1 = round(max(train_y), 5)
        plt.xlabel('Train Data Size (#images)')
        plt.ylabel(f'Test {metric_name}')
        if print_conclusions:
            print(
                f'For data size {round(projected_data_point)} the predicted test {metric_name} is {predicted_f1}.')
            print(
                f'Thus, adding {images} images to current dataset would change test {metric_name} of {max_test_f1} to {predicted_f1}.'
            )
    else:
        plt.scatter(y, x, label='Data Point')
        if split_point:
            plt.scatter(y_test, test_x, label='Predicted Data Point', color='orange')
        pred_dataset_size = int(func_powerlaw_exponent_inv(projected_data_point, params[0], params[1], params[2]))
        plt.plot(projected_data_point,
                 pred_dataset_size,
                 marker='x',
                 markersize=25,
                 color='y',
                 label='Predicted Train Dataset Size (#images)')
        images = int(pred_dataset_size - max(train_x))
        max_test_f1 = round(max(train_y), 5)
        plt.yscale('log')
        plt.ylabel('Train Data Size in #images (log scale)')
        plt.xlabel(f'Test {metric_name}')
        if print_conclusions:
            print(
                f'To achieve test {metric_name} {projected_data_point} we need train data size of {pred_dataset_size}.'
            )
            print(
                f'Thus, to improve {metric_name} from {max_test_f1} to {projected_data_point} we need {images} more images.'
            )
    plt.xticks(rotation=15)
    plt.legend()
    plt.title(title)

    if return_cov_matrix:
        return params, params_cov
    else:
        return params, np.sqrt(np.diag(params_cov))


def fit_power_law_for_all_dataframe_classes(fitting_points_df: pd.DataFrame,
                                            *,
                                            split_point: int = None,
                                            prediction_dataset_multiplicator: int = None,
                                            metric_column: str = 'class_f1score',
                                            metric_name: str = 'F1 score',
                                            dataset_size_unit: str = 'images',
                                            conf_level: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Fit power law curve for all classes in a dataframe df. The dataframe must include dataset_size, metric_column
    (e.g. F1 score, IoU) and class_name columns.
    Args:
        df: Dataframe containing data points (x/dataset_size,y/class_f1score) of different classes with class_name.
        split_point: if not None split the x and y data into train and test/evaluate data and train only on the test
            part and predict on the test data.
        prediction_dataset_multiplicator: if not None use the multiplicator to predicted performance would be if we 
            had prediction_dataset_multiplicator more i.e. 2x, 5x, 10x more train data then we do now.
        metric_column: name of the column in fitting_points_df that contains the performance metric e.g. F1 score, IoU
        metric_name: name of the performance metric used e.g. F1 score, IoU etc. 
        dataset_size_unit: string stating in which unit the train dataset size is e.g. #images, # #GT labels per class
        conf_level: confidence level for which the bounds shall be calculated.
    Returns:
       Tuple of arrays (eval_results, fitting_parameters) containing the evaluation results and the fitted parameters 
       of the power law curve and their variance.
   '''
    if not set(['dataset_size', metric_column, 'class_name']).issubset(fitting_points_df.columns):
        raise ValueError(f"Dataframe must contain 'dataset_size',{metric_column} , 'class_name' columns.")
    number_of_subplots = len(fitting_points_df.class_name.unique())
    number_of_columns = 3
    number_of_rows = int(math.ceil(number_of_subplots / number_of_columns))
    gs = gridspec.GridSpec(number_of_rows, number_of_columns)
    fig = plt.figure(figsize=(30, 30))
    eval_results = []
    fitting_parameters = []
    predicted_f1_scores = []

    for i, label_class_name in enumerate(fitting_points_df.class_name.unique()):
        class_df = fitting_points_df[fitting_points_df.class_name == label_class_name]
        class_df = class_df.sort_values(by='dataset_size')
        class_df = class_df.reset_index(drop=True)
        train = class_df.iloc[:split_point, :]
        test = class_df.iloc[split_point:, :]
        params, params_cov = curve_fit(func_powerlaw_exponent,
                                       train.dataset_size,
                                       train[metric_column],
                                       maxfev=20000,
                                       p0=PARAMETER_INITIAL_GUESS,
                                       bounds=PARAMETER_BOUNDS)
        p_std = np.sqrt(np.diag(params_cov))
        fitting_parameters.append({
            'class_name': label_class_name,
            f'current_train_max_dataset (#{dataset_size_unit})': max(train.dataset_size),
            'a': params[0],
            'b': params[1],
            'c': params[2],
            'a_std': p_std[0],
            'b_std': p_std[1],
            'c_std': p_std[2]
        })
        if not test.empty:
            y_eval = func_powerlaw_exponent(test.dataset_size, params[0], params[1], params[2])
            eval_results.append({
                'class_name': label_class_name,
                'mse': metrics.mean_squared_error(test[metric_column], y_eval),
                'rms': np.sqrt(metrics.mean_squared_error(test[metric_column], y_eval))
            })

        bounds_lower = []
        bounds_upper = []
        train = train.sample(frac=1).reset_index(drop=True)
        test_ground_truth, test_predictions = predict_leave_one_out(train.dataset_size,
                                                                    train[metric_column],
                                                                    func_powerlaw_exponent,
                                                                    p0=PARAMETER_INITIAL_GUESS,
                                                                    bounds=PARAMETER_BOUNDS)
        dataset_max = class_df.dataset_size.max()
        if prediction_dataset_multiplicator:
            dataset_target = prediction_dataset_multiplicator * dataset_max
            # add some additional buffer such that the extrapolation plot does not end exactly at the
            # dataset_target point
            extrapolation_target = dataset_target + 1e4
            x_pred = np.linspace(1.0, extrapolation_target, 1000)
        else:
            # add some additional buffer such that the extrapolation plot does not end exactly at the
            # current dataset_max point
            extrapolation_target = dataset_max
            x_pred = np.linspace(1, extrapolation_target, 1000)
        y_pred = func_powerlaw_exponent(x_pred, params[0], params[1], params[2])
        for j in range(len(y_pred)):
            lower, upper = calculate_confidence_bounds(y_pred[j],
                                                       np.asarray(test_ground_truth),
                                                       np.asarray(test_predictions),
                                                       num_params=len(params),
                                                       conf_level=conf_level)
            bounds_lower.append(lower)
            bounds_upper.append(upper)
        ax = fig.add_subplot(gs[i])
        ax.scatter(class_df.dataset_size, class_df[metric_column], label=f'Test {metric_name}')
        if not test.empty:
            ax.scatter(test.dataset_size, y_eval, label=f'Predicted Test {metric_name}', color='orange')
        ax.plot(x_pred, y_pred, color='red', label='Fitted Curve')
        plt.fill_between(x_pred,
                         bounds_lower,
                         bounds_upper,
                         color='black',
                         alpha=0.15,
                         label=f'{round((1-conf_level)*100)}% upper and lower bound')
        if prediction_dataset_multiplicator:
            temp = train.sort_values(by=['dataset_size'])
            current_max_metric = max(temp[metric_column])
            predicted_f1 = round(func_powerlaw_exponent(dataset_target, params[0], params[1], params[2]), 5)
            predicted_f1_scores.append({
                'class_name': label_class_name,
                f'current_max_dataset (#{dataset_size_unit})': dataset_max,
                f'current_{metric_name}': current_max_metric,
                f'dataset_target in #{dataset_size_unit} ({prediction_dataset_multiplicator}xcurrent_max_dataset)': dataset_target,
                f'predicted_{metric_name}': predicted_f1
            })
            ax.plot(dataset_target,
                    predicted_f1,
                    marker='x',
                    markersize=25,
                    color='y',
                    label=f'Predicted Test {metric_name} for {prediction_dataset_multiplicator}*max_dataset_size')
        ax.set(ylabel=f'Test {metric_name}')
        ax.set(xlabel=f'Train datset size (#{dataset_size_unit})')
        ax.set_title(label_class_name)
        ax.legend(loc='lower right')
        ax.set_ylim(min(class_df[metric_column]), max(bounds_upper) + 0.01)
    plt.suptitle('POWER LAW CURVES', size=16, y=1.0)
    plt.tight_layout()
    return eval_results, fitting_parameters, predicted_f1_scores
