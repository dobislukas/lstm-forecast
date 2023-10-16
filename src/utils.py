# Imports
import os
import random
import yaml
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

import plotly.express as px
import plotly.graph_objects as go


def read_experiment_config(config_filepath):
    """
    Reads experiment configuration from yaml file.

    :param config_filepath: filepath to yaml file
    :return: dictionary with configuration parameters
    """
    with open(config_filepath, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return config


def set_seed(seed: int = 42) -> None:
    """
    Sets seed for each library for reproducibility of results.

    :param seed: number to seed the machine
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_mape_metric(
        prediction: np.ndarray,
        original: np.ndarray) -> float:
    """Computes Mean Absolute Percentage Error between two numpy arrays."""
    return np.mean(np.abs(prediction - original) / original) * 100


def calculate_mae_metric(
        prediction: np.ndarray,
        original: np.ndarray) -> float:
    """Computes Mean Absolute Error between two numpy arrays."""
    return np.mean(np.abs(prediction - original))


def calculate_rmse_metric(
        prediction: np.ndarray,
        original: np.ndarray) -> float:
    """Computes Root Mean Square Error between two numpy arrays."""
    return np.sqrt(np.mean(np.power(prediction - original, 2)))


def plot_forecast(original, forecast, timestamps, quantity_label) -> go.Figure:
    """
    Plots ground truth and predicted values in time by using Plotly.

    :param original: ground truth values
    :param forecast: predicted values
    :param timestamps: date time timestamps
    :param quantity_label: forecasted quantity
    :return: Plotly figure
    """
    df = pd.DataFrame(
        list(zip(original, forecast, timestamps)),
        columns=["Original", "Forecast", "Time"]
    )
    figure = px.line(df, x="Time", y=df.columns, title="Forecast on test data")
    figure.update_layout(
        yaxis_title=f"{quantity_label} [W]",
        xaxis_title=None,
    )

    figure.show()

    return figure


def plot_loss(loss_dict: dict[list[float]]) -> go.Figure:
    """
    Plots progression of training and test loss
    during training epochs by using Plotly.

    :param loss_dict: dictionary with loss values
    :return: Plotly figure
    """
    figure = go.Figure()
    for key, values in loss_dict.items():

        epoch_axis = np.arange(len(values)) + 1
        loss_name = f"{key} loss"

        figure.add_scatter(x=epoch_axis, y=values, name=loss_name)

    figure.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Loss",
    )

    figure.show()

    return figure


def save_experiment(
        save_directory,
        model: torch.nn.Module, scaler: MinMaxScaler,
        metrics: dict[float], quantity_type) -> None:
    """
    Saves parameters of model and scaler.
    Additionaly saves computed metrics and loss/forecast plots.

    :param save_directory: save directory filepath
    :param model: forecast pytorch model
    :param scaler: scikit-learn scaler object
    :param metrics: dictionary with computed metrics
    """

    # Save forecast metrics
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_df.to_csv(f"{save_directory}/metrics.csv", header=False)

    # Save scaling parameters
    scaling_params = scaler.get_params()
    with open(f"{save_directory}/scaling_parameters.pkl", "wb") as f:
        pickle.dump(scaling_params, f)

    # Save model weights
    torch.save(model, f"{save_directory}/model.pt")
