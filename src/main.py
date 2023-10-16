# Imports
import os
import argparse
import pandas as pd
import numpy as np
import torch

from model import LSTMModel
from utils import read_experiment_config, set_seed, calculate_mape_metric, \
                  calculate_mae_metric, calculate_rmse_metric, \
                  plot_loss, plot_forecast, save_experiment
from train_and_evaluate import get_dataloaders, train_model, \
                               predict_data_from_loader


def main() -> None:

    # Create argument parser
    parser = argparse.ArgumentParser(description="Process CSV data")

    # Define the command line arguments
    parser.add_argument(
        "--input",
        default="../data/SG.csv",
        help="Input CSV file (default: ../data/SG.csv)",
    )
    parser.add_argument(
        "--quantity",
        default="Consumption",
        help="Quantity type (default: Consumption)",
    )
    parser.add_argument(
        "--save_results",
        default=True,
        help="Save results (default: True)",
    )

    # Parse the command line arguments
    args = parser.parse_args()
    input_filepath = args.input
    quantity_type = args.quantity
    save_results = args.save_results

    # Read experiment configuration
    config_filepath = "../configuration/config.yaml"
    config_dict = read_experiment_config(config_filepath)
    device = config_dict["device"] if torch.cuda.is_available() else "cpu"

    # Set seed
    set_seed(config_dict["seed"])

    # Read data
    data_df = pd.read_csv(input_filepath, delimiter=";")

    # Detect NaN and fill them
    data_has_nan_values = data_df.isnull().sum().sum() > 0
    if data_has_nan_values:
        data_df = data_df.interpolate(method="spline", order=3)

    # Pick data of selected quantity
    data_array = np.array(data_df[quantity_type]).reshape(-1, 1)

    # Divide data into train and test parts
    division_index = round((1 - config_dict["test_ratio"]) * len(data_array))
    train_data = data_array[:division_index]
    test_data = data_array[division_index:]

    # Prepare data
    train_loader, test_loader, scaler = get_dataloaders(
        train_data, test_data,
        config_dict["sequence_length"],
        config_dict["batch_size"],
        device)

    # Initialize model
    model = LSTMModel(
        config_dict["input_dim"],
        config_dict["hidden_dim"],
        config_dict["number_of_lstm_layers"],
        config_dict["output_dim"])
    model.to(device)

    # Train model
    loss_dict = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=config_dict["epoch_count"],
        learning_rate=config_dict["learning_rate"])

    # Forecast values for test dataset
    scaled_test_predictions = predict_data_from_loader(model, test_loader)
    test_predictions = scaler.inverse_transform(scaled_test_predictions)

    # Skip unforecasted values and reshape from (size, 1) to (size)
    test_data = test_data[config_dict["sequence_length"]:].squeeze()
    test_predictions = test_predictions.squeeze()

    # Calculate metrics for forecast
    mape = calculate_mape_metric(test_predictions, test_data)
    mae = calculate_mae_metric(test_predictions, test_data)
    rmse = calculate_rmse_metric(test_predictions, test_data)
    metrics_dict = {"MAPE": mape, "MAE": mae, "RMSE": rmse}

    print("Test dataset evaluation")
    print(f"Mean Absolute Percentage Error: {mape} %")
    print(f"Mean Absolute Error: {mae} W")
    print(f"Root Mean Square Error: {rmse} W")

    # Extract test data timestamps for forecast plot
    test_timestamps = data_df["Time"].loc[division_index:]
    test_timestamps = pd.to_datetime(
        test_timestamps, format='%Y-%m-%dT%H:%M:%S')

    loss_figure = plot_loss(loss_dict)
    forecast_figure = plot_forecast(
        original=test_data,
        forecast=test_predictions,
        timestamps=test_timestamps,
        quantity_label=quantity_type)

    # Save experiment
    if save_results:

        save_directory = config_dict["results_path"] + quantity_type + "/"
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)

        # Save plots
        loss_figure.write_image(f"{save_directory}/loss.pdf")
        forecast_figure.write_image(f"{save_directory}/forecast.pdf")

        save_experiment(
            save_directory=save_directory,
            model=model, scaler=scaler,
            metrics=metrics_dict, quantity_type=quantity_type)

        print(f"Results were successfully saved in directory {save_directory}")


if __name__ == "__main__":
    main()
