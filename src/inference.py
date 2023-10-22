# Imports
import argparse
from textwrap import dedent
import pandas as pd
import numpy as np

import onnxruntime

from utils import set_seed, plot_forecast, \
                  load_minmax_scaler, make_onnx_inference, \
                  calculate_percentage_error


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
        "--model_dir",
        default="../experiments/Consumption/",
        help="Path to trained experiment dir \
        (default: ../experiments/Consumption/)",
    )

    # Parse the command line arguments
    args = parser.parse_args()
    input_filepath = args.input
    quantity_type = args.quantity
    model_dir = args.model_dir

    # Set (default) seed
    set_seed()

    # Read data
    data_df = pd.read_csv(input_filepath, delimiter=";")

    # Detect NaN and fill them
    data_has_nan_values = data_df.isnull().sum().sum() > 0
    if data_has_nan_values:
        data_df = data_df.interpolate(method="spline", order=3)

    # Pick data of selected quantity
    data_array = np.array(
        data_df[quantity_type]).reshape(-1, 1).astype(np.single)

    # Preprocess inference data
    scaler = load_minmax_scaler(model_dir)
    scaled_data_array = scaler.transform(data_array)

    # Initialize ONNX model
    ort_session = onnxruntime.InferenceSession(
        f"{model_dir}/model.onnx", providers=['CPUExecutionProvider'])
    sequence_length = ort_session.get_inputs()[0].shape[1]

    # Predict
    predictions = make_onnx_inference(
        onnx_model=ort_session, inference_data=scaled_data_array,
        sequence_length=sequence_length, autoregression_size=0)

    # Rescale to original scale
    data_array = data_array[sequence_length:].squeeze()
    inversed_predictions = scaler.inverse_transform(predictions)
    inversed_predictions = inversed_predictions.squeeze()

    # Compute percentage error for prediction vs original
    pe_values = calculate_percentage_error(inversed_predictions, data_array)
    mape_metric = round(np.mean(pe_values), 1)
    print(dedent("""\
        Infered predictions had 
        Mean Absolute Percentage Error: {mape_metric}% """))

    # Create figure
    timestamps = pd.to_datetime(
        data_df["Time"], format='%Y-%m-%dT%H:%M:%S')
    forecast_figure = plot_forecast(
        original=data_array,
        forecast=inversed_predictions,
        timestamps=timestamps,
        quantity_label=quantity_type,
        no_show=True)

    # Save results
    df = pd.DataFrame(
        list(zip(timestamps, data_array, inversed_predictions, pe_values)),
        columns=["Time", "Original", "Forecast", "Percentage Error"]
    )
    df.to_csv(f"{model_dir}/inference_results.csv")
    forecast_figure.write_image(f"{model_dir}/inference_results.pdf")


if __name__ == "__main__":
    main()
