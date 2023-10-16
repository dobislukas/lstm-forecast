# Time Series Forecasting with LSTM

This project is a Python implementation of a time series forecasting model using Long Short-Term Memory (LSTM) neural network in PyTorch. It includes code for data preprocessing, model training, evaluation, and visualization. Model by default (as specified in `configuration/config.yaml`) takes sequence of 10 samples to predict next sample. Model does not use validation dataset, only tracks test loss during training.

## Installation

1. Clone the repository to your local machine:
```
git clone https://github.com/yourusername/timeseries-forecasting-lstm.git
```
2. Install the required packages:
```
pip install -r requirements.txt
```

## Usage

### Configuration

Parameters for training model are stored in YAML file in path `configuration/config.yaml`.

### Training

1. Select timeseries data csv file for forecasting, and specify it using the `--input` argument.. By default, the code assumes that the data is in the file `../data/SG.csv`.
2. Choose the quantity type (from timeseries csv columns) for forecasting, and specify it using the `--quantity` argument. By default, the code assumes `Consumption.csv`.
3. Option to save trained model and his results, by using the `--save_results` argument. By default, results are saved in `experiments/{quantity_type}/`.

Run the main script to train the LSTM model and make predictions on the test dataset. The following command will train the model:
```
python main.py --input ../data/SG.csv --quantity Consumption`
```

### Evaluation

The project calculates and prints three key metrics for forecast evaluation:

- Mean Absolute Percentage Error (MAPE) - metric for comparing performance across various forecasting models.
- Mean Absolute Error (MAE) - basic metric, in original units.
- Root Mean Square Error (RMSE) - more sensitive to outliers, not in original units.

### Results

The results can be found in the specified results directory, which is set in the `config.yaml` file. The results include:

- model.pt: A trained LSTM model weights.
- scaling_parameters.pkl: A pickle file containing parameters for preprocessing scaling MinMaxScaler instance from sciki-learn.
- loss.pdf: A plot showing the progression of training and test loss during training.
- forecast.pdf: A plot displaying the ground truth and predicted values over time for data not used in training.
- metrics.csv: A CSV file containing the computed metrics.

### Unit Testing
The project has single unit test, that deals with testing model's capability to forward propagate input sequence. Can be run by following command:

```
python tests.py
```

## TODO
- Add script to load trained model and then make predictions on new data of similar format.
- Remove overuse of config_dict[key] syntax in main.py and achieve higher experiment customization by using hydra configs.
- Add training specificication for optimizer and loss function types.
- Add proper validation training.
- Generate new folder for each new experiment with naming convention based on ID generated from hash seeded by current time.
- Add more tests.
- LARGE: Setup lightweight (no pytorch, trained weight in ONNX format) docker fastapi container with endpoints for inference.

## License

This project is licensed under the MIT License.