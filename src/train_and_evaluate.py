# Imports
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from dataset import TimeSeriesDataset


def get_dataloaders(
        train_data: np.ndarray,
        test_data: np.ndarray,
        sequence_length: int,
        batch_size: int,
        device: str) -> tuple[DataLoader, DataLoader, MinMaxScaler]:
    """
    Scales and prepares data for pytorch training loop.

    :param train_data: numpy array of train part of data
    :param test_data: numpy array of test part of data
    :param sequence_length: size of input sequence
    :param batch_size: size of inference batch
    :param device: computation device
    :return: tuple containing dataloaders and scaler object for preprocessing
    """
    # Preprocess data
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)

    # Prepare data for training and evaluation
    train_dataset = TimeSeriesDataset(
        train_data_scaled,
        sequence_length, device)
    train_loader = DataLoader(
        train_dataset,
        batch_size, shuffle=True)

    test_dataset = TimeSeriesDataset(
        test_data_scaled,
        sequence_length, device)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler


def compute_epoch(
        model: torch.nn.Module, loader: DataLoader,
        loss_function: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        training: bool = True) -> float:
    """
    Computes epoch with model over data inside torch dataloader.

    :param model: pytorch model
    :param loader: pytorch dataloader
    :param loss_function: pytorch module for computing loss
    :param optimizer: pytorch optimizer for weight updates
    :param batch_size: size of inference batch
    :param training: boolean for if it is training epoch
    :return: average batch loss value
    """
    if training:
        model.train()
    else:
        model.eval()

    loss_list = []

    for x, y in loader:
        pred_y = model(x)
        loss = loss_function(y, pred_y)
        loss_list.append(loss.item())

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return np.mean(loss_list)


def predict_data_from_loader(model, loader):
    """
    Computes inferences with model over data inside torch dataloader.

    :param model: pytorch model
    :param loader: pytorch dataloader

    :return: numpy array of predictions
    """
    model.eval()
    predictions = None
    for batch, _ in loader:
        predicted_batch = model(batch).cpu().detach().numpy()
        predictions = predicted_batch if predictions is None \
            else np.vstack((predictions, predicted_batch))

    return predictions


# TODO Add configurable loss_function_type
# TODO Add configurable optimizer_type
def train_model(
        model: torch.nn.Module,
        train_loader: DataLoader, test_loader: DataLoader,
        epochs: int, learning_rate: float) -> dict[float]:
    """
    Trains passed model on training data.
    Returns dictionary with loss from epochs.

    :param model: pytorch model
    :param train_loader: pytorch dataloader for training data
    :param test_loader: pytorch dataloader for test data
    :param epochs: number of training epochs
    :param learning_rate: rate for optimizer
    :return: dictionary with lists of losses
    """
    mse_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_dict = {"train": [], "test": []}

    for epoch in range(epochs):

        train_epoch_loss = compute_epoch(model=model,
                                         loader=train_loader,
                                         loss_function=mse_function,
                                         optimizer=optimizer,
                                         training=True)
        loss_dict["train"].append(train_epoch_loss)

        with torch.no_grad():
            test_epoch_loss = compute_epoch(model=model,
                                            loader=test_loader,
                                            loss_function=mse_function,
                                            training=False)
            loss_dict["test"].append(test_epoch_loss)

        # Visualize training progress
        with tqdm(total=1, unit='epoch') as t:

            t.set_description(f'Epoch {epoch+1}/{epochs}')
            t.set_postfix(train_loss=train_epoch_loss,
                          test_loss=test_epoch_loss)
            t.update()

    return loss_dict
