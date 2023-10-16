# Imports
import numpy as np
import torch


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data: np.ndarray, sequence_length: int, device: str) -> None:
        """
        Creates dataset and returns initialized class instance.

        :param data: numpy array of timeseries
        :param sequence_length: length of sequence for forecast
        :param device: computation device
        :return: initialized class instance
        """
        self.data = data
        self.sequence_length = sequence_length
        self.device = device

    def __len__(self) -> int:
        """Returns number of available samples in this dataset."""
        return len(self.data) - self.sequence_length

    def __getitem__(
            self,
            sequence_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns tuple of (vector of values as a sequence,
           next value as a goal of prediction)."""
        sequence_start_index = sequence_index
        sequence_end_index = sequence_start_index + self.sequence_length

        x = self.data[sequence_start_index:sequence_end_index]
        y = self.data[sequence_end_index]

        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)

        return x_tensor, y_tensor
