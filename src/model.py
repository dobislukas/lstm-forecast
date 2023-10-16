# Imports
import torch
import torch.nn as nn


class LSTMModel(torch.nn.Module):
    def __init__(
            self,
            input_size: int, hidden_size: int,
            num_layers: int, output_size: int) -> None:
        """
        Creates LSTM model layers and returns initialized class instance.

        :param input_size: input dimensionality
        :param hidden_size: size of LSTM layer
        :param num_layers: number of LSTM layers
        :param output_size: output dimensionality
        :return: initialized class instance
        """

        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes inference over batch of input sequences.

        :param x: batch of input sequences
        :return: batch of predicted values
        """
        h0 = torch.zeros(
            self.num_layers, x.size(0),
            self.hidden_size).to(x.device)
        c0 = torch.zeros(
            self.num_layers, x.size(0),
            self.hidden_size).to(x.device)

        x, _ = self.lstm(x, (h0, c0))
        x = self.fc1(x[:, -1, :])

        return x
