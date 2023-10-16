import unittest

import numpy as np
import torch

from model import LSTMModel


class TestLSTMModelInference(unittest.TestCase):
    def test_lstm_inference(self) -> None:
        """Test model inference on input sequence."""

        # Sample input data
        input_data = np.array([1.0, 2.0, 3.0]).reshape(-1, 1, 1)
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        # Initialize the model
        input_dim = 1
        hidden_dim = 64
        num_layers = 2
        output_dim = 1
        model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)

        # Inference
        output = model(input_tensor)
        self.assertIsNotNone(output)


if __name__ == '__main__':
    unittest.main()
