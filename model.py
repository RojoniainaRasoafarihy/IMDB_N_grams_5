import torch
import torch.nn as nn

class LogisticRegressionModel(nn.Module):

    def __init__(self, input_dim):

        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # One output for binary classification

    def forward(self, x):

        x = self.linear(x)
        return torch.sigmoid(x)
