import torch


class PredictMean(torch.nn.Module):
    """
    Basic baseline model that predicts the mean value of the input tensor, structured as a PyTorch module for integration into pipelines.
    """

    def __init__(self, mean=None):
        super(PredictMean, self).__init__()
        self.mean = mean

    def forward(self, x):
        if self.mean is not None:
            return self.mean
        return torch.mean(x, dim=-1, keepdim=True)
