import torch

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.input = data

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx]

    def sample(self, num_samples):
        self.input = self.input[torch.randperm(len(self.input))[:num_samples]]
