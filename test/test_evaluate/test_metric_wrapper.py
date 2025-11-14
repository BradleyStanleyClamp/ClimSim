import evaluate
import torch 
from omegaconf import DictConfig

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def sample(self, num_samples):
         self.data = self.data[torch.randperm(len(self.data))[:num_samples]]
    

def test_metric_wrapper():
    torch.random.manual_seed(0)
    X = torch.rand((1000, 2))
    Y = torch.rand((1500, 2))
    X_dataset = DummyDataset(X)
    Y_dataset = DummyDataset(Y)    

    data_loader_cfg = DictConfig({
        'num_workers': 1,
        'persistent_workers': False,
        'prefetch_factor': 2
    })
    metric_wrapper = evaluate.MetricWrapper(
        samples_size=False,
        n_samples=100,
        batch_size=False,
        metric_name="energy_distance",
        dataloader_cfg=data_loader_cfg
    )

    dist = metric_wrapper.calculate(X_dataset, Y_dataset)
    expected_dist = evaluate.energy_distance(X, Y)

    print(f"MetricWrapper distance: {dist}")
    print(f"Expected distance: {expected_dist}")

    assert abs(dist - expected_dist) < 1e-7