import evaluate
import torch

def test_no_overlapp():
    x = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5]])
    y = torch.tensor([[0.6], [0.7], [0.8], [0.9], [1.0]])

    out = evaluate.marginal_composition_evaluation(x, y)

    assert out["overlap_percentage"] == 0.0
    assert out["coverage"] == False
    assert torch.allclose(torch.tensor(out["train_min"]), torch.tensor(0.1))
    assert torch.allclose(torch.tensor(out["train_max"]), torch.tensor(0.5))
    assert torch.allclose(torch.tensor(out["test_min"]), torch.tensor(0.6))
    assert torch.allclose(torch.tensor(out["test_max"]), torch.tensor(1.0))

def test_full_overlapp():
    x = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5]])
    y = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5]])

    out = evaluate.marginal_composition_evaluation(x, y)

    assert out["overlap_percentage"] == 100.0
    assert out["coverage"] == True
    assert torch.allclose(torch.tensor(out["train_min"]), torch.tensor(0.1))
    assert torch.allclose(torch.tensor(out["train_max"]), torch.tensor(0.5))
    assert torch.allclose(torch.tensor(out["test_min"]), torch.tensor(0.1))
    assert torch.allclose(torch.tensor(out["test_max"]), torch.tensor(0.5))

def test_partial_overlapp():
    x = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5]])
    y = torch.tensor([[0.4], [0.5], [0.6], [0.7], [0.8]])

    out = evaluate.marginal_composition_evaluation(x, y)

    assert out["overlap_percentage"] == 40.0
    assert out["coverage"] == False
    assert torch.allclose(torch.tensor(out["train_min"]), torch.tensor(0.1))
    assert torch.allclose(torch.tensor(out["train_max"]), torch.tensor(0.5))
    assert torch.allclose(torch.tensor(out["test_min"]), torch.tensor(0.4))
    assert torch.allclose(torch.tensor(out["test_max"]), torch.tensor(0.8))

def test_internal_partial_overlapp():
    x = torch.tensor([[0.1], [0.2], [0.3], [0.6], [0.9]])
    y = torch.tensor([[0.4], [0.2], [0.3], [0.6]])

    out = evaluate.marginal_composition_evaluation(x, y)

    assert out["overlap_percentage"] == 75.0
    assert out["coverage"] == True
    assert torch.allclose(torch.tensor(out["train_min"]), torch.tensor(0.1))
    assert torch.allclose(torch.tensor(out["train_max"]), torch.tensor(0.9))
    assert torch.allclose(torch.tensor(out["test_min"]), torch.tensor(0.2))
    assert torch.allclose(torch.tensor(out["test_max"]), torch.tensor(0.6))