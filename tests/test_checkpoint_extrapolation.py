import torch

from evaluation.extrapolate_checkpoint import extrapolate_state_dict


def test_negative_one_extrapolation_reflects_finetune_update():
    baseline = {
        "weight": torch.tensor([2.0]),
        "counter": torch.tensor(3, dtype=torch.long),
    }
    finetuned = {
        "weight": torch.tensor([1.5]),
        "counter": torch.tensor(4, dtype=torch.long),
        "new_gate": torch.tensor([20.0]),
    }

    result = extrapolate_state_dict(baseline, finetuned, alpha=-1.0)

    assert torch.equal(result["weight"], torch.tensor([2.5]))
    assert torch.equal(result["counter"], baseline["counter"])
    assert torch.equal(result["new_gate"], finetuned["new_gate"])
