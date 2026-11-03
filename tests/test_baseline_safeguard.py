import pytest
import torch
from torch import nn

from models.maplocnet import one_sided_regression_penalty
from module import build_staged_optimizer, freeze_baseline_batch_normalization


def test_one_sided_safeguard_only_penalizes_regressions():
    student = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
    teacher = torch.tensor([2.0, 2.0, 2.0], requires_grad=True)

    penalty = one_sided_regression_penalty(student, teacher, margin=0.25)

    assert torch.allclose(penalty, torch.tensor([0.0, 0.0, 1.75]))
    penalty.sum().backward()
    assert torch.equal(student.grad, torch.tensor([0.0, 0.0, 1.0]))
    assert teacher.grad is None


def test_one_sided_safeguard_rejects_negative_margin():
    with pytest.raises(ValueError, match="non-negative"):
        one_sided_regression_penalty(torch.ones(1), torch.ones(1), margin=-0.1)


class _TinyHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Linear(1, 1)
        self.map_encoder = nn.Module()
        self.map_encoder.cnn_encoder = nn.Linear(1, 1)
        self.map_encoder.graph_encoder = nn.Linear(1, 1)
        self.map_encoder.graph_proj = nn.Linear(1, 1)
        self.map_encoder.fusion_gate = nn.Parameter(torch.zeros(1))


class _TinyHybridWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Sequential(nn.BatchNorm2d(1))
        self.map_encoder = nn.Module()
        self.map_encoder.cnn_encoder = nn.Sequential(nn.BatchNorm2d(1))
        self.map_encoder.graph_encoder = nn.Sequential(nn.BatchNorm2d(1))


def test_only_baseline_batch_norm_is_forced_to_eval():
    model = _TinyHybridWithBatchNorm().train()

    freeze_baseline_batch_normalization(model)

    assert model.image_encoder[0].training is False
    assert model.map_encoder.cnn_encoder[0].training is False
    assert model.map_encoder.graph_encoder[0].training is True


def _group_lr(optimizer, name):
    return next(group["lr"] for group in optimizer.param_groups if group["name"] == name)


def test_staged_optimizer_warms_graph_before_baseline():
    model = _TinyHybrid()
    optimizer, scheduler = build_staged_optimizer(
        model.named_parameters(),
        lr=1e-4,
        baseline_lr_scale=0.1,
        baseline_warmup_steps=2,
    )

    assert _group_lr(optimizer, "baseline") == 0.0
    assert _group_lr(optimizer, "graph") == pytest.approx(1e-4)

    for _ in range(2):
        optimizer.step()
        scheduler.step()

    assert _group_lr(optimizer, "baseline") == pytest.approx(1e-5)
    assert _group_lr(optimizer, "graph") == pytest.approx(1e-4)

    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(grouped) == len(list(model.parameters()))
    assert len({id(parameter) for parameter in grouped}) == len(grouped)
