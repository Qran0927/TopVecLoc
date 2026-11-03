import torch
from torch import nn

from models.hybrid_checkpoint import (
    load_cnn_baseline_into_hybrid,
    load_topveloc_into_hybrid,
)


class _SourceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Linear(2, 2, bias=False)
        self.map_encoder = nn.Linear(2, 2, bias=False)


class _HybridMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_encoder = nn.Linear(2, 2, bias=False)
        self.graph_encoder = nn.Linear(2, 2, bias=False)
        self.graph_proj = nn.Linear(2, 2, bias=False)
        self.fusion_gate = nn.Parameter(torch.full((1,), -5.0))


class _TargetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Linear(2, 2, bias=False)
        self.map_encoder = _HybridMap()


class _Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


def _baseline_checkpoint():
    source = _Wrapper(_SourceModel())
    with torch.no_grad():
        source.model.image_encoder.weight.fill_(2.0)
        source.model.map_encoder.weight.fill_(3.0)
    return {"state_dict": source.state_dict()}


def test_loads_shared_image_and_remaps_cnn_map_weights():
    target = _Wrapper(_TargetModel())
    graph_before = target.model.map_encoder.graph_encoder.weight.detach().clone()

    report = load_cnn_baseline_into_hybrid(target, _baseline_checkpoint())

    assert torch.equal(
        target.model.image_encoder.weight,
        torch.full_like(target.model.image_encoder.weight, 2.0),
    )
    assert torch.equal(
        target.model.map_encoder.cnn_encoder.weight,
        torch.full_like(target.model.map_encoder.cnn_encoder.weight, 3.0),
    )
    assert torch.equal(target.model.map_encoder.graph_encoder.weight, graph_before)
    assert report.loaded_image_keys == 1
    assert report.loaded_map_keys == 1


def test_rejects_checkpoint_without_required_image_subsystem():
    target = _Wrapper(_TargetModel())
    checkpoint = _baseline_checkpoint()
    del checkpoint["state_dict"]["model.image_encoder.weight"]

    try:
        load_cnn_baseline_into_hybrid(target, checkpoint)
    except ValueError as error:
        assert "image encoder" in str(error)
    else:
        raise AssertionError("missing image encoder weights should be rejected")


def test_rejects_shape_incompatible_shared_weight():
    target = _Wrapper(_TargetModel())
    checkpoint = _baseline_checkpoint()
    checkpoint["state_dict"]["model.image_encoder.weight"] = torch.ones(3, 3)

    try:
        load_cnn_baseline_into_hybrid(target, checkpoint)
    except ValueError as error:
        assert "incompatible" in str(error)
        assert "model.image_encoder.weight" in str(error)
    else:
        raise AssertionError("shape-incompatible weights should be rejected")


def test_loads_legacy_topveloc_and_initializes_identity_gate():
    source = _Wrapper(_TargetModel())
    target = _Wrapper(_TargetModel())
    with torch.no_grad():
        source.model.image_encoder.weight.fill_(2.0)
        source.model.map_encoder.cnn_encoder.weight.fill_(3.0)
        source.model.map_encoder.graph_encoder.weight.fill_(4.0)
    state = source.state_dict()
    del state["model.map_encoder.fusion_gate"]

    report = load_topveloc_into_hybrid(
        target,
        {"state_dict": state},
        fusion_gate_value=20.0,
    )

    assert torch.equal(
        target.model.map_encoder.graph_encoder.weight,
        torch.full_like(target.model.map_encoder.graph_encoder.weight, 4.0),
    )
    assert torch.equal(
        target.model.map_encoder.fusion_gate,
        torch.full_like(target.model.map_encoder.fusion_gate, 20.0),
    )
    assert report.loaded_keys == len(state)
    assert report.missing_target_keys == ("model.map_encoder.fusion_gate",)
