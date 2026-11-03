"""Safe initialization of a hybrid model from a trained CNN baseline."""

from dataclasses import dataclass
from os import PathLike
from typing import Mapping, Union

import torch


@dataclass(frozen=True)
class HybridLoadReport:
    loaded_image_keys: int
    loaded_map_keys: int
    loaded_shared_keys: int
    skipped_source_keys: tuple[str, ...]
    incompatible_keys: tuple[str, ...]


@dataclass(frozen=True)
class TopVeLocLoadReport:
    loaded_keys: int
    missing_target_keys: tuple[str, ...]


CheckpointSource = Union[str, PathLike, Mapping[str, object]]


def _read_checkpoint(source: CheckpointSource) -> Mapping[str, object]:
    if isinstance(source, Mapping):
        return source
    try:
        return torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(source, map_location="cpu")


def _target_key(source_key: str) -> str:
    prefix = "model.map_encoder."
    if source_key.startswith(prefix):
        return f"{prefix}cnn_encoder.{source_key[len(prefix):]}"
    return source_key


def load_cnn_baseline_into_hybrid(module, source: CheckpointSource) -> HybridLoadReport:
    """Load every compatible shared CNN weight while preserving new GNN weights.

    Baseline map keys are nested below ``cnn_encoder`` in the hybrid model. Any
    shape mismatch for an otherwise matching shared key is treated as an error,
    and both the image and map subsystems must be represented in the checkpoint.
    """

    checkpoint = _read_checkpoint(source)
    source_state = checkpoint.get("state_dict")
    if not isinstance(source_state, Mapping):
        raise ValueError("baseline checkpoint does not contain a state_dict mapping")

    target_state = module.state_dict()
    merged_state = dict(target_state)
    skipped = []
    incompatible = []
    loaded_image = 0
    loaded_map = 0
    loaded_shared = 0

    for source_key, value in source_state.items():
        if not torch.is_tensor(value):
            skipped.append(source_key)
            continue
        target_key = _target_key(source_key)
        if target_key not in target_state:
            skipped.append(source_key)
            continue
        if target_state[target_key].shape != value.shape:
            incompatible.append(
                f"{source_key}: {tuple(value.shape)} -> "
                f"{target_key}: {tuple(target_state[target_key].shape)}"
            )
            continue

        merged_state[target_key] = value
        if source_key.startswith("model.image_encoder."):
            loaded_image += 1
        elif source_key.startswith("model.map_encoder."):
            loaded_map += 1
        else:
            loaded_shared += 1

    if incompatible:
        raise ValueError(
            "baseline checkpoint contains incompatible shared weights: "
            + "; ".join(incompatible)
        )
    if loaded_image == 0:
        raise ValueError("baseline checkpoint did not load any image encoder weights")
    if loaded_map == 0:
        raise ValueError("baseline checkpoint did not load any map encoder weights")

    module.load_state_dict(merged_state, strict=True)
    return HybridLoadReport(
        loaded_image_keys=loaded_image,
        loaded_map_keys=loaded_map,
        loaded_shared_keys=loaded_shared,
        skipped_source_keys=tuple(skipped),
        incompatible_keys=tuple(incompatible),
    )


def load_topveloc_into_hybrid(
    module,
    source: CheckpointSource,
    fusion_gate_value: float = 20.0,
) -> TopVeLocLoadReport:
    """Migrate a legacy ungated TopVeLoc checkpoint into the gated model."""
    checkpoint = _read_checkpoint(source)
    source_state = checkpoint.get("state_dict")
    if not isinstance(source_state, Mapping):
        raise ValueError("TopVeLoc checkpoint does not contain a state_dict mapping")

    target_state = module.state_dict()
    merged_state = dict(target_state)
    incompatible = []
    loaded = []
    for key, value in source_state.items():
        if key not in target_state or not torch.is_tensor(value):
            continue
        if target_state[key].shape != value.shape:
            incompatible.append(
                f"{key}: {tuple(value.shape)} -> {tuple(target_state[key].shape)}"
            )
            continue
        merged_state[key] = value
        loaded.append(key)

    if incompatible:
        raise ValueError(
            "TopVeLoc checkpoint contains incompatible weights: "
            + "; ".join(incompatible)
        )
    required_prefixes = (
        "model.image_encoder.",
        "model.map_encoder.cnn_encoder.",
        "model.map_encoder.graph_encoder.",
        "model.map_encoder.graph_proj.",
    )
    for prefix in required_prefixes:
        if not any(key.startswith(prefix) for key in loaded):
            raise ValueError(f"TopVeLoc checkpoint did not load subsystem {prefix}")

    gate_key = "model.map_encoder.fusion_gate"
    if gate_key not in target_state:
        raise ValueError("target hybrid model does not expose fusion_gate")
    merged_state[gate_key] = torch.full_like(
        target_state[gate_key], float(fusion_gate_value)
    )
    module.load_state_dict(merged_state, strict=True)
    missing = tuple(sorted(set(target_state).difference(loaded)))
    return TopVeLocLoadReport(
        loaded_keys=len(loaded),
        missing_target_keys=missing,
    )
