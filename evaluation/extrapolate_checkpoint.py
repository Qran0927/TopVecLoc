"""Create a checkpoint by extrapolating away from a fine-tuning update."""

import argparse
from pathlib import Path

import torch


def extrapolate_state_dict(baseline, finetuned, alpha=-1.0):
    """Return ``baseline + alpha * (finetuned - baseline)`` for shared floats."""
    result = {}
    for key, value in finetuned.items():
        if key not in baseline:
            result[key] = value
            continue
        base_value = baseline[key]
        if value.shape != base_value.shape:
            raise ValueError(
                f"state shape mismatch for {key}: "
                f"{tuple(base_value.shape)} vs {tuple(value.shape)}"
            )
        if torch.is_floating_point(value):
            base_value = base_value.to(value)
            result[key] = base_value + float(alpha) * (value - base_value)
        else:
            result[key] = base_value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_checkpoint")
    parser.add_argument("finetuned_checkpoint")
    parser.add_argument("output_checkpoint")
    parser.add_argument("--alpha", type=float, default=-1.0)
    args = parser.parse_args()

    baseline = torch.load(
        args.baseline_checkpoint, map_location="cpu", weights_only=False
    )
    finetuned = torch.load(
        args.finetuned_checkpoint, map_location="cpu", weights_only=False
    )
    output = dict(finetuned)
    output["state_dict"] = extrapolate_state_dict(
        baseline["state_dict"], finetuned["state_dict"], alpha=args.alpha
    )
    output_path = Path(args.output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"Saved extrapolated checkpoint to {output_path}")


if __name__ == "__main__":
    main()
