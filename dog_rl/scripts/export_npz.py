"""Convert an exported rsl_rl policy.pt (TorchScript) into the .npz used by dog_gait/rl_policy_node.

Checks that the numpy MLP reproduces the TorchScript output before writing.

    cd ~/IsaacLab
    .venv/bin/python ~/robotdog_ws/dog_rl/scripts/export_npz.py \
        ~/IsaacLab/logs/rsl_rl/dog_flat/<run>/exported/policy.pt \
        ~/robotdog_ws/dog_gait/policies/dog_flat_<name>.npz
"""
import sys

import numpy as np
import torch


def numpy_mlp(w, x):
    for i in (0, 2, 4):
        x = x @ w[f"l{i}_weight"].T + w[f"l{i}_bias"]
        x = np.where(x > 0, x, np.expm1(x))  # ELU(alpha=1)
    return x @ w["l6_weight"].T + w["l6_bias"]


def main(src, dst):
    model = torch.jit.load(src, map_location="cpu").eval()
    weights = {k.replace("mlp.", "l").replace(".", "_"): v.numpy() for k, v in model.state_dict().items()}
    expected = {"l0_weight", "l0_bias", "l2_weight", "l2_bias", "l4_weight", "l4_bias", "l6_weight", "l6_bias"}
    if set(weights) != expected:
        raise SystemExit(f"unexpected policy layout {sorted(weights)} (obs normalization on? different MLP?)")
    x = np.random.randn(64, weights["l0_weight"].shape[1]).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    err = float(np.abs(numpy_mlp(weights, x) - ref).max())
    if err > 1e-4:
        raise SystemExit(f"numpy MLP does not match TorchScript (max err {err})")
    np.savez(dst, **weights)
    print(f"wrote {dst}  (obs {weights['l0_weight'].shape[1]}, actions {weights['l6_weight'].shape[0]}, max err {err:.1e})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
