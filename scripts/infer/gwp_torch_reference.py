#!/usr/bin/env python
"""Stage F reference: run the PyTorch GWP 0.5 pipeline on one fixed sample and
export everything the C++ side needs for a like-for-like comparison.

Outputs (all in --out):
  torch_action.npy        [48,14] physical joint targets from pipe.inference
  cpp_input.npz           state[14] f32, image.<role> HWC u8, embedding u16[64,4096],
                          embedding_attention_mask i32[64], action_noise f32[1,48,W]
  reference.json          metadata, timings, rollout values

The initial action noise is captured by wrapping
``inference_openloop.randn_tensor`` (the symbol the pipeline uses) instead of
editing it, so the C++ session can be driven with the exact same noise tensor.
Without that, the two implementations sample different noise and any numeric
comparison is meaningless.

Usage:
  python scripts/81_gwp_torch_reference.py --data-root datasets/... --t5 ... \
      --norm-stats artifacts/norm_stats_delta_zscore_equiv.json --out runs/F_compare
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "lib"))

import env as project_env  # noqa: E402

project_env.load_env()
_env = project_env.require("GWP_SOURCE_DIR", "GWP_CHECKPOINT_DIR", "WAN_BASE_MODEL_DIR")
GWP = _env["GWP_SOURCE_DIR"]
MODEL = _env["GWP_CHECKPOINT_DIR"]
BASE = _env["WAN_BASE_MODEL_DIR"]

sys.path.insert(0, GWP)
sys.path.insert(0, os.path.join(GWP, "scripts"))
os.environ.setdefault("GIGA_MODELS_LIGHT_IMPORT", "1")
os.environ["WAM_GIGA_ROOT"] = os.path.join(GWP, "third_party")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--t5", required=True)
    ap.add_argument("--norm-stats", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.environ.get("RUNS_DIR", os.path.join(REPO_ROOT, "runs")), "F_compare"))
    args = ap.parse_args()

    import torch

    out_dir = os.path.join(WS, args.out)
    os.makedirs(out_dir, exist_ok=True)

    from inference_openloop import get_policy, load_lerobot_v3_episode
    import inference_openloop as iol

    # capture the action noise without touching the pipeline source
    captured = {}
    _orig_randn = iol.randn_tensor

    def spy(shape, **kw):
        tensor = _orig_randn(shape, **kw)
        if isinstance(shape, tuple) and len(shape) == 3 and shape[1] == 48:
            captured["action_noise"] = tensor.detach().clone()
        return tensor

    iol.randn_tensor = spy

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pipe = get_policy(
        checkpoint=MODEL, base_model=BASE, norm_stats=args.norm_stats,
        data_paths=[WS], data_idx=args.episode, device=args.device,
        fixed_t5_path=None, model_dtype="bf16", compile_transformer=False,
        text_encoder_device="cpu", output_dir=out_dir, timing_jsonl=None,
    )

    root = args.data_root if os.path.isabs(args.data_root) else os.path.join(WS, args.data_root)
    front, left, right, state, action, t5_ep, task = load_lerobot_v3_episode(root, args.episode)
    f = args.frame

    # identical to inference_client(): /255 then CHW, float64
    def chw(img):
        a = np.array(img) / 255.0
        return torch.from_numpy(np.transpose(a, (2, 0, 1)))

    emb = torch.load(args.t5, map_location="cpu")
    if isinstance(emb, dict):
        emb = emb.get("t5_embedding", next(iter(emb.values())))
    emb = emb.float()[:64]
    if emb.shape[0] < 64:
        emb = torch.nn.functional.pad(emb, (0, 0, 0, 64 - emb.shape[0]))
    emb_batch = emb.unsqueeze(0)

    obs = {
        "observation.state": torch.from_numpy(state[f][:14]).float().unsqueeze(0),
        "observation.images.cam_high": chw(front[f]),
        "observation.images.cam_left_wrist": chw(left[f]),
        "observation.images.cam_right_wrist": chw(right[f]),
        "prompt_embedding": emb_batch,
        "chunk_idx": 0,
        "episode_idx": args.episode,
        "_quiet": True,
        "_skip_ref_image_save": True,
    }

    t0 = time.perf_counter()
    pred = pipe.inference(obs)
    dt = time.perf_counter() - t0
    pred = np.asarray(pred)
    noise = captured.get("action_noise")

    torch_action = os.path.join(out_dir, "torch_action.npy")
    np.save(torch_action, pred)

    npz_path = os.path.join(out_dir, "cpp_input.npz")
    # The C API takes the language embedding as raw BF16 bits viewed as uint16
    # (local.py: tensor_view(..., np.uint16, b"T,D")). Round-trip through bf16
    # first: the cached tensor already holds bf16-representable values.
    emb_bf16_bits = emb.to(torch.bfloat16).view(torch.uint16).numpy().reshape(64, 4096)
    payload = {
        "state": state[f][:14].astype(np.float32),
        "image.camera_high": np.asarray(front[f], dtype=np.uint8),
        "image.camera_left_wrist": np.asarray(left[f], dtype=np.uint8),
        "image.camera_right_wrist": np.asarray(right[f], dtype=np.uint8),
        "embedding": emb_bf16_bits,
        "embedding_attention_mask": np.ones(64, dtype=np.int32),
    }
    if noise is not None:
        n = noise.detach().float().cpu().numpy()
        payload["action_noise"] = n
    np.savez(npz_path, **payload)

    report = {
        "episode": args.episode,
        "frame": f,
        "task": task,
        "seed": args.seed,
        "norm_stats": os.path.abspath(args.norm_stats),
        "torch_action_shape": list(pred.shape),
        "torch_action_finite": bool(np.isfinite(pred).all()),
        "torch_action_min": float(pred.min()),
        "torch_action_max": float(pred.max()),
        "torch_action_first6": pred[0, :6].tolist(),
        "torch_inference_s": dt,
        "action_noise_captured": noise is not None,
        "action_noise_shape": list(noise.shape) if noise is not None else None,
        "gpu_peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "npz": npz_path,
        "npz_sha256": hashlib.sha256(open(npz_path, "rb").read()).hexdigest(),
        "state": payload["state"].tolist(),
    }
    with open(os.path.join(out_dir, "reference.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
