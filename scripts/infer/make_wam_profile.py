#!/usr/bin/env python
"""Derive the wam.cpp GWP05 PolicySpec used for this project's C++ comparison.

The released profile ``profiles/gwp05_robotwin_14d_zscore.json`` cannot be used
as-is with the public checkpoint, for two independently verifiable reasons:

1. ``load_policy_profile`` requires ``state.model_dim == action.model_dim ==
   transformer.in_action_channels``. The profile says 14; the public checkpoint
   (Giga-World-Policy-0.5/config.json) says ``in_action_channels: 16`` and its
   ``state_encoder.in_proj.weight`` / ``action_encoder.in_proj.weight`` are
   ``[2, 16, 128]``. The converter therefore aborts with "PolicySpec model
   dimensions differ from the checkpoint". 14 real dims + 2 zero pad = 16.
2. Its canvas geometry (head 320x256 on top, wrists 160x128 below) does not
   match the published PyTorch pipeline, which builds 320x192 for the head and
   160x192 for each wrist (``get_ref_image_3views`` with dst_size=(320,384),
   ``top_h = 384//2``). Comparing C++ against PyTorch is only meaningful with
   the same image contract.

Everything else (image roles, view resize modes, state/action field names,
recovery indices, unclipped z_score) is copied from the released profile
unchanged, and this script asserts that.

Usage:
  python scripts/infer/make_wam_profile.py --src src/wam.cpp/profiles/gwp05_robotwin_14d_zscore.json \
      --out artifacts/wam_profiles/gwp05_robotwin_14d_zscore_public05.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os

CANVAS = (384, 320)  # height, width
HEAD = (192, 320)
WRIST = (192, 160)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dim", type=int, default=16)
    args = ap.parse_args()

    with open(args.src) as f:
        profile = json.load(f)
    src = copy.deepcopy(profile)

    profile["identity"]["profile"] = "gwp05_robotwin_dual_arm_14d_zscore_public05"
    profile["identity"]["checkpoint_revision"] = "open-gigaai/Giga-World-Policy-0.5 (public release)"
    profile["identity"]["training_dataset"] = "robotwin beat_block_hammer demo_clean (50 demos)"

    images = profile["images"]
    images["composition"] = {"kind": "canvas", "height": CANVAS[0], "width": CANVAS[1]}
    images["views"]["camera_high"].update(
        {"target_height": HEAD[0], "target_width": HEAD[1], "rect": [0, 0, HEAD[1], HEAD[0]]}
    )
    images["views"]["camera_left_wrist"].update(
        {"target_height": WRIST[0], "target_width": WRIST[1], "rect": [0, HEAD[0], WRIST[1], WRIST[0]]}
    )
    images["views"]["camera_right_wrist"].update(
        {"target_height": WRIST[0], "target_width": WRIST[1],
         "rect": [WRIST[1], HEAD[0], WRIST[1], WRIST[0]]}
    )

    for domain in ("state", "action"):
        profile[domain]["model_dim"] = int(args.model_dim)
        # real_dim/fields/recovery stay exactly as released
        assert len(profile[domain]["fields"]) == profile[domain]["real_dim"]

    # --- invariants that must not drift from the released profile
    for key in ("roles", "color_space", "pixel_range", "tensor_layout", "resample_boundary"):
        assert profile["images"][key] == src["images"][key], key
    for domain in ("state", "action"):
        assert profile[domain]["fields"] == src[domain]["fields"], domain
        assert profile[domain]["real_dim"] == src[domain]["real_dim"], domain
    assert profile["action"]["horizon"] == src["action"]["horizon"]
    assert profile["action"]["recovery"] == src["action"]["recovery"]
    assert profile["normalization"]["state"]["kind"] == "z_score"
    assert profile["normalization"]["action"]["kind"] == "z_score"
    assert profile["normalization"]["epsilon"] == src["normalization"]["epsilon"]
    assert profile["language"] == src["language"]

    area = sum(v["rect"][2] * v["rect"][3] for v in profile["images"]["views"].values())
    assert area == CANVAS[0] * CANVAS[1], (area, CANVAS)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(profile, f, indent=2)

    print(json.dumps({
        "out": args.out,
        "canvas": CANVAS,
        "views": {k: {"target": [v["target_height"], v["target_width"]], "rect": v["rect"]}
                  for k, v in profile["images"]["views"].items()},
        "model_dim": profile["state"]["model_dim"],
        "real_dim": profile["state"]["real_dim"],
        "normalization": profile["normalization"],
        "covered_area": area,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
