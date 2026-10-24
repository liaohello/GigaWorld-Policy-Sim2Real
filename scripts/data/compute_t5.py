#!/usr/bin/env python
"""Stage C2: episode-level T5 embedding cache built with the diffusers UMT5.

Why not upstream ``scripts/compute_t5_embedding.py``: it hard-requires the
*original* Wan release layout (``models_t5_umt5-xxl-enc-bf16.pth`` +
``google/umt5-xxl``) plus ``wan/modules/t5.py``. This project only has the
diffusers conversion (``Wan2.2-TI2V-5B-Diffusers/{text_encoder,tokenizer}``),
and re-downloading ~11 GB to satisfy a loader difference is not justified.
The weights are the same UMT5-XXL.

This script therefore reproduces ``WAPipeline._get_t5_prompt_embeds`` from the
inference code path *exactly* (same tokenizer call: padding=max_length,
max_length=512, truncation, add_special_tokens; same prompt_clean; same
truncate-to-seq_len + zero-pad back to 512) and writes the identical on-disk
contract: ``<root>/<t5_folder_name>/episode_{index:06d}.pt`` holding a
``[512, 4096]`` float32 tensor.

``load_lerobot_v3_episode`` slices ``[:64]`` from it, which matches what the
server does for the live embedding.

Usage:
  python scripts/data/compute_t5.py --root datasets/robotwin_gwp/... \
      --wan ${WAN_BASE_MODEL_DIR:?set WAN_BASE_MODEL_DIR} --device cuda:0
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import html
import json
import os
import sys

import numpy as np


def prompt_clean(text: str) -> str:
    """Same cleaning order as inference_openloop.prompt_clean (ftfy -> html -> ws)."""
    try:
        import ftfy

        text = ftfy.fix_text(text)
    except Exception:
        pass
    text = html.unescape(html.unescape(text))
    import regex as re

    return re.sub(r"\s+", " ", text).strip()


def load_episode_tasks(root: str) -> dict[int, str]:
    import pandas as pd

    paths = sorted(glob.glob(os.path.join(root, "meta", "episodes", "*", "*.parquet")))
    paths += sorted(glob.glob(os.path.join(root, "meta", "episodes", "*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no meta/episodes parquet under {root}")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    out: dict[int, str] = {}
    for _, row in df.iterrows():
        tasks = row["tasks"]
        if isinstance(tasks, (list, tuple, np.ndarray)):
            tasks = list(tasks)
            text = str(tasks[0]) if tasks else ""
        else:
            text = str(tasks)
        out[int(row["episode_index"])] = text
    return dict(sorted(out.items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LeRobot dataset root")
    ap.add_argument("--wan", required=True, help="diffusers Wan model dir")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--t5-folder-name", default="t5_embedding")
    ap.add_argument("--text-len", type=int, default=512)
    ap.add_argument("--max-episodes", type=int, default=0)
    args = ap.parse_args()

    root = args.root if os.path.isabs(args.root) else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.root
    )

    import torch
    from transformers import AutoTokenizer, UMT5EncoderModel

    tasks = load_episode_tasks(root)
    if args.max_episodes:
        tasks = dict(list(tasks.items())[: args.max_episodes])

    tok = AutoTokenizer.from_pretrained(os.path.join(args.wan, "tokenizer"))
    te = UMT5EncoderModel.from_pretrained(
        os.path.join(args.wan, "text_encoder"), torch_dtype=torch.bfloat16
    ).to(args.device)
    te.eval()

    out_dir = os.path.join(root, args.t5_folder_name)
    os.makedirs(out_dir, exist_ok=True)

    index = []
    for ep_idx, raw_text in tasks.items():
        text = prompt_clean(raw_text or "")
        inputs = tok(
            [text],
            padding="max_length",
            max_length=args.text_len,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = inputs.input_ids, inputs.attention_mask
        seq_len = int(mask.gt(0).sum(dim=1).long()[0])
        with torch.no_grad():
            emb = te(ids.to(args.device), mask.to(args.device)).last_hidden_state
        emb = emb.to(dtype=torch.bfloat16, device=args.device)
        emb = emb[0, :seq_len]
        emb = torch.cat(
            [emb, emb.new_zeros(args.text_len - emb.size(0), emb.size(1))], dim=0
        ).float().cpu()

        path = os.path.join(out_dir, f"episode_{int(ep_idx):06d}.pt")
        torch.save(emb, path)
        digest = hashlib.sha256(emb.numpy().tobytes()).hexdigest()[:16]
        rec = {
            "episode_index": int(ep_idx),
            "instruction_raw": raw_text,
            "instruction_clean": text,
            "seq_len": seq_len,
            "path": path,
            "shape": list(emb.shape),
            "finite": bool(torch.isfinite(emb).all()),
            "abs_max": float(emb.abs().max()),
            "sha256_16": digest,
        }
        index.append(rec)
        print(f"[ep {ep_idx:03d}] seq_len={seq_len:3d} absmax={rec['abs_max']:.4f} text={text[:70]!r}")

    with open(os.path.join(out_dir, "t5_index.json"), "w", encoding="utf-8") as f:
        json.dump({"wan": args.wan, "text_len": args.text_len, "episodes": index}, f, indent=2)
    print(f"wrote {len(index)} embeddings -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
