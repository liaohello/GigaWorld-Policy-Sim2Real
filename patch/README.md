# Local patches

Two patches were needed to run the native RoboTwin + GigaWorld-Policy 0.5
pipeline. Both are recorded here with the original file next to the patch
(`orig/`) so the diff can be read without checking anything out.

| Patch | Applies to | Applied by | Why |
|---|---|---|---|
| `gwp05_inference_openloop.diff` | `scripts/inference_openloop.py` in the GWP checkout | `scripts/setup/patch_gwp.py --write` | Component-level device placement, pinned execution device, action width read from the checkpoint, output-directory plumbing, per-call timing |
| `sapien_urdf_loader.diff` | `sapien/wrapper/urdf_loader.py` in the simulation env's `site-packages` | `scripts/setup/patch_sapien.sh` | Open SRDF as UTF-8; the upstream RoboTwin installer does the same `sed` |

## Why these are patches and not forks

The GWP tree ships without Git metadata, so a fork would be unreviewable. A
patch against `orig/` is verifiable in one diff, and `patch_gwp.py --check`
fails loudly if the upstream text has moved.

## What was deliberately not patched

RoboTwin's own installer removes the collision term from `mplib`'s screw-planner
abort condition. That weakens a safety check, and the code path is unreachable
for this embodiment: `assets/embodiments/aloha-agilex/config.yml` selects
`planner: curobo`, so `plan_screw` never runs. Not applied, and recorded in
`docs/experiments/robotwin-gwp-native-results.md`.

## GWP patch in detail

The 206-line diff does four things, in order of how much they mattered:

1. **Component placement.** `get_policy` ended with `pipe.to(device)`, which
   moves the 11.36 GB UMT5 encoder onto the GPU as well. The patch moves only
   the transformer and the VAE, leaving T5 on CPU, and exposes
   `--text-encoder-device {auto,cpu,cuda}` so the old behaviour is one flag away.
2. **Execution device pinning.** With T5 on CPU, `DiffusionPipeline.device`
   starts returning `cpu` (it reads the first `nn.Module` in signature order),
   so VAE inputs were built on the wrong device. The patch derives a subclass
   with `device` and `_execution_device` pinned to the compute device.
3. **Action width from the checkpoint.** `action_dim` was hardcoded to 32 while
   the checkpoint declares `in_action_channels: 16`; the patch reads the value
   from `pipe.transformer.config.in_action_channels`. Wrong width fails
   immediately with a `bmm` shape mismatch, so this is not a silent bug.
4. **Observability.** `--output-dir`, `--timing-jsonl`, and per-call timing so
   the numbers in `docs/experiments/` can be re-derived from a log.
