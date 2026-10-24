# Local patches

Two patches were needed to run the native RoboTwin + GigaWorld-Policy 0.5
pipeline. The diffs are committed; the pristine upstream files are **not**.

| Patch | Applies to | Applied by | Why |
|---|---|---|---|
| `gwp05_inference_openloop.diff` | `scripts/inference_openloop.py` in the GWP checkout | `scripts/setup/patch_gwp.py --write` | Component-level device placement, pinned execution device, action width read from the checkpoint, output-directory plumbing, per-call timing |
| `sapien_urdf_loader.diff` | `sapien/wrapper/urdf_loader.py` in the simulation env's `site-packages` | `scripts/setup/patch_sapien.sh` | Open SRDF as UTF-8; the upstream RoboTwin installer does the same `sed` |

## Where the original files went

A full copy of an upstream source file is not redistributable, and an earlier
revision of this repository committed two of them under `patch/orig/` while the
docs claimed no GWP source was redistributed. That contradiction is resolved by
removing them from Git.

Each patch identifies its original by hash instead — see
`upstream-manifest.json`:

| File | SHA-256 of the pristine version |
|---|---|
| `scripts/inference_openloop.py` | `750625547d2165142828a272fefc50163ea9522dd98be6f4648a26d6dcf3e268` (57 232 B) |
| `sapien/wrapper/urdf_loader.py` | recorded in the manifest (sapien 3.0.0b1 wheel) |

The pristine copies live outside Git under `$LOCKS_DIR/upstream/`:

- `patch_gwp.py --write` copies the file there before editing, and refuses to run
  if the anchors match neither the pristine nor the patched form.
- `patch_sapien.sh` does the same before its `sed`.
- `patch_gwp.py --check` works without any pristine copy: it classifies the file
  as `pristine`, `patched`, or `unknown` by anchor presence.

## Why patches and not forks

The GWP tree ships without Git metadata, so a fork would be unreviewable and its
provenance unverifiable. A patch plus a recorded hash is checkable in one diff,
and `--check` fails loudly when the upstream text moves.

What a diff necessarily does is quote a handful of context lines from the
original. That is the normal form of a patch and is not the same as shipping the
file; the manifest makes the distinction explicit rather than implicit.

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
