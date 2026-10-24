# RoboTwin + GWP native validation — results

Measured 2026-10-22 → 2026-10-23. Every number below comes from a retained
artifact; see [`evidence-index.md`](evidence-index.md) for the mapping.

Environment and host facts: [`environment-lock.md`](environment-lock.md).
Commands: [`robotwin-gwp-native-repro.md`](robotwin-gwp-native-repro.md).

## A — Renderer and expert data

| Check | Value |
|---|---|
| Cameras | `cam_head`, `cam_left_wrist`, `cam_right_wrist`, 240×320 uint8 |
| Pixel range within one frame | min 0, max 255 (not blank, not constant) |
| Consecutive frames | **identical** — the probe steps the scene without commanding joints; `head_frames_distinct = false` |
| Motion evidence | the collected episode's 116-frame video, not the probe |
| Expert episode | planning seed 0 failed, seed 1 succeeded (`failed 1 times / 2 tries`) |
| Output | 1 HDF5 + 1 MP4, 116 frames |
| Original dataset | 50 files, modification times unchanged |

The upstream installer's own render test prints `Sapien_TEST … Render Well`; that
string is not sufficient evidence on its own, so the frames were also written to
disk and inspected. Note the renderer catches exceptions and still returns exit
code 0 (Q-1).

## B — Demonstration contract and conversion

| Check | Value |
|---|---|
| Files | 50 HDF5 |
| Action dimension | 14 (`[left_arm 6, left_ee 1, right_arm 6, right_ee 1]`) |
| Finiteness | all finite |
| Rate | 15 Hz |
| `action[t]` vs `state[t+1]` | max absolute error **0.0** |
| Image encoding | JPEG inside a length-prefixed container, `XPL-RGB1` marker |
| Conversion | 50/50 episodes, 5682 frames, 96 s wall clock |
| Feature names after conversion | `observation.images.cam_high`, `cam_left_wrist`, `cam_right_wrist` |
| Re-rendered frame 0 vs stored frame | 0.68/255 mean absolute difference |
| Converted video frame 0 vs stored frame | 1.10–1.15/255 (AV1 CRF 18) |
| Expert replay | 119 steps, `check_success() = true` |
| Replay state fidelity | live vs stored frame-0 state error **0.0** |

The `action[t] == state[t+1]` identity is what makes the delta/absolute question
answerable: the stored actions are targets for the *next* step, not the current
pose. The conversion preserves that offset.

## C — Normalization and text embeddings

| Check | Value |
|---|---|
| State samples | 5682 |
| Action samples | 272 736 (5682 × 48) |
| Non-degenerate action dims | 14 of 32; dims 14/15 are structural padding |
| Embedding files | 50 × `[512, 4096]` float32 |
| Nonzero rows vs declared length | equal for every file |
| Distinct instructions | 46 |
| Repo conversion vs the statistics used in the E runs | identical: `q01`/`q99` match at `atol = 0` on the real 32-dim file |

## D — Open loop

| Metric | p50 | p95 |
|---|---:|---:|
| Model forward | 674.9 ms | 702.9 ms |
| Preprocess | 6.5 ms | 8.2 ms |
| Postprocess | 1.7 ms | 4.7 ms |
| Transport | 43.4 ms | 48.6 ms |
| End to end | 727.5 ms | 753.2 ms |

| Check | Value |
|---|---|
| Output | `[48, 14]` float32, all finite |
| Range | `[-1.598, 2.395]` rad |
| GPU after load | allocated 13.47 GB / reserved 13.66 GB |
| Peak reserved | 13.98 GB (card shared with ~5.6 GB of other users' work) |
| Zero-shot error, 10 chunks | overall action MSE 0.169459 |

The upstream report quotes 122 ms on an A800. That is not a gate here: all four
4090s were at 95–99 % utilisation from other jobs during the measurement.

## E — Closed loop

| Run | Normalization | Seeds | Success |
|---|---|---|---:|
| `n1_seed0_minmax` | q01/q99 min-max (script default) | 100000 | 0/1 |
| `n3_seed0` | z-score (matches wam.cpp) | 100000 / 100002 / 100003 | 0/3 |

Interface-side: real policy actions reached the simulator, every episode produced
a video and a result record, consecutive episodes reset cleanly, and no NaN,
shape, or protocol error occurred.

Behaviour: the first chunk is well-scaled (`[-0.13, 0.86, 0.63, -0.37, …]`), then
the *commanded* joint-1 target climbs by 0.8–1.0 rad per chunk while the measured
joint saturates at the SAPIEN limit:

| Run | Episode | Chunks | joint-1 at chunk 1 | joint-1 at chunk 50 |
|---|---|---:|---:|---:|
| `n1_seed0_minmax` | 0 | 50 | 0.860 | 44.17 |
| `n3_seed0` | 0 | 50 | 0.855 | 43.93 |
| `n3_seed0` | 1 | 50 | 0.836 | 43.12 |
| `n3_seed0` | 2 | 50 | 0.855 | 44.53 |
| `n1_seed0_trace` | 0 | 50 | 0.850 | 43.32 |

All four episodes exhaust the 400-step budget (50 chunks × 8 executed steps).
Two different normalization conventions produce the same trajectory shape, which
is why this is classified as a policy/embodiment fit problem rather than a
plumbing bug.

The open-loop plot explains the mechanism: dimensions 0–7 track the ground truth
loosely, while dimensions 8–13 stay near-constant even where the ground truth
moves. A near-constant output on those joints, fed back as absolute targets,
produces exactly the slow drift seen in the simulator.

Incidental observation: at the home pose the wrist cameras see mostly wall and
table (mean ≈ 235/255). The stored HDF5 frames show the same thing, so this is
the data at that pose rather than a rendering or decoding fault.

## F — wam.cpp

| Check | Value |
|---|---|
| Source | `e189107a5c9424781dea58c0481355350a051385`, Apache-2.0 |
| `mot-vae-bf16` artifact | 23.51 GB — `gwp` 12.04 GB (F32→BF16), `t5` 11.36 GB (BF16→BF16), `vae` 0.11 GB (encoder + `quant_conv` only) |
| `source-f32` artifact | 35.66 GB, produced for the CPU path |
| `wam-inspect` / `wam-validate` | `valid: true`, architecture `gwp05`, `state_dim 14`, `action_horizon 48` |
| Tensor namespaces | `gwp.` 1664, `t5.` 242, `vae.` 82 |

Five configurations were tried:

| # | Artifact | Backend | Precision | Outcome |
|---|---|---|---|---|
| 1 | `mot-vae-bf16` | cuda | bf16 | `cudaMalloc failed` requesting 10 835 MiB for a T5 shard after ~21.9 GiB resident |
| 2 | `mot-vae-bf16` | cpu | bf16 | rejected: "the BF16 latency profile requires CUDA" |
| 3 | `mot-vae-bf16` | cpu | f32 | rejected: "F32 execution requires an F32 GWP artifact" |
| 4 | `source-f32` | cpu | f32 | `failed to load resident GWP weights` (code 5, `resource_exhausted`, no details); process RSS stayed at 4.5 MB, so it returned before allocating |
| 5 | `source-f32` | automatic | f32 | reached allocation: 22 970 MiB requested, `cudaMalloc failed` — proves the f32 artifact is structurally sound |

Excluded as causes: host RAM (a 24 GiB `numpy` allocation succeeds; 384 GB
available; no cgroup limit), tensor prefixes (identical across both artifacts),
and artifact corruption (#5 reaches the allocator).

Conclusion: 21.9 GiB of resident weights do not fit beside other tenants on a
24 GiB card. The upstream README's reference platform is an 80 GB A800.

## Defects found and fixed

| ID | Symptom | Root cause | Fix |
|---|---|---|---|
| P-1 | 24 GiB card filled at load time | `pipe.to(device)` moves the 11.36 GB T5 encoder too | component-level placement; T5 stays on CPU (fixed embeddings mean it is never executed: `inference()` passes `prompt_embeds` and `guidance_scale = 0.0` disables the uncond branch) |
| P-2 | `Input type (CPUBFloat16Type) and weight type (CUDABFloat16Type)` | `DiffusionPipeline.device` returns the device of the first `nn.Module` in signature order, which is T5 → `cpu` | subclass pinning `device` and `_execution_device` to the compute device |
| P-3 | `bmm` shape mismatch inside the transformer | `inference()` hardcodes `action_dim = 32`; the checkpoint is `in_action_channels = 16` (encoder weights `[2, 16, 128]`) | width read from `transformer.config.in_action_channels`; 14 real dims + 2 zero-padded, matching wam.cpp's `model_dim 16 / real_dim 14` |
| E-1 | Same observation yields different inputs in C++ and PyTorch | wam.cpp's converter forces unclipped z-score; the official script uses q01/q99 min-max | set `q01' = mean - std`, `q99' = mean + std`; then `2(x-(μ-σ))/2σ - 1 = (x-μ)/σ` exactly, so min-max and z-score become algebraically identical without touching weights |
| E-2 | `wam-inspect` rejects the artifact: "normalization std is too small", index 14 | padding dims 14/15 have zero standard deviation by construction | separate norm file with std = 1 on those dims; under z-score they stay exactly 0, matching PyTorch's zero padding |
| F-2 | wam.cpp will not configure | system CMake 3.22 lacks `DOWNLOAD_EXTRACT_TIMESTAMP` (needs ≥ 3.24); protoc 3.12 lacks proto3 `optional`; direct GitHub downloads die with a TLS EOF | conda-forge CMake 4.4.4 and protoc 35.1 in the tools env; llama.cpp from the `codeload` archive, SHA-256 matched against the value pinned in `cmake/Dependencies.cmake` |
| F-3 | C++ inference cannot run | `cudaMalloc failed` for a 10 835 MiB T5 shard with ~21.9 GiB already resident, on a 24 091 MiB card shared with other tenants | none available; documented as BLOCKED. The "~22 GiB free" figure is an **estimated attempt condition**, not a measured requirement — the peak working set of a complete inference was never observed |

## Environment quirks worth remembering

| ID | Quirk | Handling |
|---|---|---|
| Q-1 | RoboTwin reads `assets/objects/objaverse/list.json` through a relative path at import time | every entrypoint must run from the repository root; the wrapper `chdir`s explicitly |
| Q-2 | A worktree copied without `data/` also loses `data/decode_image_bit.py`, which `envs/utils/pkl2hdf5.py` imports | copy the module back; the dataset directory stays a symlink |
| Q-3 | `websockets` import resolved from the user site directory because pip considered `>= 14` already satisfied | force-reinstall into the environment; recorded in the lock |
| Q-4 | The LeRobot converter deletes an existing target dataset with the same name | always convert into a fresh, version-suffixed directory |
| Q-5 | `compute_wam_task_norm.py` prints an alignment warning | false positive: LeRobot stores `names` as `[[...]]`, the checker indexes into it and gets `None`. `state` and `action` names are identical in `meta/info.json`; the delta mask (6 True + False + 6 True + False) is correct |
| Q-6 | Camera names differ either side of the bridge | simulator uses `cam_head`; LeRobot features use `cam_high`; the adapter maps explicitly |

## What these numbers do not prove

- That the policy can solve the task. It cannot, zero-shot, on this embodiment —
  and no fine-tuning was attempted.
- That the C++ path agrees with PyTorch. It has never run to completion.
- That the latency figures are representative of an idle machine. They are not.
- That the 14-D contract holds for any embodiment other than Aloha-AgileX.
