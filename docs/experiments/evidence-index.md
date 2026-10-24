# Evidence index

Every claim in `docs/` resolves to one artifact. Artifacts live outside Git; this
file records which root they sit under, the path relative to that root, the value
that was read, and how strong the claim is.

Evidence classes follow `AGENTS.md`:

| Class | Meaning |
|---|---|
| `measured` | produced by a run whose log and output are retained |
| `documented` | quoted from an upstream document or source file |
| `inferred` | follows from measured data but was not itself measured |
| `not verified` | asserted or expected, with no retained artifact |

## Where artifacts live

Six roots, defined in `.env.example`. Naming them separately matters: a path like
`open_loop/0_action_plot.png` is meaningless without knowing that it is under
`$RUNS_DIR` and not under `$ARTIFACTS_DIR`.

| Root | Purpose | Holds |
|---|---|---|
| `$RUNS_DIR` | per-run outputs | `A_smoke/`, `B_contract/`, `B_replay/`, `D_openloop/`, `E_closed_loop/`, `F_compare/`, `demo/` |
| `$ARTIFACTS_DIR` | derived model-side files | norm stats, PolicySpec, GGUF + manifests, merged checkpoint |
| `$DATASETS_DIR` | converted datasets and caches | `robotwin_gwp/beat_block_hammer_50_v3/` (parquet, videos, `meta/`, `t5_embedding/`) |
| `$GENERATED_DIR` | episodes collected by this project | `demo_clean_smoke/beat_block_hammer/aloha_agilex/` |
| `$LOGS_DIR` | raw logs | `*.log` |
| `$LOCKS_DIR` | locks, baseline, pristine upstream copy | `*_pip_freeze.txt`, `source_baseline.json`, `upstream/` |

## Claims

### Stage A — renderer and expert data

| ID | Claim | Root + path | Value | Class |
|---|---|---|---|---|
| A1 | Each camera returns a non-blank RGB image with a real dynamic range | `$RUNS_DIR/A_smoke/smoke_report.json` | 3 × 240×320 uint8, per-frame min 0 / max 255, mean ≈ 235–241 | measured |
| A2 | **The smoke probe did not observe motion**: its three saved frames per camera are byte-identical | `$RUNS_DIR/A_smoke/frame{0,1,2}_*.png` | `head_frames_distinct = false`; identical SHA-256 per camera (`844b70d2…`) | measured |
| A3 | Motion is present in a collected episode, which is what the smoke probe could not show | `$GENERATED_DIR/demo_clean_smoke/.../video/episode_0000000.mp4` | 116 frames; planning seed 1 (`failed 1 times / 2 tries`) | measured |

A1 and A2 are deliberately separate. An earlier version of this table merged them
into "consecutive frames differ", which the artifact contradicts: the probe steps
the scene without commanding joints, so the frames are identical. Only A3
establishes that rendered images change over time.

### Stage B — contract, conversion, replay

| ID | Claim | Root + path | Value | Class |
|---|---|---|---|---|
| B1 | Stored demonstrations obey a 14-D contract with a one-step action offset | `$RUNS_DIR/B_contract/hdf5_contract.json` | 50 files, all finite, `max(abs(action[t]-state[t+1])) = 0.0`, 15 Hz | measured |
| B2 | A single-episode LeRobot v3 conversion is contract-correct | `$RUNS_DIR/B_contract/lerobot_contract.json` | keys `observation.images.cam_high\|cam_left_wrist\|cam_right_wrist`; frame 0 vs HDF5 1.10–1.15/255 | measured |
| B3 | All 50 demonstrations convert | `$DATASETS_DIR/robotwin_gwp/beat_block_hammer_50_v3`, `$LOGS_DIR/B_convert_full.log` | 50/50, 5682 frames, 96 s | measured |
| B4 | Expert replay reproduces the recording | `$RUNS_DIR/B_replay/replay_ep0.json` | 119 steps, `check_success() = true`, live vs stored state 0.0 | measured |

### Stage C — statistics and embeddings

| ID | Claim | Root + path | Value | Class |
|---|---|---|---|---|
| C1 | Statistics computed from the converted dataset | `$ARTIFACTS_DIR/norm_stats_delta.json` | 5682 state / 272 736 action samples, 14 non-degenerate dims of 32 | measured |
| C2 | A fixed text embedding exists for every episode | `$DATASETS_DIR/robotwin_gwp/beat_block_hammer_50_v3/t5_embedding/t5_index.json` + `episode_*.pt` | 50 × `[512, 4096]` f32, nonzero rows == declared length, 46 distinct instructions | measured |

### Stage D — open loop

| ID | Claim | Root + path | Value | Class |
|---|---|---|---|---|
| D1 | Component placement keeps the pipeline inside 24 GiB | `$RUNS_DIR/D_openloop/device_placement.json` | T5 on CPU (bf16), transformer+VAE on GPU, allocated 13.47 / reserved 13.66 GB, peak 13.98 GB | measured |
| D2 | Open-loop latency | `$RUNS_DIR/D_openloop/openloop_bench.json`, `server_timing.jsonl` | model p50 674.9 / p95 702.9 ms; e2e p50 727.5 / p95 753.2 ms | measured |
| D3 | Zero-shot open-loop error against ground truth | `$RUNS_DIR/D_openloop/open_loop/0_action_plot.png` | overall action MSE 0.169459; recorded, not a gate | measured |

### Stage E — closed loop

| ID | Claim | Root + path | Value | Class |
|---|---|---|---|---|
| E1 | Closed loop consumes real policy actions | `$RUNS_DIR/E_closed_loop/n1_seed0_minmax/` | seed 100000, 400-step budget exhausted, `_result.txt` fail, MP4 present | measured |
| E2 | The failure is stable across seeds | `$RUNS_DIR/E_closed_loop/n3_seed0/` | seeds 100000/100002/100003, 0/3, no NaN/shape/protocol error | measured |
| E3 | Per-call action trace retained | `$RUNS_DIR/E_closed_loop/n1_seed0_trace/action_trace.jsonl` | 50 calls × 48×14, joint-1 target 0.850 → 43.32 rad | measured |
| E4 | **Attribution**: the drift is a policy/embodiment fit problem rather than a normalization or plumbing fault | `$RUNS_DIR/E_closed_loop/n1_seed0_minmax/` vs `n3_seed0/` vs `$RUNS_DIR/D_openloop/open_loop/0_action_plot.png` | same trajectory shape under two conventions; dims 8–13 near-constant while GT moves | **inferred** |

E4 is an inference, not a measurement: two runs agreeing on a shape rules out
"one convention is broken", but nothing here isolates *which* part of the policy
is responsible. Fine-tuning would be the experiment that tests it.

### Stage F — wam.cpp

| ID | Claim | Root + path | Value | Class |
|---|---|---|---|---|
| F1 | The GGUF converts with the official toolchain | `$ARTIFACTS_DIR/gwp05_robotwin_14d_mot_vae_bf16.gguf` (+ manifest) | 23.51 GB: gwp 12.04 F32→BF16, t5 11.36 BF16, vae 0.11 | measured |
| F2 | The artifact passes the upstream self-check | `$RUNS_DIR/F_compare/inspect.txt`, `validate.txt` | `valid: true`, `gwp05`, `state_dim 14`, `action_horizon 48` | measured |
| F3 | C++ GPU inference did not fit on this card | `$RUNS_DIR/F_compare/predict_cuda.log` | `cudaMalloc failed` requesting 10 835 MiB for one T5 shard | measured |
| F4 | PyTorch and C++ agree numerically | `$RUNS_DIR/F_compare/{cpp_input.npz,torch_action.npy,reference.json}` | inputs and PyTorch output exist; **no C++ output** | not verified |

F3 establishes what failed, not what would succeed. The required free memory is an
estimate derived from the resident set of the components, and no run has yet shown
that 22 GiB is enough — see `docs/specifications/v0.3-wam-cpp-parity.md`.

## Deliberately not in Git

| Artifact | Size | Why excluded |
|---|---|---|
| GWP 0.5 checkpoint | ~24.1 GB | model weights |
| Wan 2.2 TI2V 5B base model | ~34.2 GB | model weights |
| LeRobot v3 dataset (50 episodes) | ~1 GB | derived dataset |
| T5 embedding cache | ~170 MB | derived cache |
| GGUF artifacts | 23.51 + 35.66 GB | build products |
| Episode videos | 0.2–0.5 MB each | regenerated by the evaluation script; `.gitignore` excludes `*.mp4` |
| Package locks | 126–290 lines each | under `$LOCKS_DIR`, hashed in `environment-lock.md` |
| Pristine upstream source | 57 KB | **not redistributable**; kept under `$LOCKS_DIR/upstream/`, identified by SHA-256 in `patch/upstream-manifest.json` |

One still frame (`docs/assets/poster.png`) is committed so the README renders
without the video.
