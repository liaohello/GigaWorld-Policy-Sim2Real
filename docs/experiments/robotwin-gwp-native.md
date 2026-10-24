# RoboTwin + GWP native validation plan

Status: **planned; not run from this repository**. The notes below summarize a prior resource audit and must be rechecked on the execution host before use.

## Target and sequence

Use RoboTwin 2.0 task `beat_block_hammer`, split `demo_clean`, embodiment `aloha_agilex`. Preserve its native 14-D action contract.

1. Create separate simulation and policy environments; record package locks and source revisions.
2. Verify the RoboTwin renderer produces real RGB frames, then run one expert episode into a new output directory.
3. Convert one demonstration to LeRobot v3, inspect image encoding, state/action shape, timestamps, and replay it as expert data. Expand to all 50 demonstrations only after the single-episode contract passes.
4. Run the real GWP PyTorch inference path on recorded observations. Confirm finite action chunks and measure latency/memory.
5. Add a GWP adapter to the RoboTwin policy interface; validate camera names, RGB handling, action normalization, absolute-vs-delta semantics, and chunk execution before closed-loop evaluation on fixed seeds.
6. Treat wam.cpp as a separate parity milestone: pin source, convert using the official toolchain, then compare C++ and PyTorch outputs and resource usage on identical inputs.

Expert replay and model control must be reported separately. Zero-shot task success is an experimental result, not the criterion for proving the inference and interface path works.

## Prior resource inventory

The previous server-side audit reported these assets as downloaded and CRC-checked. The machine-specific paths were intentionally omitted; set them locally through `.env` or the execution environment:

| Resource | Previously audited state |
|---|---|
| RoboTwin 2.0 source and XPolicyLab | Extracted source archives; Git metadata not guaranteed |
| RoboTwin assets | Extracted, approximately 16.6 GB |
| `beat_block_hammer/demo_clean/aloha_agilex` | 50 HDF5 demonstrations, 50 MP4 videos, and instruction metadata; archive CRC passed |
| GWP 0.5 checkpoint | Three safetensors shards; approximately 24.1 GB |
| Wan 2.2 TI2V 5B base model | Approximately 34.2 GB; audit reported the required components present |

The audit expected the simulation and policy environments to be isolated. It also identified a likely 24 GiB GPU pressure point: the default PyTorch pipeline can place the transformer, T5 text encoder, and VAE on the same GPU. A fixed T5 embedding path alone does not imply that the T5 weights are unloaded. Verify component placement and peak memory in the actual run.

## Validation cautions

- The RoboTwin renderer may catch exceptions and still return a zero exit code; inspect logs and actual RGB output.
- The LeRobot converter may remove an existing target dataset directory. Use a fresh output/repo id and verify the resolved destination before running it.
- Keep image decoding in the upstream RGB helper path. Do not apply an extra channel swap or decode step without inspecting the adapter contract.
- Check whether GWP outputs are denormalized absolute joint targets before adding any delta or denormalization logic in the simulator bridge.
- Check current GPU occupancy before launching. Manage only processes started for this project.
- New downloads should be inventoried by URL, pinned revision, size, and checksum. If the expected addition exceeds 30 GB, prepare a local-download and transfer manifest first.

## Evidence to save

Save commands, environment locks, source hashes, patches, raw logs, action arrays, videos, simulator success flags, latency/memory metrics, and per-stage PASS/FAIL/BLOCKED status under an external run directory. Keep large data and artifacts out of Git. Do not record public IPs, usernames, SSH commands, credentials, private filesystem paths, or machine identifiers in tracked records.
