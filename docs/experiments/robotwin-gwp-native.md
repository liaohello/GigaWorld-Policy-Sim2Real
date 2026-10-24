# RoboTwin + GWP native validation

Status: **executed 2026-10-23**. Plan below is kept as written before the run; the
outcome and the deviations are in [Results](#outcome).

| Stage | Scope | Result |
|---|---|---|
| A | Renderer smoke + one expert episode | PASS |
| B | Demonstration contract, LeRobot v3 conversion, expert replay | PASS |
| C | Normalization statistics + fixed text embeddings | PASS |
| D | GWP 0.5 PyTorch open loop, latency and memory | PASS |
| E | Closed loop on fixed seeds through the native 14-D interface | PASS (interface) / **0 of 3 seeds succeed** |
| F | wam.cpp GGUF conversion and PyTorch↔C++ parity | PASS (conversion, self-check) / **BLOCKED** (parity) |

Details: [`robotwin-gwp-native-results.md`](robotwin-gwp-native-results.md) ·
commands: [`robotwin-gwp-native-repro.md`](robotwin-gwp-native-repro.md) ·
claims→artifacts: [`evidence-index.md`](evidence-index.md).

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

## Outcome

### What the plan got right

Every caution in the list above materialised as a real problem. The GPU pressure
point was real — not from the transformer or the VAE, but because
`pipe.to(device)` also moves the 11.36 GB T5 encoder. The renderer *did* catch an
exception and still exit 0. The converter *did* delete a same-named target
directory. The GWP output *was* already denormalized and already absolute, so the
first bridge design would have double-converted.

### Where execution deviated

| Plan step | What actually happened |
|---|---|
| 1. Two isolated environments | **Three** were needed. `lerobot 0.4.4` and the GWP stack cannot share a resolver: `pip install --dry-run` refuses the combination. Split into sim / policy / data. |
| 3. Convert one, then all 50 | Unchanged, and it paid off: the single-episode check caught the `cam_head` → `cam_high` feature-name rename and the 1.1/255 AV1 delta before 50 episodes were produced. |
| 4. Run the PyTorch path | Three defects had to be fixed first (`docs/experiments/robotwin-gwp-native-results.md`, P-1/P-2/P-3). None were configuration issues; all were code. |
| 5. "validate normalization" | Turned into a specification question: wam.cpp forces unclipped z-score, the official PyTorch script uses q01/q99 min-max. Reconciled with an algebraic identity so the weights never change (E-1). |
| 6. C++ parity | Conversion and self-check passed on the first working build, then stopped at GPU memory. Five device/precision combinations were tried and excluded; the artifact is correct, the card is too small (F-3). |

### What is still open

1. **C++ parity has never been executed.** Inputs are staged; a device with
   ≥ 22 GiB free is the only missing piece.
2. **Zero-shot success is 0/3.** The interface is proven; the policy is not
   adapted to this embodiment. Fine-tuning belongs to M3, not here.
3. **GWP license is unverified** — the snapshot ships without a `LICENSE` file.

None of these are blocking for the stated goal (a native single-task validation
of the interface and the inference path). All three are recorded rather than
smoothed over.
