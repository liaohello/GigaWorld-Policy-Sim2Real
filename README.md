# GigaWorld-Policy-Sim2Real

Research workspace for reproducing GigaWorld-Policy 0.5, measuring action generation and inference latency, adapting policies to bimanual simulation, and documenting later sim-to-real work.

## Scope

- **M0–M1:** reproduce the official Python and C++ inference paths; save valid action chunks, latency, and peak memory.
- **M2:** evaluate a native RoboTwin task in closed loop.
- **M3:** adapt and fine-tune for the MuJoCo dual SO101 setup.
- **M4:** prepare real-robot deployment with explicit safety checks.
- **M5:** compare GWP, π0.5, and ACT under a shared protocol.

This repository contains experiment code, configurations, source pins, and reports. Model checkpoints, datasets, generated videos, environments, and caches live outside Git and are referenced through local environment variables.

## Repository map

```text
configs/                 model, environment, and experiment configuration
docs/specifications/     stage acceptance criteria
docs/experiments/        run plans, provenance, and results
external/                instructions for separately installed upstream repos
scripts/                 setup, data, inference, training, and evaluation entrypoints
src/gwp_sim2real/        policy, adapter, dataset, control, and evaluation code
tests/                   focused contract checks
```

## Current reproducible target

The first simulation target is RoboTwin 2.0 `beat_block_hammer`, `demo_clean`, Aloha-AgileX, using its native 14-dimensional action interface. A local execution plan and known resource inventory are in [`docs/experiments/robotwin-gwp-native.md`](docs/experiments/robotwin-gwp-native.md). It records the prior resource audit as historical evidence; check local paths, versions, free disk, and GPU availability again before running.

## Quick start

This repository is at the scaffold stage. See [`docs/specifications/v0.1-inference.md`](docs/specifications/v0.1-inference.md) for the first acceptance target and [`external/README.md`](external/README.md) for upstream source handling. No environment or inference run is claimed yet.

Configure resource locations in an ignored `.env` file based on `.env.example`. Never place credentials, private keys, public IPs, or personal server connection strings in tracked files.
