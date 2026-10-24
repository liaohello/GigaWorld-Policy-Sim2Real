# Environment lock

Recorded 2026-10-23 on the execution host. Every value below was read from the
live environments (`python -V`, `--version`, `import <pkg>.__version__`), not
copied from documentation.

## Host

| Item | Value |
|---|---|
| GPUs | 4 × RTX 4090, 24 091 MiB each |
| NVIDIA driver | 570.86.10 |
| CUDA toolkit | 12.8 (`V12.8.61`), used for the wam.cpp build |
| Host compiler | gcc 11.4.0 (Ubuntu 22.04) |
| CMake | 4.4.4 (conda-forge, installed into the tools env) |
| protoc | libprotoc 35.1 |
| RAM | 503 GiB |

The system CMake is 3.22 and the system protoc is 3.12; both are too old for
wam.cpp (`DOWNLOAD_EXTRACT_TIMESTAMP` needs ≥ 3.24, proto3 `optional` needs a
newer protoc). The tools environment supplies the newer pair — see
`docs/experiments/robotwin-gwp-native-results.md`, defect F-2.

## Environments

Three environments, deliberately not merged. A single environment was attempted
first and the resolver refused: the data stack (`lerobot 0.4.4`) pins
`diffusers`/`transformers` versions that the GWP policy stack does not accept.
That conflict was observed directly with `pip install --dry-run`, not inferred.

| Role | Python | Key packages | Why separate |
|---|---|---|---|
| Simulation | 3.10.22 | `torch 2.4.1+cu121`, `sapien 3.0.0b1`, `mplib 0.2.1`, `curobo 0.0.0` (source build), `numpy 1.26.4`, `h5py 3.16.0`, `av 17.1.0`, `websockets 15.0.1` | SAPIEN 3.0b1 and mplib 0.2.1 need numpy < 2 and an older torch ABI |
| Policy | 3.11.17 | `torch 2.7.1+cu126`, `diffusers 0.36.0`, `transformers 4.54.1`, `numpy 2.4.6`, `einops 0.8.0` | GWP 0.5 ships `_diffusers_version: 0.36.0` |
| Data | 3.11.17 | `torch 2.10.0+cu128`, `lerobot 0.4.4`, `diffusers 0.35.2`, `numpy 2.2.6`, `h5py 3.16.0`, `av 15.1.0` | `lerobot 0.4.4` and `diffusers 0.36.0` cannot coexist |

Environment roots are resolved from `$ENVS_DIR` (see `.env.example`); the three
directories are `robotwin_gwp_sim`, `robotwin_gwp_policy`, `robotwin_gwp_data`,
plus `robotwin_gwp_tools` for CMake, protoc and ffmpeg.

## Package locks

The full `conda list --explicit` and `pip freeze` files are archived outside Git
under `$RUNS_DIR/locks/`. Their SHA-256 values are recorded here so the archived
files can be identified without trusting the filesystem:

| File | Lines | SHA-256 |
|---|---:|---|
| `sim_env_pip_freeze.txt` | 287 | `5186a792c1ea9f49b59feb93b5ac0ee3323fa1ac2e7708a163ecb7c4fb6bf38c` |
| `sim_env_conda_explicit.txt` | 290 | `724f4780eedb920b895b06cd12ea1aceab19d40f6a4d58504819931c08a4f5c9` |
| `policy_env_pip_freeze.txt` | 126 | `6c7cc6bae4eba00bcd49e0db98895065b3ebe031af794ee053d3832bd625b3de` |
| `policy_env_conda_explicit.txt` | 153 | `329e8a7b183b4a3ae60433ef495f86b619bd7b63ef7a2bcad8b1473b6cb2159a` |
| `data_env_pip_freeze.txt` | — | `279d3035a04a557420173902710ceab156532761109828a326dca42c3524a48a` |
| `source_baseline.json` | — | `db5c648d073947b3975380cf8d8690d84cdfbb82e5efe726b07b50a75aed20eb` |

`source_baseline.json` records file counts and SHA-256 values for the four
vendored source trees (RoboTwin, GWP, giga-models, wam.cpp) so an edited tree can
be detected. It is the only way to identify the two trees that ship without Git
metadata.

## Known environment hazards

- `curobo` reports `__version__ = 0.0.0` because it is a source build. The exact
  tag used for this run was not captured → treat as **not verified** and pin it
  explicitly on the next rebuild.
- `XPolicyLab` declares `websockets>=14`. `pip` considered the user site
  directory to satisfy it and never installed it into the environment; the
  import then resolved from `~/.local/...`. Fixed with an explicit
  `pip install --force-reinstall websockets==15.0.1`, and the resolution path is
  now recorded in `$RUNS_DIR/locks/`.
- `sapien`'s URDF loader opens SRDF files as UTF-8. Without the local patch it
  fails on non-ASCII paths; see `patch/sapien_urdf_loader.diff`.
