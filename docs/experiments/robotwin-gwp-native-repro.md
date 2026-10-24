# Reproduction commands

Commands are ordered; each stage is a gate — do not continue past a failure.

## 0.1 — How `.env` is loaded

Every shell script sources `scripts/lib/env.sh`, which reads `$REPO_ROOT/.env`
with `set -a` (so the values are exported to child processes) and then unsets it
again. Values already present in the environment win, so a one-off override works:

```bash
cp .env.example .env        # fill in paths outside this repository
RUNS_DIR=/tmp/run bash scripts/evaluate/smoke_env.py   # override for one command
```

Python entrypoints that can be launched directly call `scripts/lib/env.py`'s
`load_env()`, so `python3 scripts/evaluate/smoke_env.py` behaves the same as the
shell wrapper. A missing variable fails at the first line that needs it with
`missing environment variables: ...` rather than producing a path named `None`.

There is no `source .env` in the docs on purpose: `.env` is not shell syntax
guaranteed to be safe to execute, and half the entrypoints are Python.

## 0.2 — Environments

```bash
bash scripts/setup/env_tools.sh      # cmake>=3.24, protoc with proto3 optional, ffmpeg
bash scripts/setup/env_sim.sh        # py3.10, sapien 3.0.0b1, mplib 0.2.1, curobo, torch 2.4.1
bash scripts/setup/env_policy.sh     # py3.11, torch 2.7.1, diffusers 0.36.0, transformers 4.54.1
bash scripts/setup/env_data.sh       # py3.11, lerobot 0.4.4 (diffusers conflict with the above)

python3 scripts/setup/patch_gwp.py --check   # classify upstream as pristine/patched/unknown
python3 scripts/setup/patch_gwp.py --write   # apply, keeping the pristine copy outside Git
bash scripts/setup/patch_sapien.sh           # UTF-8 URDF loading
bash scripts/setup/install_adapter.sh        # renders the adapter into the RoboTwin tree
```

`patch_gwp.py` is a Python tool, not a shell script: `--check` exits 2 when the
anchors match neither the pristine nor the patched form, which means the upstream
file is a different revision than this patch was written against.

Record the result:

```bash
python3 scripts/checks/source_baseline.py     # -> $LOCKS_DIR/source_baseline.json
```

## A — renderer and one expert episode

```bash
$PY_SIM scripts/evaluate/smoke_env.py                 # three cameras, real RGB
bash scripts/data/collect_expert.sh                   # -> $GENERATED_DIR
```

Do not proceed unless the smoke report shows non-constant frames and the collector
logs a success. The renderer can fail while exiting 0.

Note what the smoke probe does *not* show: it steps the scene without commanding
joints, so its three frames are byte-identical (`head_frames_distinct = false`).
Motion is evidenced by the collected episode, not by this probe.

## B — contract, conversion, replay

```bash
$PY_SIM  scripts/data/probe_hdf5.py
bash scripts/data/convert_lerobot.sh smoke            # one episode first
$PY_DATA scripts/data/check_lerobot.py --repo "$DATASETS_DIR/robotwin_gwp/beat_block_hammer_smoke_v3" --episodes 0
bash scripts/data/convert_lerobot.sh full             # then all 50
$PY_SIM  scripts/evaluate/expert_replay.py --episode 0
```

The converter deletes an existing target dataset with the same name — always
convert into a fresh, version-suffixed directory.

## C — statistics and text embeddings

```bash
bash scripts/data/compute_norm.sh
$PY_POLICY scripts/data/compute_t5.py --root "$DATASETS_DIR/robotwin_gwp/beat_block_hammer_50_v3" \
    --wan "$WAN_BASE_MODEL_DIR" --device cuda:0
python3 scripts/data/norm_convention.py \
    --src "$ARTIFACTS_DIR/norm_stats_delta.json" \
    --out "$ARTIFACTS_DIR/norm_stats_delta_zscore_equiv.json"
```

## D — open loop

```bash
GPU=2 bash scripts/infer/gwp_server.sh &              # ZMQ server on 11444
$PY_POLICY scripts/infer/openloop_bench.py \
    --data-root "$DATASETS_DIR/robotwin_gwp/beat_block_hammer_50_v3" \
    --episode 0 --frame 0 \
    --t5 "$DATASETS_DIR/robotwin_gwp/beat_block_hammer_50_v3/t5_embedding/episode_000000.pt" \
    --host 127.0.0.1 --port 11444 --warmup 10 --iters 30 \
    --out "$RUNS_DIR/D_openloop"
```

## E — closed loop

The statistics file decides the normalization convention, and the two runs in the
report differ only by that. Reproduce both, with distinct tags so they do not
overwrite each other:

```bash
# (a) q01/q99 min-max, the server default -> n1_seed0_minmax, 0/1
EVAL_GPU=0 TAG_SUFFIX=_minmax bash scripts/evaluate/closed_loop.sh 1 0

# (b) z-score-equivalent bounds, matching wam.cpp -> n3_seed0, 0/3
NORM_STATS="$ARTIFACTS_DIR/norm_stats_delta_zscore_equiv.json" \
  GPU=2 bash scripts/infer/gwp_server.sh        # restart the server with those stats
EVAL_GPU=0 bash scripts/evaluate/closed_loop.sh 3 0
```

`TAG_SUFFIX` exists because without it a second run of the same `(count, seed)`
overwrites the first — which is how an earlier version of this work lost the
min-max artifacts.

## F — wam.cpp

```bash
bash scripts/infer/wam_convert.sh      # merge shards -> PolicySpec -> GGUF
$PY_POLICY scripts/infer/gwp_torch_reference.py \
    --data-root "$DATASETS_DIR/robotwin_gwp/beat_block_hammer_50_v3" --episode 0 --frame 0 \
    --t5 "$DATASETS_DIR/robotwin_gwp/beat_block_hammer_50_v3/t5_embedding/episode_000000.pt" \
    --norm-stats "$ARTIFACTS_DIR/norm_stats_delta_zscore_equiv.json" --device cuda:0
bash scripts/infer/wam_infer.sh        # inspect -> validate -> predict -> compare
```

`wam_infer.sh` exits 3 when no C++ action file is produced (the expected outcome
on a shared 24 GiB device; it says so on the way out), 2 when the artifact fails
the upstream self-check, and otherwise hands off to `scripts/infer/wam_compare.py`.
A tolerance is applied only if you ask for one:

```bash
MAX_ABS_TOL=0.02 bash scripts/infer/wam_infer.sh
```

## Video (optional, for sharing)

```bash
bash scripts/evaluate/make_demo_video.sh   # 42 s, ~190 KB, 320x240
```
