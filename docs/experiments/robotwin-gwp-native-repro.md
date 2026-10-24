# Reproduction commands

Everything below assumes `.env` is filled in (`cp .env.example .env`) and the
upstream checkouts exist. Commands are ordered; each stage is a gate — do not
continue past a failure.

`$PY_SIM`, `$PY_POLICY`, `$PY_DATA` refer to
`$ENVS_DIR/robotwin_gwp_{sim,policy,data}/bin/python`.

## 0 — environments

```bash
bash scripts/setup/env_tools.sh      # cmake>=3.24, protoc with proto3 optional, ffmpeg
bash scripts/setup/env_sim.sh        # py3.10, sapien 3.0.0b1, mplib 0.2.1, curobo, torch 2.4.1
bash scripts/setup/env_policy.sh     # py3.11, torch 2.7.1, diffusers 0.36.0, transformers 4.54.1
bash scripts/setup/env_data.sh       # py3.11, lerobot 0.4.4 (diffusers conflict with the above)
bash scripts/setup/patch_gwp.py --check    # verify the upstream text still matches
bash scripts/setup/patch_gwp.py --write
bash scripts/setup/patch_sapien.sh   # UTF-8 URDF loading
bash scripts/setup/install_adapter.sh
```

Record the result:

```bash
python3 scripts/checks/source_baseline.py     # -> $LOCKS_DIR/source_baseline.json
```

## A — renderer and one expert episode

```bash
$PY_SIM scripts/evaluate/smoke_env.py                 # three cameras, real RGB
bash scripts/data/collect_expert.sh                   # -> $GENERATED_DIR
```

Do not proceed unless the smoke report shows non-constant frames and the
collector logs a success. The renderer can fail while exiting 0.

## B — contract, conversion, replay

```bash
$PY_SIM  scripts/data/probe_hdf5.py --root "$ROBOTWIN_DATA_DIR"
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
$PY_POLICY scripts/data/norm_convention.py \
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

```bash
# server from stage D must still be up
EVAL_GPU=0 bash scripts/evaluate/closed_loop.sh 3 0
```

`TAG_SUFFIX` distinguishes runs of the same (count, seed) made under different
normalization statistics; without it the second run overwrites the first.

## F — wam.cpp

```bash
bash scripts/infer/wam_convert.sh      # merge shards -> PolicySpec -> GGUF
bash scripts/infer/wam_infer.sh        # inspect / validate / predict / compare
```

`wam_infer.sh` exits non-zero on a card without ≥ 22 GiB free; that is the
expected outcome on a shared 24 GiB device, not a broken build.

## Video (optional, for sharing)

```bash
bash scripts/evaluate/make_demo_video.sh   # 42 s, ~190 KB, 320x240
```
