# Experiment scripts

Keep executable entrypoints small, parameterized, and safe to rerun. Resolve
private resource paths from environment variables; write models, datasets, logs,
and run outputs outside Git. Record the exact command and configuration with
each run.

Everything here reads its inputs from the environment (`.env.example`); a script
that needs a path it did not receive fails at the first line with
`set: <VAR>` rather than silently defaulting.

| Group | Scripts | Stage |
|---|---|---|
| `setup/` | `env_tools.sh`, `env_sim.sh`, `env_policy.sh`, `env_data.sh`, `patch_gwp.py`, `patch_sapien.sh`, `install_adapter.sh` | 0 |
| `data/` | `probe_hdf5.py`, `collect_expert.sh`, `convert_lerobot.sh`, `check_lerobot.py`, `compute_norm.sh`, `compute_t5.py`, `norm_convention.py` | A–C |
| `infer/` | `gwp_server.sh`, `openloop_bench.py`, `make_wam_profile.py`, `norm_for_gguf.py`, `wam_convert.sh`, `wam_infer.sh` | D, F |
| `evaluate/` | `smoke_env.py`, `expert_replay.py`, `closed_loop.sh`, `make_demo_video.sh` | A, E |
| `checks/` | `no_private_paths.py`, `source_baseline.py` | any |
| `lib/` | `rt_args.py` | shared |

`train/` is intentionally empty: no fine-tuning has been run, and an empty
directory would imply otherwise.

## Ordering rules that are not obvious

- `collect_expert.sh` renders `configs/environments/robotwin_task_demo_clean_smoke.yml`
  into the RoboTwin tree; the shipped config carries `__GENERATED_DIR__` so a
  checkout cannot write into someone else's dataset directory.
- `closed_loop.sh` expects `infer/gwp_server.sh` to be running already, and it
  reconnects per episode on purpose (the server keeps an action-only prefix
  cache).
- `wam_convert.sh` and `wam_infer.sh` are separate because conversion is cheap
  and inference is not: the artifact is produced once and reused.
