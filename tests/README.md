# Validation checks

Focused contract checks that run without a GPU, a simulator, or upstream model
weights. They guard the parts of the pipeline where a silent change would still
"work" while degrading every policy input.

| Test | Guards |
|---|---|
| `test_contracts.py` | 14-D state/action shape, horizon, finiteness, camera names, segment order |
| `test_norm_convention.py` | the min-max ⇄ z-score identity that reconciles the PyTorch and wam.cpp paths |
| `test_image_preprocess.py` | RGB channel order, HWC→CHW, scale detection, grayscale replication |

```bash
pip install -e ".[dev]"
pytest -q          # 16 tests, < 1 s
```

These tests do **not** claim simulator or model success. Anything that requires
the real integration path lives in `docs/experiments/` as a recorded run with an
artifact, not as a unit test.
