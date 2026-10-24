#!/usr/bin/env python
"""Apply the GigaWorld-Policy 0.5 inference patches to the workspace source copy.

Why this exists (all four items were verified against the shipped code):

1. ``get_policy`` ends with ``pipe.to(device)``. WAPipeline registers the UMT5
   text encoder as a component, so that call drags the 11.36 GB BF16 encoder
   onto the GPU on top of the ~12 GB MoT transformer and the VAE: ~23.1 GiB of
   static weights against 17.9-18.9 GiB free on the shared 24 GiB 4090s.
   Fixed-embedding inference never executes it (``inference()`` passes
   ``prompt_embeds`` and ``guidance_scale=0.0`` disables the unconditional
   branch), so the patch places components explicitly and leaves the text
   encoder on CPU.
2. ``get_output_dirs_from_checkpoint`` derives the open-loop result directory
   from the checkpoint path, which would write reference images and plots into
   the shared read-only model tree. The patch adds ``--output-dir``.
3. ``inference()`` reports only a single "model inference time" print. The patch
   appends one JSON record per call (entry/preprocess/model/post/total plus
   GPU allocated/reserved and peak) to ``--timing-jsonl``.
4. The patch records the resolved device/dtype of every pipeline component so
   the placement is auditable instead of assumed.

Idempotent: it always starts from the pristine copy stored under
``patch/orig/`` and rewrites the target file from scratch.

Usage:
  python scripts/setup/patch_gwp.py --write
  python scripts/setup/patch_gwp.py --check
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "lib"))

import env as project_env  # noqa: E402

project_env.load_env()

# The upstream checkout is an input, never a subdirectory of this repository.
# A full copy of an upstream source file may not be redistributed here, so the
# pristine text is kept outside Git (under $LOCKS_DIR) and identified by hash.
SRC = os.path.join(
    project_env.require("GWP_SOURCE_DIR")["GWP_SOURCE_DIR"],
    "scripts",
    "inference_openloop.py",
)
PRISTINE_DIR = os.path.join(
    os.environ.get("LOCKS_DIR") or os.path.join(REPO_ROOT, "locks"), "upstream"
)
PRISTINE = os.path.join(PRISTINE_DIR, "inference_openloop.py")
DIFF = os.path.join(REPO_ROOT, "patch", "gwp05_inference_openloop.diff")

# sha256 of the upstream revision this patch was written against. Recorded in
# patch/upstream-manifest.json as well; duplicated here so a mismatch can be
# reported without reading another file.
EXPECTED_SHA256 = "750625547d2165142828a272fefc50163ea9522dd98be6f4648a26d6dcf3e268"

EDITS: list[tuple[str, str, str]] = []


def edit(name: str, old: str, new: str) -> None:
    EDITS.append((name, old, new))


# ---------------------------------------------------------------- 1. signature
edit(
    "get_policy-signature",
    '    compile_fullgraph=False,\n    compile_scope="action-blocks",\n):\n',
    '    compile_fullgraph=False,\n    compile_scope="action-blocks",\n'
    '    text_encoder_device="cpu",\n    output_dir=None,\n    timing_jsonl=None,\n):\n',
)

# ------------------------------------------------------- 2. output dir override
edit(
    "output-dir-override",
    '    _, ref_image_save_dir = get_output_dirs_from_checkpoint(checkpoint)\n'
    '    os.makedirs(ref_image_save_dir, exist_ok=True)\n'
    '    print(f"Saving ref images to: {ref_image_save_dir}")\n',
    '    _, ref_image_save_dir = get_output_dirs_from_checkpoint(checkpoint)\n'
    '    if output_dir:\n'
    '        # Keep run artefacts inside this project, never in the shared model tree.\n'
    '        ref_image_save_dir = os.path.join(output_dir, "ref_images")\n'
    '    os.makedirs(ref_image_save_dir, exist_ok=True)\n'
    '    print(f"Saving ref images to: {ref_image_save_dir}")\n',
)

# ------------------------------------------- 3. explicit component placement
edit(
    "component-placement",
    '    pipe = WAPipeline.from_pretrained(base_model, vae=vae, transformer=transformer, scheduler=scheduler, torch_dtype=torch_dtype)\n'
    '    pipe.to(device=device, dtype=torch_dtype)\n'
    '    pipe._model_dtype_name = str(torch_dtype)\n',
    '    pipe = WAPipeline.from_pretrained(base_model, vae=vae, transformer=transformer, scheduler=scheduler, torch_dtype=torch_dtype)\n'
    '\n'
    '    # ---- explicit component placement (patched) --------------------------\n'
    '    # NOTE: the stock `pipe.to(device)` also moved the UMT5 text encoder onto\n'
    '    # the GPU. Fixed-embedding inference never executes it, and 11.36 GB of\n'
    '    # idle weights do not fit next to the MoT transformer + VAE on a 24 GiB\n'
    '    # card, so only the modules that actually run are moved.\n'
    '    _te_device = (text_encoder_device or "cpu").strip()\n'
    '    if _te_device == "auto":\n'
    '        _te_device = device\n'
    '    _te = getattr(pipe, "text_encoder", None)\n'
    '    if _te is not None:\n'
    '        _te.to(_te_device)\n'
    '    for _name in ("vae", "transformer", "transformer_2"):\n'
    '        _mod = getattr(pipe, _name, None)\n'
    '        if _mod is not None:\n'
    '            _mod.to(device=device, dtype=torch_dtype)\n'
    '    _device_report = {}\n'
    '    for _name in ("vae", "text_encoder", "transformer", "transformer_2", "image_encoder"):\n'
    '        _mod = getattr(pipe, _name, None)\n'
    '        if _mod is None:\n'
    '            _device_report[_name] = None\n'
    '            continue\n'
    '        try:\n'
    '            _p = next(_mod.parameters())\n'
    '            _device_report[_name] = {"device": str(_p.device), "dtype": str(_p.dtype)}\n'
    '        except StopIteration:\n'
    '            _device_report[_name] = {"device": "no-params", "dtype": None}\n'
    '    print("Component placement: " + json.dumps(_device_report, sort_keys=True), flush=True)\n'
    '    if str(device).startswith("cuda") and torch.cuda.is_available():\n'
    '        print(\n'
    '            "GPU after load: allocated=%.2f GB reserved=%.2f GB"\n'
    '            % (torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9),\n'
    '            flush=True,\n'
    '        )\n'
    '    if output_dir:\n'
    '        os.makedirs(output_dir, exist_ok=True)\n'
    '        with open(os.path.join(output_dir, "device_placement.json"), "w", encoding="utf-8") as _f:\n'
    '            json.dump(_device_report, _f, indent=2)\n'
    '    pipe._device_report = _device_report\n'
    '    pipe._model_dtype_name = str(torch_dtype)\n',
)

# --------------------------------------------------- 4. per-call timing records
edit(
    "inference-entry-timer",
    '    def inference(self, data):\n        images = {\n',
    '    def inference(self, data):\n        _t_entry = time.perf_counter()\n        images = {\n',
)

edit(
    "inference-preprocess-timer",
    '        model_start_time = time.perf_counter()\n        pred_action = pipe(\n',
    '        model_start_time = time.perf_counter()\n        _preprocess_s = model_start_time - _t_entry\n        pred_action = pipe(\n',
)

edit(
    "inference-timing-record",
    '        pred_action = pred_action[0] + state.repeat(action_chunk, 1).cpu().numpy() * mask\n        return pred_action\n',
    '        pred_action = pred_action[0] + state.repeat(action_chunk, 1).cpu().numpy() * mask\n'
    '        _t_exit = time.perf_counter()\n'
    '        if timing_jsonl:\n'
    '            _rec = {\n'
    '                "ts": time.time(),\n'
    '                "chunk_idx": int(data.get("chunk_idx", -1)),\n'
    '                "episode_idx": int(data.get("episode_idx", -1)),\n'
    '                "preprocess_s": _preprocess_s,\n'
    '                "model_s": self._last_model_inference_s,\n'
    '                "post_s": _t_exit - model_end_time,\n'
    '                "total_s": _t_exit - _t_entry,\n'
    '            }\n'
    '            if str(device).startswith("cuda") and torch.cuda.is_available():\n'
    '                _rec["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1e9\n'
    '                _rec["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1e9\n'
    '                _rec["gpu_peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9\n'
    '                _rec["gpu_peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9\n'
    '            with open(timing_jsonl, "a", encoding="utf-8") as _f:\n'
    '                _f.write(json.dumps(_rec) + "\\n")\n'
    '        return pred_action\n',
)

# ------------------------------------------------------------- 5. CLI plumbing
edit(
    "argparse-new-flags",
    '    parser.add_argument("--server", action="store_true", help="Run inference server instead of open-loop client.")\n',
    '    parser.add_argument(\n'
    '        "--text-encoder-device",\n'
    '        default="cpu",\n'
    '        help="Device for the UMT5 text encoder. \\"cpu\\" keeps 11.36 GB off the GPU when a "\n'
    '        "fixed T5 embedding is used; \\"auto\\" restores the stock all-on-GPU behaviour.",\n'
    '    )\n'
    '    parser.add_argument("--output-dir", default=None, help="Directory for run artefacts (ref images, placement report).")\n'
    '    parser.add_argument("--timing-jsonl", default=None, help="Append one JSON timing record per inference call.")\n'
    '    parser.add_argument("--server", action="store_true", help="Run inference server instead of open-loop client.")\n',
)

edit(
    "main-passes-flags",
    '        compile_fullgraph=args.compile_fullgraph,\n        compile_scope=args.compile_scope,\n    )\n',
    '        compile_fullgraph=args.compile_fullgraph,\n'
    '        compile_scope=args.compile_scope,\n'
    '        text_encoder_device=args.text_encoder_device,\n'
    '        output_dir=args.output_dir,\n'
    '        timing_jsonl=args.timing_jsonl,\n'
    '    )\n',
)

# ------------------------------------- 6. client: keep artefacts out of models/
edit(
    "client-signature",
    'def inference_client(\n    checkpoint=DEFAULT_CHECKPOINT,\n    data_paths=None,\n    data_idx=DEFAULT_DATA_IDX,\n    host=DEFAULT_HOST,\n    port=DEFAULT_PORT,\n    replan_steps=None,\n):\n',
    'def inference_client(\n    checkpoint=DEFAULT_CHECKPOINT,\n    data_paths=None,\n    data_idx=DEFAULT_DATA_IDX,\n    host=DEFAULT_HOST,\n    port=DEFAULT_PORT,\n    replan_steps=None,\n    output_dir=None,\n):\n',
)

edit(
    "client-save-dir",
    '    save_dir, _ = get_output_dirs_from_checkpoint(checkpoint)\n\n    action_chunk = 48\n',
    '    save_dir, _ = get_output_dirs_from_checkpoint(checkpoint)\n'
    '    if output_dir:\n'
    '        # Default derives from the checkpoint path, i.e. inside the shared model tree.\n'
    '        save_dir = os.path.join(output_dir, "open_loop")\n'
    '\n'
    '    action_chunk = 48\n',
)

edit(
    "client-call-from-test-policy",
    '        host=args.host,\n        port=args.port,\n        replan_steps=args.replan_steps,\n    )\n',
    '        host=args.host,\n        port=args.port,\n        replan_steps=args.replan_steps,\n'
    '        output_dir=args.output_dir,\n    )\n',
)

# ------------------------------- 7. pin the pipeline execution device explicitly
# DiffusionPipeline.device returns the device of the FIRST nn.Module in signature
# order, which is now the text encoder parked on CPU. Everything that asks the
# pipeline for a device (image preprocessing, prompt_embeds, state) would then
# build CPU tensors and the CUDA VAE would raise
#   RuntimeError: Input type (CPUBFloat16Type) and weight type (CUDABFloat16Type)
# Observed exactly once before this patch; see runs/D_load_smoke.
edit(
    "execution-device-pin",
    '    pipe._device_report = _device_report\n    pipe._model_dtype_name = str(torch_dtype)\n',
    '    pipe._device_report = _device_report\n'
    '    pipe._model_dtype_name = str(torch_dtype)\n'
    '\n'
    '    # Width of the model action/state block, from the checkpoint itself.\n'
    '    # prepare_latents/inference hardcode 32, but Giga-World-Policy-0.5 declares\n'
    '    # in_action_channels=16 and its state_encoder/action_encoder in_proj weights\n'
    '    # are [num_embodiments, 16, 128]: feeding 32 raises a bmm shape error.\n'
    '    _action_width = int(getattr(pipe.transformer.config, "in_action_channels", 32))\n'
    '    print(f"Model action/state width: {_action_width}", flush=True)\n'
    '\n'
    '    _compute_device = torch.device(device)\n'
    '    _base_cls = type(pipe)\n'
    '\n'
    '    class _PlacedPipeline(_base_cls):\n'
    '        """Same pipeline, execution device pinned to the compute device.\n'
    '\n'
    '        DiffusionPipeline.device resolves from the first nn.Module in the\n'
    '        signature, which is the CPU-resident text encoder after the placement\n'
    '        above; without this the VAE would be fed CPU tensors.\n'
    '        """\n'
    '\n'
    '        @property\n'
    '        def device(self):\n'
    '            return _compute_device\n'
    '\n'
    '        @property\n'
    '        def _execution_device(self):\n'
    '            return _compute_device\n'
    '\n'
    '    pipe.__class__ = _PlacedPipeline\n'
    '    print(f"Execution device pinned to {_compute_device}", flush=True)\n',
)

# ---------------------------------- 8. use the model width instead of a literal
edit(
    "norm-state-pad-width",
    '        norm_state = torch_F.pad(norm_state, (0, 32 - norm_state.shape[-1]), value=0.0)\n',
    '        norm_state = torch_F.pad(norm_state, (0, _action_width - norm_state.shape[-1]), value=0.0)\n',
)

edit(
    "pipe-action-dim-width",
    '            action_dim=32,\n',
    '            action_dim=_action_width,\n',
)


def _state(text: str) -> str:
    """Classify a file as `pristine`, `patched`, or `unknown`."""
    old_ok = all(text.count(old) == 1 and new not in text for _, old, new in EDITS)
    new_ok = all(text.count(new) == 1 and old not in text for _, old, new in EDITS)
    if old_ok:
        return "pristine"
    if new_ok:
        return "patched"
    return "unknown"


def _sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _report_mismatch() -> None:
    if not os.path.isfile(SRC):
        return
    got = _sha256(SRC)
    if got != EXPECTED_SHA256:
        print(f"note: sha256={got}")
        print(f"      expected {EXPECTED_SHA256} (the revision this patch was written against)")
        print("      anchors will decide whether the patch still applies")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report whether anchors still match")
    ap.add_argument("--write", action="store_true", help="apply the patch in place")
    ap.add_argument("--restore", action="store_true", help="restore the pristine upstream file")
    ap.add_argument("--diff", action="store_true", help="regenerate patch/gwp05_inference_openloop.diff")
    args = ap.parse_args()

    if not os.path.isfile(SRC):
        print(f"missing upstream file: {SRC}")
        print("set GWP_SOURCE_DIR to the GigaWorld-Policy checkout")
        return 2

    with open(SRC, "r", encoding="utf-8") as f:
        current = f.read()
    state = _state(current)

    if args.check:
        print(f"upstream={SRC}")
        print(f"state={state}")
        if state == "unknown":
            _report_mismatch()
            missing = [n for n, old, _ in EDITS if old not in current and _ not in current]
            print(f"anchors_not_found={missing or 'partial'}")
            return 2
        return 0

    if args.restore:
        if not os.path.isfile(PRISTINE):
            print(f"no pristine copy at {PRISTINE}; cannot restore")
            return 2
        shutil.copy2(PRISTINE, SRC)
        print(f"restored {SRC} from {PRISTINE}")
        return 0

    if args.write:
        if state == "patched":
            print("already patched; nothing to do")
            return 0
        if state != "pristine":
            print("refusing to write: upstream text matches neither the pristine nor the patched form")
            _report_mismatch()
            return 2

        os.makedirs(PRISTINE_DIR, exist_ok=True)
        if not os.path.isfile(PRISTINE):
            shutil.copy2(SRC, PRISTINE)
            print(f"kept pristine copy outside Git: {PRISTINE}")

        patched = current
        for name, old, new in EDITS:
            patched = patched.replace(old, new)
        with open(SRC, "w", encoding="utf-8") as f:
            f.write(patched)
        print(f"patched {SRC} ({len(EDITS)} edits)")
        state = "patched"
        current = patched

    if args.diff or args.write:
        if not os.path.isfile(PRISTINE):
            print("skip diff: pristine copy unavailable (it lives outside Git by design)")
            return 0
        with open(PRISTINE, "r", encoding="utf-8") as f:
            pristine = f.read()
        diff = "".join(
            difflib.unified_diff(
                pristine.splitlines(keepends=True),
                current.splitlines(keepends=True),
                fromfile="a/scripts/inference_openloop.py",
                tofile="b/scripts/inference_openloop.py",
            )
        )
        os.makedirs(os.path.dirname(DIFF), exist_ok=True)
        with open(DIFF, "w", encoding="utf-8") as f:
            f.write(diff)
        print(f"diff -> {DIFF} ({len(diff.splitlines())} lines)")

    if not (args.write or args.check or args.restore or args.diff):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
