# Upstream source pins

Upstream trees live outside this repository (see `external/README.md`). This
file records how each tree was identified, its license status, and every local
patch applied. Two of the three trees ship without Git metadata, so identity is
pinned by content hash instead.

| Project | Upstream | Revision | License | Local patches |
|---|---|---|---|---|
| GigaWorld-Policy 0.5 | `https://github.com/open-gigaai/giga-world-policy` (the Hugging Face model card links to it) | **no Git metadata** in the local snapshot: identified by content hash; the pristine inference file hash matches no revision in the public Git history checked 2026-10-24 | **not verified for this code snapshot** — the official code repository has no license file; the model card's Apache-2.0 label covers model weights and does not establish a license for this separate source tree | `patch/gwp05_inference_openloop.diff` (206 lines) |
| RoboTwin 2.0 | `https://github.com/RoboTwin-Platform/RoboTwin` (per README) | **no Git metadata**: archive CRC checked during the prior download audit | MIT (`LICENSE`, Copyright (c) 2025 Tianxing Chen) | none in the upstream tree; one added task config and the GWP adapter |
| wam.cpp | `https://github.com/GigaAI-research/wam.cpp` | `e189107a5c9424781dea58c0481355350a051385` (2026-07-30) | Apache-2.0 | none; build configuration only |
| llama.cpp | `https://github.com/ggml-org/llama.cpp` | tag `b9866` | MIT (upstream) | none; vendored by wam.cpp's CMake, `WAM_LLAMA_ARCHIVE` points at the locally verified tarball |
| giga-models | bundled in the GWP snapshot under `third_party/giga-models` | follows the GWP snapshot | **not verified** | none (imported read-only, never installed) |
| SAPIEN | `https://github.com/haosulab/SAPIEN` | tag `3.0.0b1`, commit `d9031a351c39986e51bd0716bc3b51f9a05070d8`; `urdf_loader.py` matches the installed original SHA-256 | MIT (`LICENSE` at this exact tag) | `patch/sapien_urdf_loader.diff` |

## Identity of the Git-less trees

| Tree | Evidence used | Value |
|---|---|---|
| GWP snapshot | `sha256(README.md)` | `9a9ab36d74b689a6db697ba27e4cf4d02b91a43dbd2e7f81d975891c2b08ab3f` |
| GWP snapshot | `sha256(requirements.txt)` | `ae2501b6f9caf5baecd04584935925c4dc9994356ec06a62b77aad3d88dd9e29` |
| GWP checkpoint | `sha256(config.json)` | `8ad8cadca4f41dbbe7c117b5b81efb1d082665727a118f8746508f54ed507a4f` |
| GWP checkpoint | three safetensors shards | 9 995 462 960 + 9 963 659 592 + 4 127 312 888 B = 24 086 435 440 B |
| RoboTwin | archive CRC during the prior download audit | pass (documented; the archive is no longer on disk) |
| llama.cpp tarball | `sha256`, compared with the value pinned in wam.cpp's `cmake/Dependencies.cmake` | `fb4f9e480938406dd17a27e9a3b4b99fb558685d8ad44040778062c5e999d8ae` |

## License status — open item

The GWP snapshot has no `LICENSE` file and no license statement in its README or
its report PDF. Its pristine inference-file hash does not match any revision in
the public source history checked on 2026-10-24. The Hugging Face model card
labels the model weights Apache-2.0, but that label does not grant a license for
the code snapshot or its derived patch.

Until it is checked against that revision, this project:

- does not redistribute GWP source files, weights, or derived artifacts. An
  earlier revision of this repository did commit two pristine upstream files
  (`patch/orig/`) while making that claim; they are now removed, and each patch
  refers to its original by SHA-256 (`patch/upstream-manifest.json`);
- keeps the GWP patch diff only pending confirmation that its source is licensed
  for redistribution; a patch may itself be a derivative work;
- marks the license as **not verified** rather than guessing one;
- keeps every derived GGUF and checkpoint outside Git.

The GWP patch must not be redistributed until its applicable license is
confirmed. The SAPIEN patch targets MIT-licensed source at the exact pinned tag.

## Patch inventory

| Patch | Target | Lines | Reason |
|---|---|---:|---|
| `patch/gwp05_inference_openloop.diff` | `scripts/inference_openloop.py` | 206 | Component-level device placement, pinned execution device, action width read from the checkpoint, output-directory plumbing, per-call timing |
| `patch/sapien_urdf_loader.diff` | `sapien/wrapper/urdf_loader.py` (installed package, not an upstream repo) | small | Open SRDF as UTF-8, matching RoboTwin's own `_install.sh` |

The mplib `plan_screw` patch used by RoboTwin's installer is deliberately **not**
applied: it weakens the collision check, and the Aloha-AgileX embodiment uses
`planner: curobo`, so the patched function is never reached.
