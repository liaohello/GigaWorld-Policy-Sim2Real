# Upstream repositories

Keep third-party checkouts outside the tracked repository (for example, in a
user-selected `external/` directory ignored by Git). Do not copy full upstream
projects into this repository. Record source URL, pinned commit/tag, license,
install command, and local patch hashes in
`docs/experiments/upstream-sources.md`.

Required sources:

| Project | Upstream | Used for |
|---|---|---|
| GigaWorld-Policy | `https://github.com/open-gigaai/giga-world-policy` | policy runtime, checkpoint, inference server, normalization script |
| RoboTwin 2.0 | `https://github.com/RoboTwin-Platform/RoboTwin` | simulator, native Aloha-AgileX task, XPolicyLab policy interface |
| wam.cpp | `https://github.com/GigaAI-research/wam.cpp` | GGUF conversion, C++ runtime, PyTorch↔C++ parity |

Two of the three were used without Git metadata (source archives). Their
identity is pinned by content hash in `docs/experiments/upstream-sources.md`,
which is the only way to tell an edited checkout from an original one.

Not vendored, but pulled in at build time: `llama.cpp` tag `b9866`, fetched by
wam.cpp's CMake from a locally verified tarball.
