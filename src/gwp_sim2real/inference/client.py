"""Thin client for the GWP 0.5 PyTorch inference server.

The server side is `scripts/inference_openloop.py --server` from the GWP
checkout: a ZMQ REP socket that accepts a dict of tensors and answers with an
action chunk. We speak the same protocol the official open-loop client speaks,
so there is exactly one wire format in play.

Nothing here loads model weights. The process that owns the GPU is separate and
usually in a different Python environment; this module only exists so that the
simulation environment does not need `diffusers`, `transformers`, or a second
copy of giga-models.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from gwp_sim2real import contracts


class GwpInferenceClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11444,
        timeout_ms: int = 120_000,
        giga_models_dir: str | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self.giga_models_dir = giga_models_dir or os.environ.get("GIGA_MODELS_DIR")
        self._client = None

    # ------------------------------------------------------------------ setup
    def _connect(self):
        if self._client is not None:
            return self._client
        try:
            from giga_models.sockets import RobotInferenceClient
        except ModuleNotFoundError:
            if not self.giga_models_dir:
                raise
            if self.giga_models_dir not in sys.path:
                sys.path.insert(0, self.giga_models_dir)
            from giga_models.sockets import RobotInferenceClient

        self._client = RobotInferenceClient(
            host=self.host, port=self.port, timeout_ms=self.timeout_ms
        )
        return self._client

    def ping(self) -> bool:
        return bool(self._connect().ping())

    def close(self) -> None:
        """Drop the socket.

        The server keeps an action-only prefix cache, so a new episode must not
        inherit the previous episode's history. Reconnecting is the cheapest
        reliable reset; the REP socket is stateless.
        """
        self._client = None

    # -------------------------------------------------------------- inference
    def infer(
        self,
        images: dict[str, np.ndarray],
        state,
        *,
        prompt: str | None = None,
        prompt_embedding=None,
        chunk_idx: int = 0,
    ) -> np.ndarray:
        """Run one inference call and return a validated `[T, 14]` chunk.

        `images` maps camera name -> HxWx3 uint8 (or CHW float in [0, 1]); the
        server expects CHW float64 in [0, 1], which is what the official client
        sends, so the conversion happens here once.
        """
        import torch

        contracts.assert_images(images)
        payload: dict[str, object] = {
            "observation.state": torch.from_numpy(contracts.assert_state(state)),
            "chunk_idx": int(chunk_idx),
            "_quiet": True,
            "_skip_ref_image_save": True,
        }
        for cam in contracts.CAMERA_KEYS:
            key = f"observation.images.{contracts.LEROBOT_CAMERA_KEYS[cam]}"
            payload[key] = torch.from_numpy(_to_chw_float01(images[cam]))

        if prompt_embedding is not None:
            emb = np.asarray(prompt_embedding, dtype=np.float32)
            if emb.ndim == 2:
                emb = emb[None, ...]
            payload["prompt_embedding"] = torch.from_numpy(emb)
        elif prompt is not None:
            payload["prompt"] = prompt
        else:
            raise ValueError("either prompt or prompt_embedding is required")

        raw = self._connect().inference(payload)
        return contracts.assert_action_chunk(raw)


def _to_chw_float01(arr) -> np.ndarray:
    """Match the preprocessing the official open-loop client performs."""
    a = np.asarray(arr)
    if a.ndim != 3:
        raise contracts.ContractError(f"expected a 3-D image, got {a.shape}")
    if a.shape[0] not in (1, 3) and a.shape[-1] in (1, 3):
        a = np.transpose(a, (2, 0, 1))
    if a.shape[0] == 1:
        a = np.repeat(a, 3, axis=0)
    if np.issubdtype(a.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(a)) <= 1.5 else 1.0
        a = np.clip(a * scale, 0, 255)
    return a.astype(np.float64) / 255.0
