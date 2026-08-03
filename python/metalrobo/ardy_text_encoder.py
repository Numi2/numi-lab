"""Prompt encoder for TREE Industries' reusable ARDY Llama 3 ONNX export."""

from __future__ import annotations

import hashlib
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


TEXT_ENCODER_REPOSITORY = "TREEIndustries/Llama-3-ARDY-Text-Encoder-ONNX"
TEXT_ENCODER_REVISION = "7aa52a05d54c2fd9177366aeb3f88e9e7f3c5766"
TEXT_ENCODER_HASHES = {
    "onnx/text_encoder_int4.onnx": (
        "c44f7ce04611a1a6f44ddf346ee89f4bc37356e45c0348e72ccd80bbf7b84e61"
    ),
    "onnx/text_encoder_int4.onnx.data": (
        "765e9ce4f9def78042dbd7e275500a1dd87f3f65f9dc8c518b54d163ae46fca3"
    ),
    "tokenizer/tokenizer.json": (
        "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7"
    ),
}
_SEQUENCE_LENGTH = 64
_PAD_ID = 128009
_PROMPT_PREFIX = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
)
_PROMPT_SUFFIX = "<|eot_id|>"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def inspect_text_encoder(directory: Path, verify_hashes: bool) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for relative, expected in TEXT_ENCODER_HASHES.items():
        path = directory / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path) if verify_hashes else expected
        if actual != expected:
            raise ValueError(f"ARDY text encoder sha256 mismatch: {relative}")
        artifacts[relative] = {
            "bytes": path.stat().st_size,
            "sha256": actual,
            "hash_verified": verify_hashes,
        }
    return {
        "repository": TEXT_ENCODER_REPOSITORY,
        "revision": TEXT_ENCODER_REVISION,
        "artifacts": artifacts,
    }


def tokenize_prompt(
    tokenizer_path: Path,
    prompt: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError("tokenizers is required for ARDY prompt encoding") from error
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    prefix = tokenizer.encode(_PROMPT_PREFIX, add_special_tokens=False).ids
    full = tokenizer.encode(
        _PROMPT_PREFIX + prompt + _PROMPT_SUFFIX,
        add_special_tokens=False,
    ).ids[:_SEQUENCE_LENGTH]
    if len(full) <= len(prefix):
        raise ValueError("ARDY prompt tokenization produced no embedding tokens")
    padding = _SEQUENCE_LENGTH - len(full)
    input_ids = np.asarray([_PAD_ID] * padding + full, dtype=np.int64)
    attention_mask = np.asarray(
        [0] * padding + [1] * len(full),
        dtype=np.int64,
    )
    embed_mask = np.asarray(
        [0] * (padding + len(prefix)) + [1] * (len(full) - len(prefix)),
        dtype=np.int64,
    )
    return (
        input_ids.reshape(1, _SEQUENCE_LENGTH),
        attention_mask.reshape(1, _SEQUENCE_LENGTH),
        embed_mask.reshape(1, _SEQUENCE_LENGTH),
    )


def encode_prompt(
    directory: Path,
    prompt: str,
    provider: str,
    verify_hashes: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("onnxruntime is required for ARDY text encoding") from error
    model = inspect_text_encoder(directory, verify_hashes)
    input_ids, attention_mask, embed_mask = tokenize_prompt(
        directory / "tokenizer" / "tokenizer.json",
        prompt,
    )
    from .ardy_onnx import _output_parity, _provider_attempts

    available = ort.get_available_providers()
    attempts = _provider_attempts(provider, available)

    graph = directory / "onnx" / "text_encoder_int4.onnx"
    feeds = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "embed_mask": embed_mask,
    }

    def run_session(
        providers: Sequence[str],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        load_started = time.perf_counter()
        session = ort.InferenceSession(str(graph), providers=list(providers))
        loaded = time.perf_counter()
        inference_started = time.perf_counter()
        value = session.run(["text_embedding"], feeds)[0]
        finished = time.perf_counter()
        result = np.asarray(value, dtype=np.float32).reshape(1, 1, 4096)
        if not np.isfinite(result).all():
            raise RuntimeError("ARDY text encoder produced non-finite output")
        return result, {
            "session_providers": session.get_providers(),
            "load_seconds": loaded - load_started,
            "inference_seconds": finished - inference_started,
        }

    failures: list[dict[str, Any]] = []
    embedding: np.ndarray | None = None
    selected_stage: dict[str, Any] | None = None
    cpu_cache: tuple[np.ndarray, dict[str, Any]] | None = None
    selected = ""
    started = time.perf_counter()
    for name, providers in attempts:
        parity = None
        try:
            if name == "cpu" and cpu_cache is not None:
                candidate, stage = cpu_cache
            else:
                candidate, stage = run_session(providers)
            if name == "coreml":
                cpu_cache = run_session(("CPUExecutionProvider",))
                parity = _output_parity(
                    (candidate,),
                    (cpu_cache[0],),
                    absolute_tolerance=1.0e-2,
                    relative_tolerance=5.0e-3,
                )
                if not parity["passed"]:
                    raise RuntimeError(
                        "CoreML text encoder failed CPU parity: "
                        f"max_abs={parity['maximum_absolute_error']:.6g}"
                    )
            embedding = candidate
            selected_stage = {**stage, "cpu_parity": parity}
            selected = name
            break
        except Exception as error:
            failure: dict[str, Any] = {"provider": name, "error": str(error)}
            if parity is not None:
                failure["cpu_parity"] = parity
            failures.append(failure)
            if provider != "auto":
                raise
    if embedding is None or selected_stage is None:
        raise RuntimeError("ARDY text encoder failed on every available provider")
    return embedding, {
        **model,
        "prompt": prompt,
        "selected_provider": selected,
        "provider_failures": failures,
        "available_providers": available,
        **selected_stage,
        "elapsed_seconds": time.perf_counter() - started,
        "token_count": int(np.sum(attention_mask)),
        "embedded_token_count": int(np.sum(embed_mask)),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "onnxruntime": ort.__version__,
        },
    }
