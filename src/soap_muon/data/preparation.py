"""Deterministic creation of fixed token shards from a streaming text corpus."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from soap_muon.config import DataConfig


class TextTokenizer(Protocol):
    eot_token: int

    def encode_ordinary(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class ShardTargets:
    train: int
    validation: int
    test: int

    @classmethod
    def from_data_config(cls, config: DataConfig) -> "ShardTargets":
        return cls(config.train_tokens, config.validation_tokens, config.test_tokens)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_token_shards(output_dir: str | Path) -> dict[str, object]:
    """Validate a completed token-shard manifest and return its contents."""
    destination = Path(output_dir)
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing shard manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Shard manifest must be a JSON object.")

    shards = manifest.get("shards")
    selection_manifest = manifest.get("selection_manifest")
    if not isinstance(shards, dict) or not isinstance(selection_manifest, dict):
        raise ValueError("Shard manifest is missing shard or selection metadata.")
    for split in ("train", "validation", "test"):
        shard = shards.get(split)
        if not isinstance(shard, dict):
            raise ValueError(f"Shard manifest is missing the {split!r} shard.")
        path = destination / str(shard.get("path", ""))
        expected_tokens = shard.get("tokens")
        expected_hash = shard.get("sha256")
        if not isinstance(expected_tokens, int) or expected_tokens <= 0 or not isinstance(expected_hash, str):
            raise ValueError(f"Shard manifest has invalid metadata for {split!r}.")
        tokens = np.load(path, mmap_mode="r")
        if tokens.shape != (expected_tokens,) or tokens.dtype != np.uint32:
            raise ValueError(f"{split} shard layout does not match its manifest.")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"{split} shard SHA-256 does not match its manifest.")

    selection_path = destination / str(selection_manifest.get("path", ""))
    selection_hash = selection_manifest.get("sha256")
    if not selection_path.is_file() or not isinstance(selection_hash, str):
        raise ValueError("Selection manifest metadata is invalid.")
    if sha256_file(selection_path) != selection_hash:
        raise ValueError("selection.jsonl SHA-256 does not match its manifest.")
    return manifest


def _write_json(path: Path, content: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _source_identifier(document: Mapping[str, object], fallback_ordinal: int) -> str:
    identifier = document.get("id")
    return str(fallback_ordinal if identifier is None else identifier)


def prepare_token_shards(
    documents: Iterable[Mapping[str, object]],
    tokenizer: TextTokenizer,
    output_dir: str | Path,
    targets: ShardTargets,
    *,
    dataset: str,
    revision: str,
    seed: int,
    tokenizer_name: str = "gpt2",
    sampling_description: str = "source stream order",
    resume: bool = False,
    progress: Callable[[int, dict[str, int]], None] | None = None,
) -> dict[str, object]:
    """Stream documents into exact, document-disjoint train/validation/test shards.

    A document that would exceed the remaining capacity of its assigned split is
    truncated *within that split* and never continued in another split. Thus a
    source document cannot cross a split boundary.
    """
    if min(targets.train, targets.validation, targets.test) <= 0:
        raise ValueError("Every split target must be positive.")
    destination = Path(output_dir)
    split_targets = asdict(targets)
    selection_path = destination / "selection.jsonl"
    preparation_path = destination / "preparation.json"
    preparation = {
        "dataset": dataset,
        "revision": revision,
        "sampling_description": sampling_description,
        "seed": seed,
        "targets": split_targets,
        "tokenizer": tokenizer_name,
    }
    if resume:
        if not destination.is_dir() or not selection_path.is_file() or not preparation_path.is_file():
            raise ValueError(
                "Cannot resume: token shards, selection.jsonl, or preparation.json are missing."
            )
        with preparation_path.open(encoding="utf-8") as handle:
            saved_preparation = json.load(handle)
        if saved_preparation != preparation:
            raise ValueError("Cannot resume: preparation settings do not match the existing shards.")
        positions = dict.fromkeys(split_targets, 0)
        with selection_path.open(encoding="utf-8") as selection:
            for line in selection:
                record = json.loads(line)
                split = record["split"]
                if split not in positions:
                    raise ValueError(f"Unknown split in selection log: {split!r}")
                positions[split] += int(record["tokens_written"])
        if any(positions[name] > count for name, count in split_targets.items()):
            raise ValueError(f"Selection log exceeds configured targets: {positions}")
        arrays = {
            name: np.lib.format.open_memmap(destination / f"{name}.npy", mode="r+")
            for name in split_targets
        }
        if any(arrays[name].shape != (count,) or arrays[name].dtype != np.uint32 for name, count in split_targets.items()):
            raise ValueError("Existing shard layout does not match the requested targets.")
        selection_mode = "a"
    else:
        destination.mkdir(parents=True, exist_ok=False)
        _write_json(preparation_path, preparation)
        arrays = {
            name: np.lib.format.open_memmap(destination / f"{name}.npy", mode="w+", dtype=np.uint32, shape=(count,))
            for name, count in split_targets.items()
        }
        positions = dict.fromkeys(split_targets, 0)
        selection_mode = "w"

    split_names = iter(split_targets)
    active_split = next(
        (name for name in split_names if positions[name] < split_targets[name]), None
    )
    if active_split is None:
        documents = ()
    with selection_path.open(selection_mode, encoding="utf-8") as selection:
        for ordinal, document in enumerate(documents):
            text = document.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            token_ids = tokenizer.encode_ordinary(text) + [tokenizer.eot_token]
            remaining = split_targets[active_split] - positions[active_split]
            taken = min(len(token_ids), remaining)
            arrays[active_split][positions[active_split] : positions[active_split] + taken] = token_ids[:taken]
            positions[active_split] += taken
            selection.write(
                json.dumps(
                    {
                        "split": active_split,
                        "source_id": _source_identifier(document, ordinal),
                        "source_ordinal": int(document.get("_source_ordinal", ordinal)),
                        "source_tokens": len(token_ids),
                        "tokens_written": taken,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            if progress is not None and ordinal > 0 and ordinal % 1_000 == 0:
                progress(ordinal, dict(positions))
            if positions[active_split] == split_targets[active_split]:
                try:
                    active_split = next(split_names)
                except StopIteration:
                    break
    del arrays  # Flush memory-mapped arrays before hashing.
    if positions != split_targets:
        raise RuntimeError(f"Source stream ended early: wrote {positions}, expected {split_targets}.")

    shards = {
        name: {"path": f"{name}.npy", "tokens": count, "sha256": sha256_file(destination / f"{name}.npy")}
        for name, count in split_targets.items()
    }
    manifest: dict[str, object] = {
        "dataset": dataset,
        "revision": revision,
        "seed": seed,
        "tokenizer": tokenizer_name,
        "split_policy": f"{sampling_description}; document-disjoint train, validation, test",
        "shards": shards,
        "selection_manifest": {"path": "selection.jsonl", "sha256": sha256_file(selection_path)},
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def stream_fineweb_edu(
    config: DataConfig,
    seed: int,
    selection_modulus: int,
    skip_through_ordinal: int | None = None,
) -> Iterable[Mapping[str, object]]:
    """Yield a deterministic hash-selected FineWeb-Edu stream at a pinned revision."""
    if not config.revision:
        raise ValueError("FineWeb-Edu preparation requires a pinned dataset revision.")
    if selection_modulus <= 0:
        raise ValueError("selection_modulus must be positive.")
    if skip_through_ordinal is not None and skip_through_ordinal < 0:
        raise ValueError("skip_through_ordinal must be non-negative when supplied.")
    from datasets import load_dataset

    stream = load_dataset(config.dataset, split="train", streaming=True, revision=config.revision)

    def selected() -> Iterable[Mapping[str, object]]:
        for ordinal, document in enumerate(stream):
            if skip_through_ordinal is not None and ordinal <= skip_through_ordinal:
                continue
            identifier = _source_identifier(document, ordinal)
            digest = hashlib.sha256(f"{seed}:{identifier}".encode()).digest()
            if int.from_bytes(digest[:8], "big") % selection_modulus == 0:
                yield {**document, "_source_ordinal": ordinal}

    return selected()
