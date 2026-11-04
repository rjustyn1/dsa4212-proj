import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from soap_muon.config import DataConfig
from soap_muon.data.preparation import (
    ShardTargets,
    prepare_token_shards,
    stream_fineweb_edu,
    verify_token_shards,
)


class CharacterTokenizer:
    eot_token = 99

    def encode_ordinary(self, text: str) -> list[int]:
        return [ord(character) for character in text]


def test_preparation_creates_exact_disjoint_token_shards(tmp_path) -> None:
    documents = [
        {"id": "one", "text": "abc"},
        {"id": "two", "text": "defg"},
        {"id": "three", "text": "hijkl"},
        {"id": "four", "text": "mnop"},
    ]
    manifest = prepare_token_shards(
        documents,
        CharacterTokenizer(),
        tmp_path / "shards",
        ShardTargets(train=5, validation=4, test=3),
        dataset="example",
        revision="abc123",
        seed=7,
    )

    train = np.load(tmp_path / "shards/train.npy")
    validation = np.load(tmp_path / "shards/validation.npy")
    test = np.load(tmp_path / "shards/test.npy")
    assert train.tolist() == [97, 98, 99, 99, 100]
    assert validation.tolist() == [104, 105, 106, 107]
    assert test.tolist() == [109, 110, 111]
    assert manifest["shards"]["train"]["tokens"] == 5
    assert verify_token_shards(tmp_path / "shards") == manifest

    records = [json.loads(line) for line in (tmp_path / "shards/selection.jsonl").read_text().splitlines()]
    split_by_id = {record["source_id"]: record["split"] for record in records}
    assert split_by_id == {
        "one": "train",
        "two": "train",
        "three": "validation",
        "four": "test",
    }


def test_preparation_resume_requires_matching_provenance(tmp_path) -> None:
    targets = ShardTargets(train=5, validation=4, test=3)
    output_dir = tmp_path / "shards"
    with pytest.raises(RuntimeError, match="Source stream ended early"):
        prepare_token_shards(
            [{"id": "one", "text": "abc"}],
            CharacterTokenizer(),
            output_dir,
            targets,
            dataset="example",
            revision="abc123",
            seed=7,
        )

    with pytest.raises(ValueError, match="settings do not match"):
        prepare_token_shards(
            [],
            CharacterTokenizer(),
            output_dir,
            targets,
            dataset="example",
            revision="abc123",
            seed=8,
            resume=True,
        )


def test_preparation_resume_records_original_source_ordinal(tmp_path) -> None:
    targets = ShardTargets(train=5, validation=4, test=3)
    output_dir = tmp_path / "shards"
    with pytest.raises(RuntimeError, match="Source stream ended early"):
        prepare_token_shards(
            [
                {"id": "one", "text": "abc", "_source_ordinal": 10},
                {"id": "two", "text": "d", "_source_ordinal": 11},
            ],
            CharacterTokenizer(),
            output_dir,
            targets,
            dataset="example",
            revision="abc123",
            seed=7,
        )

    prepare_token_shards(
        [
            {"id": "three", "text": "efg", "_source_ordinal": 12},
            {"id": "four", "text": "hij", "_source_ordinal": 13},
        ],
        CharacterTokenizer(),
        output_dir,
        targets,
        dataset="example",
        revision="abc123",
        seed=7,
        resume=True,
    )

    records = [json.loads(line) for line in (output_dir / "selection.jsonl").read_text().splitlines()]
    assert [record["source_ordinal"] for record in records] == [10, 11, 12, 13]


def test_stream_resume_skips_through_source_ordinal(monkeypatch) -> None:
    source = [
        {"text": "a"},
        {"id": "two", "text": "b"},
        {"id": "three", "text": "c"},
    ]
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *_, **__: iter(source)),
    )
    config = DataConfig(
        dataset="example",
        tokenizer="gpt2",
        train_tokens=5,
        validation_tokens=4,
        test_tokens=3,
        sequence_length=1,
        revision="abc123",
    )

    remaining = list(stream_fineweb_edu(config, seed=7, selection_modulus=1, skip_through_ordinal=1))

    assert [document["id"] for document in remaining] == ["three"]
    assert [document["_source_ordinal"] for document in remaining] == [2]


def test_verification_rejects_modified_shard(tmp_path) -> None:
    output_dir = tmp_path / "shards"
    prepare_token_shards(
        [
            {"id": "one", "text": "abc"},
            {"id": "two", "text": "defg"},
            {"id": "three", "text": "hijkl"},
            {"id": "four", "text": "mnop"},
        ],
        CharacterTokenizer(),
        output_dir,
        ShardTargets(train=5, validation=4, test=3),
        dataset="example",
        revision="abc123",
        seed=7,
    )
    train = np.lib.format.open_memmap(output_dir / "train.npy", mode="r+")
    train[0] = 0
    del train

    with pytest.raises(ValueError, match="SHA-256"):
        verify_token_shards(output_dir)
