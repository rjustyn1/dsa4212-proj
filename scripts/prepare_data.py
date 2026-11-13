"""Create frozen FineWeb-Edu train/validation/test token shards.

Example:
    uv run python scripts/prepare_data.py \
      --config configs/data/fineweb_edu_75m.yaml \
      --output-dir data/processed/fineweb_edu_75m_gpt2 \
      --seed 2026
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tiktoken

from soap_muon.config import DataConfig, load_yaml
from soap_muon.data.preparation import ShardTargets, prepare_token_shards, stream_fineweb_edu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted preparation run using its selection log.",
    )
    parser.add_argument(
        "--selection-modulus",
        type=int,
        default=4,
        help="Keep documents whose seeded SHA-256 bucket is zero modulo this value.",
    )
    args = parser.parse_args()

    config = DataConfig(**load_yaml(args.config))
    tokenizer = tiktoken.get_encoding(config.tokenizer)
    selection_path = args.output_dir / "selection.jsonl"
    completed_tokens = {"train": 0, "validation": 0, "test": 0}
    resume_through_ordinal: int | None = None
    if args.resume:
        if not selection_path.is_file():
            parser.error("--resume requires an existing selection.jsonl in --output-dir")
        with selection_path.open(encoding="utf-8") as selection:
            for line in selection:
                record = json.loads(line)
                completed_tokens[record["split"]] += int(record["tokens_written"])
                resume_through_ordinal = int(record["source_ordinal"])
    targets = ShardTargets.from_data_config(config)
    target_tokens = {"train": targets.train, "validation": targets.validation, "test": targets.test}
    documents = () if completed_tokens == target_tokens else stream_fineweb_edu(
        config,
        args.seed,
        args.selection_modulus,
        skip_through_ordinal=resume_through_ordinal,
    )
    manifest = prepare_token_shards(
        documents,
        tokenizer,
        args.output_dir,
        targets,
        dataset=config.dataset,
        revision=config.revision or "",
        seed=args.seed,
        tokenizer_name=config.tokenizer,
        sampling_description=(
            f"seeded SHA-256 document-id sampling; retained bucket 0 of {args.selection_modulus}"
        ),
        resume=args.resume,
        progress=lambda documents, positions: print(
            f"processed_documents={documents} tokens={positions}", flush=True
        ),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
