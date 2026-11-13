"""Validate a frozen token-shard manifest without modifying its contents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soap_muon.data.preparation import verify_token_shards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = verify_token_shards(args.data_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
