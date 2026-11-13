"""Validate and display the resolved configuration for one experiment."""

from __future__ import annotations

import argparse
import json

from soap_muon.config import load_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to an experiment YAML file.")
    args = parser.parse_args()
    print(json.dumps(load_experiment(args.config).as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
