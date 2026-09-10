"""Fetch prompt bags from Hugging Face benchmark datasets.

A "bag" is a JSON file that holds a list of prompts. Later, a runner feeds
each prompt to the ``/v1/chat/completions`` endpoint and stores the response.
Each bag comes from one benchmark dataset, downloaded through the Hugging Face
datasets-server REST API (no ``datasets`` package needed — stdlib only).

Run with: ``uv run python fetch_prompts.py`` (all bags) or
``uv run python fetch_prompts.py humaneval_python`` (one bag).
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

DATASETS_SERVER_ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100
DEFAULT_BAGS_DIR = Path(__file__).parent / "bags"

# Type of the page-fetch callable, injectable so tests need no network.
PageFetcher = Callable[["BagSpec", int], dict]


@dataclass(frozen=True)
class BagSpec:
    """Where a bag comes from and which row fields to keep."""

    name: str
    dataset: str
    config: str
    split: str
    id_field: str
    prompt_field: str


BAG_SPECS: dict[str, BagSpec] = {
    "humaneval_python": BagSpec(
        name="humaneval_python",
        dataset="openai/openai_humaneval",
        config="openai_humaneval",
        split="test",
        id_field="task_id",
        prompt_field="prompt",
    ),
    "mbpp_python": BagSpec(
        name="mbpp_python",
        dataset="google-research-datasets/mbpp",
        config="sanitized",
        split="test",
        id_field="task_id",
        prompt_field="prompt",
    ),
}


def fetch_page(spec: BagSpec, offset: int) -> dict:
    """Download one page of rows from the datasets-server API."""
    query = urllib.parse.urlencode(
        {
            "dataset": spec.dataset,
            "config": spec.config,
            "split": spec.split,
            "offset": offset,
            "length": PAGE_SIZE,
        }
    )
    with urllib.request.urlopen(f"{DATASETS_SERVER_ROWS_URL}?{query}") as response:
        return json.load(response)


def fetch_all_rows(spec: BagSpec, page_fetcher: PageFetcher = fetch_page) -> list[dict]:
    """Page through the split and return every row dict."""
    rows: list[dict] = []
    total_rows: int | None = None
    while total_rows is None or len(rows) < total_rows:
        payload = page_fetcher(spec, len(rows))
        total_rows = payload["num_rows_total"]
        page_rows = [entry["row"] for entry in payload["rows"]]
        if not page_rows:
            break  # defensive: never loop forever on an empty page
        rows.extend(page_rows)
    return rows


def rows_to_bag(spec: BagSpec, rows: list[dict]) -> dict:
    """Convert dataset rows into the bag JSON structure."""
    prompts = [
        {"id": str(row[spec.id_field]), "prompt": row[spec.prompt_field]}
        for row in rows
        if row.get(spec.prompt_field)
    ]
    return {
        "name": spec.name,
        "source": {
            "dataset": spec.dataset,
            "config": spec.config,
            "split": spec.split,
        },
        "prompts": prompts,
    }


def write_bag(bag: dict, bags_dir: Path) -> Path:
    """Write one bag to ``<bags_dir>/<name>.json`` and return the path."""
    bags_dir.mkdir(parents=True, exist_ok=True)
    path = bags_dir / f"{bag['name']}.json"
    path.write_text(json.dumps(bag, indent=2, ensure_ascii=False) + "\n")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch_prompts",
        description="Download benchmark prompts into JSON bags.",
    )
    parser.add_argument(
        "bags",
        nargs="*",
        metavar="bag",
        help=f"bag names to fetch (default: all of {', '.join(BAG_SPECS)})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_BAGS_DIR,
        help="directory for the bag JSON files (default: ./bags)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bag_names = args.bags or list(BAG_SPECS)
    unknown_names = [name for name in bag_names if name not in BAG_SPECS]
    if unknown_names:
        known = ", ".join(BAG_SPECS)
        print(f"Unknown bag(s): {', '.join(unknown_names)}. Known bags: {known}")
        return 2
    for bag_name in bag_names:
        spec = BAG_SPECS[bag_name]
        print(f"Fetching {spec.dataset} [{spec.config}/{spec.split}] ...", flush=True)
        rows = fetch_all_rows(spec)
        bag = rows_to_bag(spec, rows)
        path = write_bag(bag, args.out_dir)
        print(f"Wrote {len(bag['prompts'])} prompts to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
