"""Unit tests for the bag fetcher. No network — the page fetcher is injected."""

import json
from pathlib import Path

import pytest

from fetch_prompts import (
    BAG_SPECS,
    BagSpec,
    fetch_all_rows,
    main,
    rows_to_bag,
    write_bag,
)

SPEC = BagSpec(
    name="test_bag",
    dataset="org/dataset",
    config="default",
    split="test",
    id_field="task_id",
    prompt_field="prompt",
)


def make_page_fetcher(all_rows: list[dict], page_size: int):
    """Simulate the datasets-server paged /rows endpoint."""

    def page_fetcher(spec: BagSpec, offset: int) -> dict:
        page = all_rows[offset : offset + page_size]
        return {
            "num_rows_total": len(all_rows),
            "rows": [{"row": row} for row in page],
        }

    return page_fetcher


def test_fetch_all_rows_pages_through_split() -> None:
    all_rows = [{"task_id": i, "prompt": f"p{i}"} for i in range(250)]
    rows = fetch_all_rows(SPEC, page_fetcher=make_page_fetcher(all_rows, page_size=100))
    assert rows == all_rows


def test_fetch_all_rows_stops_on_empty_page() -> None:
    def broken_fetcher(spec: BagSpec, offset: int) -> dict:
        return {"num_rows_total": 10, "rows": []}

    assert fetch_all_rows(SPEC, page_fetcher=broken_fetcher) == []


def test_rows_to_bag_maps_fields_and_skips_empty_prompts() -> None:
    rows = [
        {"task_id": 1, "prompt": "Write a function.", "extra": "dropped"},
        {"task_id": 2, "prompt": ""},
        {"task_id": 3},
    ]
    bag = rows_to_bag(SPEC, rows)
    assert bag["name"] == "test_bag"
    assert bag["source"] == {
        "dataset": "org/dataset",
        "config": "default",
        "split": "test",
    }
    assert bag["prompts"] == [{"id": "1", "prompt": "Write a function."}]


def test_write_bag_round_trips(tmp_path: Path) -> None:
    bag = rows_to_bag(SPEC, [{"task_id": 7, "prompt": "hello"}])
    path = write_bag(bag, tmp_path)
    assert path == tmp_path / "test_bag.json"
    assert json.loads(path.read_text()) == bag


def test_main_rejects_unknown_bag(capsys: pytest.CaptureFixture) -> None:
    assert main(["no_such_bag"]) == 2
    assert "Unknown bag(s): no_such_bag" in capsys.readouterr().out


def test_bag_specs_are_consistent() -> None:
    for key, spec in BAG_SPECS.items():
        assert key == spec.name
