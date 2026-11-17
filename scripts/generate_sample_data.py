"""Create the deterministic data.json used by the Mini NPU simulator.

This helper uses only the Python standard library.  It is not required to run
the application; it documents how the checked-in sample matrices were made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def cross(size: int) -> List[List[int]]:
    middle = size // 2
    return [[1 if row == middle or column == middle else 0 for column in range(size)] for row in range(size)]


def x_shape(size: int) -> List[List[int]]:
    return [[1 if row == column or row + column == size - 1 else 0 for column in range(size)] for row in range(size)]


def main() -> None:
    filters: Dict[str, Dict[str, List[List[int]]]] = {}
    patterns: Dict[str, Dict[str, object]] = {}
    for size in (5, 13, 25):
        cross_filter = cross(size)
        x_filter = x_shape(size)
        filters[f"size_{size}"] = {"cross": cross_filter, "x": x_filter}
        # '+' and lower-case 'x' deliberately demonstrate label normalization.
        patterns[f"size_{size}_1"] = {"input": cross_filter, "expected": "+"}
        patterns[f"size_{size}_2"] = {"input": x_filter, "expected": "x"}

    destination = Path(__file__).resolve().parents[1] / "data.json"
    destination.write_text(
        json.dumps({"filters": filters, "patterns": patterns}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Created {destination}")


if __name__ == "__main__":
    main()

