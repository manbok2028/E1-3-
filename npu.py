"""Core functions for the Mini NPU Simulator.

Only the Python standard library is used.  The module keeps the calculation,
data validation, label normalization, and timing rules separate from the
console interface so that each rule can be tested independently.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


EPSILON = 1e-9
STANDARD_LABELS = ("Cross", "X")


class MatrixValidationError(ValueError):
    """Raised when a value is not a square numeric matrix."""


class DataSchemaError(ValueError):
    """Raised when the top-level JSON document has an unusable shape."""


@dataclass(frozen=True)
class CaseResult:
    """One data.json pattern-analysis result, including a safe failure reason."""

    identifier: str
    cross_score: Optional[float]
    x_score: Optional[float]
    actual: Optional[str]
    expected: Optional[str]
    passed: bool
    reason: str = ""


def normalize_label(value: Any) -> str:
    """Convert accepted external labels to the two labels used internally."""

    if not isinstance(value, str):
        raise ValueError("라벨은 문자열이어야 합니다.")
    cleaned = value.strip().lower()
    if cleaned in ("+", "cross"):
        return "Cross"
    if cleaned == "x":
        return "X"
    raise ValueError("알 수 없는 라벨: {!r} (허용: +, cross, x)".format(value))


def validate_square_matrix(matrix: Any, expected_size: Optional[int] = None, name: str = "행렬") -> int:
    """Validate a list-of-lists numeric square matrix and return its side size."""

    if not isinstance(matrix, list) or not matrix:
        raise MatrixValidationError("{}은(는) 비어 있지 않은 2차원 배열이어야 합니다.".format(name))
    size = len(matrix)
    if expected_size is not None and size != expected_size:
        raise MatrixValidationError("{}의 행 수가 {}이어야 하지만 {}입니다.".format(name, expected_size, size))

    for row_index, row in enumerate(matrix, start=1):
        if not isinstance(row, list) or len(row) != size:
            raise MatrixValidationError(
                "{}의 {}번째 행 길이가 {}이어야 합니다.".format(name, row_index, size)
            )
        for column_index, value in enumerate(row, start=1):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MatrixValidationError(
                    "{}의 ({}, {}) 값은 숫자여야 합니다.".format(name, row_index, column_index)
                )
    return size


def matrix_size(matrix: Any, name: str = "행렬") -> int:
    """Small readable alias used by callers before timing or flattening."""

    return validate_square_matrix(matrix, name=name)


def mac_score(pattern: List[List[float]], filter_matrix: List[List[float]]) -> float:
    """Compute a MAC score with explicit nested loops (no vector library)."""

    size = validate_square_matrix(pattern, name="패턴")
    validate_square_matrix(filter_matrix, expected_size=size, name="필터")
    total = 0.0
    for row in range(size):
        for column in range(size):
            total += pattern[row][column] * filter_matrix[row][column]
    return total


def decide_label(cross_score: float, x_score: float, epsilon: float = EPSILON) -> str:
    """Return Cross, X, or UNDECIDED using the required epsilon policy."""

    if epsilon < 0:
        raise ValueError("epsilon은 0 이상이어야 합니다.")
    if abs(cross_score - x_score) < epsilon:
        return "UNDECIDED"
    return "Cross" if cross_score > x_score else "X"


def generate_cross(size: int) -> List[List[int]]:
    """Bonus: generate an odd-sized Cross pattern."""

    _validate_odd_size(size)
    middle = size // 2
    return [[1 if row == middle or column == middle else 0 for column in range(size)] for row in range(size)]


def generate_x(size: int) -> List[List[int]]:
    """Bonus: generate an odd-sized X pattern."""

    _validate_odd_size(size)
    return [[1 if row == column or row + column == size - 1 else 0 for column in range(size)] for row in range(size)]


def _validate_odd_size(size: int) -> None:
    if isinstance(size, bool) or not isinstance(size, int) or size < 3 or size % 2 == 0:
        raise ValueError("크기는 3 이상의 홀수여야 합니다.")


def flatten(matrix: List[List[float]]) -> List[float]:
    """Bonus: turn a square matrix into a simple row-major list."""

    validate_square_matrix(matrix)
    return [value for row in matrix for value in row]


def mac_score_flat(pattern: List[List[float]], filter_matrix: List[List[float]]) -> float:
    """Bonus MAC implementation using one-dimensional row-major lists."""

    pattern_size = validate_square_matrix(pattern, name="패턴")
    validate_square_matrix(filter_matrix, expected_size=pattern_size, name="필터")
    flat_pattern = flatten(pattern)
    flat_filter = flatten(filter_matrix)
    total = 0.0
    for index in range(len(flat_pattern)):
        total += flat_pattern[index] * flat_filter[index]
    return total


def measure_mac(pattern: List[List[float]], filter_matrix: List[List[float]], repeats: int = 10) -> float:
    """Measure only repeated MAC calls and return their average in milliseconds."""

    if repeats < 1:
        raise ValueError("반복 횟수는 1 이상이어야 합니다.")
    # Validate before timing so validation/file/I-O time is not included.
    mac_score(pattern, filter_matrix)
    started = time.perf_counter()
    for _ in range(repeats):
        mac_score(pattern, filter_matrix)
    elapsed = time.perf_counter() - started
    return elapsed * 1000 / repeats


def measure_mac_flat(pattern: List[List[float]], filter_matrix: List[List[float]], repeats: int = 10) -> float:
    """Measure the bonus one-dimensional MAC implementation in milliseconds."""

    if repeats < 1:
        raise ValueError("반복 횟수는 1 이상이어야 합니다.")
    mac_score_flat(pattern, filter_matrix)
    started = time.perf_counter()
    for _ in range(repeats):
        mac_score_flat(pattern, filter_matrix)
    elapsed = time.perf_counter() - started
    return elapsed * 1000 / repeats


def load_json_data(path: Path) -> Dict[str, Any]:
    """Load and minimally validate the top-level JSON schema."""

    try:
        with path.open("r", encoding="utf-8") as source:
            data = json.load(source)
    except json.JSONDecodeError as error:
        raise DataSchemaError("JSON 형식 오류: {}".format(error)) from error

    if not isinstance(data, dict):
        raise DataSchemaError("data.json의 최상위 값은 객체여야 합니다.")
    if not isinstance(data.get("filters"), dict):
        raise DataSchemaError("data.json에 filters 객체가 필요합니다.")
    if not isinstance(data.get("patterns"), dict):
        raise DataSchemaError("data.json에 patterns 객체가 필요합니다.")
    return data


def extract_size_from_key(identifier: str) -> int:
    """Read N from the required pattern key format: size_{N}_{idx}."""

    matched = re.fullmatch(r"size_(\d+)_(\d+)", identifier)
    if not matched:
        raise ValueError("패턴 키 {!r}는 size_{{N}}_{{idx}} 형식이어야 합니다.".format(identifier))
    return int(matched.group(1))


def get_filter_pair(filters: Dict[str, Any], size: int) -> Tuple[List[List[float]], List[List[float]]]:
    """Return (Cross filter, X filter) for a size after key-label normalization."""

    group_key = "size_{}".format(size)
    group = filters.get(group_key)
    if not isinstance(group, dict):
        raise DataSchemaError("filters에 {} 필터 묶음이 없습니다.".format(group_key))

    normalized: Dict[str, List[List[float]]] = {}
    for original_name, matrix in group.items():
        label = normalize_label(original_name)
        if label in normalized:
            raise DataSchemaError("{} 필터가 중복되었습니다.".format(label))
        validate_square_matrix(matrix, expected_size=size, name="{} {} 필터".format(group_key, label))
        normalized[label] = matrix

    missing = [label for label in STANDARD_LABELS if label not in normalized]
    if missing:
        raise DataSchemaError("{}에 {} 필터가 없습니다.".format(group_key, ", ".join(missing)))
    return normalized["Cross"], normalized["X"]


def analyze_data(data: Dict[str, Any], epsilon: float = EPSILON) -> List[CaseResult]:
    """Analyze every JSON pattern independently; a bad case never stops the batch."""

    if not isinstance(data, dict) or not isinstance(data.get("filters"), dict) or not isinstance(data.get("patterns"), dict):
        raise DataSchemaError("분석하려면 filters와 patterns 객체가 필요합니다.")

    results: List[CaseResult] = []
    for identifier in sorted(data["patterns"]):
        try:
            item = data["patterns"][identifier]
            if not isinstance(item, dict):
                raise DataSchemaError("패턴 항목은 객체여야 합니다.")
            size = extract_size_from_key(identifier)
            cross_filter, x_filter = get_filter_pair(data["filters"], size)
            pattern = item.get("input")
            validate_square_matrix(pattern, expected_size=size, name="{} input".format(identifier))
            expected = normalize_label(item.get("expected"))
            cross_score = mac_score(pattern, cross_filter)
            x_score = mac_score(pattern, x_filter)
            actual = decide_label(cross_score, x_score, epsilon)
            passed = actual == expected
            reason = "" if passed else "판정 {} / expected {} (동점 정책 포함)".format(actual, expected)
            results.append(CaseResult(identifier, cross_score, x_score, actual, expected, passed, reason))
        except (ValueError, MatrixValidationError, DataSchemaError, TypeError) as error:
            results.append(CaseResult(identifier, None, None, None, None, False, str(error)))
    return results


def performance_rows(filters: Dict[str, Any], repeats: int = 10) -> List[Tuple[int, float, int]]:
    """Create required 3/5/13/25 average-MAC timing rows."""

    rows: List[Tuple[int, float, int]] = []
    pattern_3 = generate_cross(3)
    rows.append((3, measure_mac(pattern_3, pattern_3, repeats), 9))
    for size in (5, 13, 25):
        cross_filter, _ = get_filter_pair(filters, size)
        rows.append((size, measure_mac(cross_filter, cross_filter, repeats), size * size))
    return rows
