import copy
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from npu import (  # noqa: E402
    EPSILON,
    DataSchemaError,
    MatrixValidationError,
    analyze_data,
    decide_label,
    flatten,
    generate_cross,
    generate_x,
    load_json_data,
    mac_score,
    mac_score_flat,
    normalize_label,
    validate_square_matrix,
)


class MiniNpuTest(unittest.TestCase):
    def test_cross_pattern_has_higher_cross_score(self):
        cross = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        x_shape = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        self.assertEqual(mac_score(cross, cross), 5.0)
        self.assertEqual(mac_score(cross, x_shape), 1.0)
        self.assertEqual(decide_label(mac_score(cross, cross), mac_score(cross, x_shape)), "Cross")

    def test_epsilon_policy_returns_undecided(self):
        self.assertEqual(decide_label(0.9, 0.9 + EPSILON / 2), "UNDECIDED")
        self.assertEqual(decide_label(1.0, 1.0 + EPSILON * 2), "X")

    def test_label_normalization(self):
        self.assertEqual(normalize_label("+"), "Cross")
        self.assertEqual(normalize_label(" CROSS "), "Cross")
        self.assertEqual(normalize_label("x"), "X")
        with self.assertRaises(ValueError):
            normalize_label("triangle")

    def test_generated_patterns_have_expected_size(self):
        self.assertEqual(generate_cross(5)[2], [1, 1, 1, 1, 1])
        self.assertEqual(generate_x(5)[0], [1, 0, 0, 0, 1])
        with self.assertRaises(ValueError):
            generate_cross(4)

    def test_flat_bonus_matches_two_dimensional_mac(self):
        pattern = generate_cross(13)
        filter_matrix = generate_x(13)
        self.assertEqual(mac_score(pattern, filter_matrix), mac_score_flat(pattern, filter_matrix))
        self.assertEqual(len(flatten(pattern)), 169)

    def test_matrix_validation_rejects_bad_shape_and_boolean(self):
        with self.assertRaises(MatrixValidationError):
            validate_square_matrix([[1, 0], [1]])
        with self.assertRaises(MatrixValidationError):
            validate_square_matrix([[True]])

    def test_checked_in_data_passes_all_cases(self):
        data = load_json_data(PROJECT_ROOT / "data.json")
        results = analyze_data(data)
        self.assertEqual(len(results), 6)
        self.assertTrue(all(result.passed for result in results))

    def test_bad_pattern_size_becomes_case_fail_not_crash(self):
        data = {
            "filters": {
                "size_3": {
                    "cross": generate_cross(3),
                    "x": generate_x(3),
                }
            },
            "patterns": {
                "size_3_1": {
                    "input": [[1, 0], [0, 1]],
                    "expected": "+",
                }
            },
        }
        result = analyze_data(data)[0]
        self.assertFalse(result.passed)
        self.assertIn("행 수", result.reason)

    def test_bad_pattern_key_becomes_case_fail_not_crash(self):
        data = {
            "filters": {"size_3": {"cross": generate_cross(3), "x": generate_x(3)}},
            "patterns": {"wrong_key": {"input": generate_cross(3), "expected": "+"}},
        }
        result = analyze_data(data)[0]
        self.assertFalse(result.passed)
        self.assertIn("size_{N}_{idx}", result.reason)

    def test_missing_filter_is_safe_case_fail(self):
        data = {
            "filters": {"size_3": {"cross": generate_cross(3)}},
            "patterns": {"size_3_1": {"input": generate_cross(3), "expected": "+"}},
        }
        result = analyze_data(data)[0]
        self.assertFalse(result.passed)
        self.assertIn("X 필터", result.reason)


if __name__ == "__main__":
    unittest.main()
