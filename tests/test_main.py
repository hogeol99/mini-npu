import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import main  # noqa: E402


class TestMatrix(unittest.TestCase):
    def test_get_set(self):
        m = main.Matrix(3)
        m.set(1, 2, 7)
        self.assertEqual(m.get(1, 2), 7.0)
        self.assertEqual(m.get(0, 0), 0.0)

    def test_required_sizes(self):
        for n in (3, 5, 13, 25):
            m = main.Matrix(n)
            m.set(n - 1, n - 1, 1)
            self.assertEqual(m.size, n)
            self.assertEqual(m.get(n - 1, n - 1), 1.0)

    def test_out_of_range(self):
        with self.assertRaises(IndexError):
            main.Matrix(3).get(3, 0)

    def test_from_rows_ok(self):
        m = main.Matrix.from_rows([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        self.assertEqual(m.size, 3)
        self.assertEqual(m.get(1, 1), 1.0)

    def test_from_rows_not_square(self):
        with self.assertRaises(ValueError):
            main.Matrix.from_rows([[0, 1, 0], [1, 1], [0, 1, 0]])

    def test_from_rows_non_number(self):
        with self.assertRaises(ValueError):
            main.Matrix.from_rows([[0, "a", 0], [1, 1, 1], [0, 1, 0]])

    def test_load_all_data_json_matrices(self):
        with open(os.path.join(ROOT, "data.json"), encoding="utf-8") as f:
            data = json.load(f)
        for key, pair in data["filters"].items():
            for name, rows in pair.items():
                self.assertEqual(main.Matrix.from_rows(rows).size, int(key.split("_")[1]))
        for key, case in data["patterns"].items():
            self.assertEqual(main.Matrix.from_rows(case["input"]).size, int(key.split("_")[1]))


if __name__ == "__main__":
    unittest.main()