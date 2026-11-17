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


CROSS_3 = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
X_3 = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]


class TestMac(unittest.TestCase):
    def test_cross_with_cross_filter(self):
        score = main.mac(main.Matrix.from_rows(CROSS_3), main.Matrix.from_rows(CROSS_3))
        self.assertEqual(score, 5.0)

    def test_cross_with_x_filter(self):
        score = main.mac(main.Matrix.from_rows(CROSS_3), main.Matrix.from_rows(X_3))
        self.assertEqual(score, 1.0)

    def test_x_with_x_filter(self):
        score = main.mac(main.Matrix.from_rows(X_3), main.Matrix.from_rows(X_3))
        self.assertEqual(score, 5.0)

    def test_returns_float(self):
        score = main.mac(main.Matrix.from_rows(CROSS_3), main.Matrix.from_rows(X_3))
        self.assertIsInstance(score, float)

    def test_size_mismatch(self):
        with self.assertRaises(ValueError):
            main.mac(main.Matrix(3), main.Matrix(5))

    def test_data_json_scores(self):
        with open(os.path.join(ROOT, "data.json"), encoding="utf-8") as f:
            data = json.load(f)
        expected = {
            "size_5_1": (0.9, 0.9),
            "size_5_2": (8.9, 0.1),
            "size_13_1": (0.3, 14.7),
            "size_13_2": (7.5, 7.5),
            "size_25_1": (4.9, 4.9),
            "size_25_2": (52.9, 0.1),
        }
        for key, (cross_score, x_score) in expected.items():
            n = key.split("_")[1]
            pattern = main.Matrix.from_rows(data["patterns"][key]["input"])
            filters = data["filters"]["size_" + n]
            self.assertAlmostEqual(main.mac(pattern, main.Matrix.from_rows(filters["cross"])), cross_score, places=9)
            self.assertAlmostEqual(main.mac(pattern, main.Matrix.from_rows(filters["x"])), x_score, places=9)


class TestCompare(unittest.TestCase):
    def test_a_wins(self):
        self.assertEqual(main.compare(5.0, 1.0), "A")

    def test_b_wins(self):
        self.assertEqual(main.compare(1.0, 5.0), "B")

    def test_exact_tie(self):
        self.assertEqual(main.compare(5.0, 5.0), "TIE")

    def test_float_error_is_tie(self):
        # 0.9 와 0.8999999999999999 는 부동소수점 오차일 뿐이므로 동점
        self.assertEqual(main.compare(0.9, 0.8999999999999999), "TIE")
        self.assertEqual(main.compare(7.499999999999997, 7.5), "TIE")

    def test_naive_compare_is_wrong(self):
        # epsilon 없이 비교하면 오차 때문에 승패가 갈려 버린다
        self.assertTrue(0.9 > 0.8999999999999999)

    def test_small_but_real_difference(self):
        # 차이가 epsilon보다 크면 동점이 아니다
        self.assertEqual(main.compare(1.0, 1.0 + 1e-6), "B")

    def test_custom_eps(self):
        self.assertEqual(main.compare(1.0, 1.05, eps=0.1), "TIE")

        
if __name__ == "__main__":
    unittest.main()