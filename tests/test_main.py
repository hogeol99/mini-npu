import contextlib
import io
import json
import os
import sys
import tempfile
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

def fake_input(lines):
    """input() 대신 쓸 가짜 입력 함수. 미리 준비한 줄을 차례로 돌려준다."""
    it = iter(lines)
    return lambda prompt="": next(it)


class TestMode1Input(unittest.TestCase):
    def test_parse_row_ok(self):
        self.assertEqual(main.parse_row("0 1 0", 3), [0.0, 1.0, 0.0])
        self.assertEqual(main.parse_row("  0.5   1  -2 ", 3), [0.5, 1.0, -2.0])

    def test_parse_row_wrong_count(self):
        with self.assertRaises(ValueError):
            main.parse_row("0 1", 3)
        with self.assertRaises(ValueError):
            main.parse_row("0 1 0 1", 3)
        with self.assertRaises(ValueError):
            main.parse_row("", 3)

    def test_parse_row_not_number(self):
        with self.assertRaises(ValueError):
            main.parse_row("1 a 0", 3)
        with self.assertRaises(ValueError):
            main.parse_row("1 nan 0", 3)

    def test_read_matrix_retries_bad_lines(self):
        lines = ["0 1", "0 1 0", "1 x 1", "1 1 1", "0 1 0"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            m = main.read_matrix("테스트", 3, fake_input(lines))
        self.assertEqual(m.get(0, 1), 1.0)
        self.assertEqual(m.get(1, 0), 1.0)
        self.assertEqual(out.getvalue().count("입력 형식 오류"), 2)

    def run_mode1(self, lines):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main.run_mode1(fake_input(lines))
        return out.getvalue()

    def test_mode1_cross(self):
        text = self.run_mode1(["0 1 0", "1 1 1", "0 1 0",
                               "1 0 1", "0 1 0", "1 0 1",
                               "0 1 0", "1 1 1", "0 1 0"])
        self.assertIn("필터 A 점수: 5.0", text)
        self.assertIn("필터 B 점수: 1.0", text)
        self.assertIn("판정 결과: A", text)
        self.assertIn("ms", text)
        self.assertIn("[성능 분석]", text)

    def test_mode1_x(self):
        text = self.run_mode1(["0 1 0", "1 1 1", "0 1 0",
                               "1 0 1", "0 1 0", "1 0 1",
                               "1 0 1", "0 1 0", "1 0 1"])
        self.assertIn("판정 결과: B", text)

    def test_mode1_tie(self):
        same = ["0 1 0", "1 1 1", "0 1 0"]
        text = self.run_mode1(same + same + same)
        self.assertIn("판정 결과: 판정 불가", text)    

class TestLabels(unittest.TestCase):
    def test_expected_values(self):
        self.assertEqual(main.normalize_label("+"), "Cross")
        self.assertEqual(main.normalize_label("x"), "X")

    def test_filter_keys(self):
        self.assertEqual(main.normalize_label("cross"), "Cross")
        self.assertEqual(main.normalize_label("x"), "X")

    def test_case_and_spaces(self):
        self.assertEqual(main.normalize_label("Cross"), "Cross")
        self.assertEqual(main.normalize_label(" X "), "X")
        self.assertEqual(main.normalize_label("CROSS"), "Cross")

    def test_already_standard_is_unchanged(self):
        for label in ("Cross", "X"):
            self.assertEqual(main.normalize_label(label), label)

    def test_unknown_label(self):
        for bad in ("?", "", "o", "plus"):
            with self.assertRaises(ValueError):
                main.normalize_label(bad)

    def test_not_string(self):
        with self.assertRaises(ValueError):
            main.normalize_label(None)
        with self.assertRaises(ValueError):
            main.normalize_label(1)

    def test_all_labels_in_data_json(self):
        with open(os.path.join(ROOT, "data.json"), encoding="utf-8") as f:
            data = json.load(f)
        for pair in data["filters"].values():
            for name in pair:
                self.assertIn(main.normalize_label(name), ("Cross", "X"))
        for case in data["patterns"].values():
            self.assertIn(main.normalize_label(case["expected"]), ("Cross", "X"))

def square(n, value=0.0):
    return [[value] * n for _ in range(n)]


class TestLoader(unittest.TestCase):
    def setUp(self):
        # 테스트마다 새로 만드는 작은 정상 데이터 (3×3 필터 1세트)
        self.data = {
            "filters": {
                "size_3": {"cross": CROSS_3, "x": X_3},
            },
            "patterns": {},
        }

    def test_load_json_ok(self):
        data = main.load_json(os.path.join(ROOT, "data.json"))
        self.assertIn("filters", data)
        self.assertIn("patterns", data)

    def test_load_json_missing_file(self):
        with self.assertRaises(ValueError):
            main.load_json(os.path.join(ROOT, "no_such_file.json"))

    def test_load_filters_from_data_json(self):
        data = main.load_json(os.path.join(ROOT, "data.json"))
        filters, errors = main.load_filters(data)
        self.assertEqual(errors, [])
        self.assertEqual(sorted(filters), [5, 13, 25])
        self.assertEqual(sorted(filters[13]), ["Cross", "X"])

    def test_load_filters_size_mismatch(self):
        self.data["filters"]["size_5"] = {"cross": square(3), "x": square(5)}
        filters, errors = main.load_filters(self.data)
        self.assertIn(3, filters)          # 정상 묶음은 살아남고
        self.assertNotIn(5, filters)       # 잘못된 묶음만 빠진다
        self.assertEqual(len(errors), 1)

    def test_load_filters_missing_section(self):
        with self.assertRaises(ValueError):
            main.load_filters({"patterns": {}})

    def test_parse_pattern_key(self):
        self.assertEqual(main.parse_pattern_key("size_13_2"), (13, 2))
        for bad in ("size_13", "13_2", "size_a_1", "size_5_1_x"):
            with self.assertRaises(ValueError):
                main.parse_pattern_key(bad)

    def test_prepare_case_ok(self):
        filters, _ = main.load_filters(self.data)
        prepared = main.prepare_case("size_3_1", {"input": CROSS_3, "expected": "+"}, filters)
        self.assertEqual(prepared["n"], 3)
        self.assertEqual(prepared["expected"], "Cross")
        self.assertEqual(prepared["pattern"].size, 3)

    def test_prepare_case_errors(self):
        filters, _ = main.load_filters(self.data)
        bad_cases = {
            "size_3_1": {"input": square(5), "expected": "+"},   # 크기 불일치
            "size_7_1": {"input": square(7), "expected": "x"},   # 필터 없음
            "size_3_2": {"input": CROSS_3, "expected": "?"},     # 알 수 없는 라벨
            "size_3_3": {"expected": "+"},                       # input 없음
            "size_3_4": {"input": CROSS_3},                      # expected 없음
            "pattern_1": {"input": CROSS_3, "expected": "+"},    # 키 형식 오류
            "size_3_5": {"input": [[0, 1], [1, 1, 1], [0, 1, 0]], "expected": "+"},  # 행 길이 불일치
        }
        for key, case in bad_cases.items():
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    main.prepare_case(key, case, filters)

    def test_prepare_all_data_json_cases(self):
        data = main.load_json(os.path.join(ROOT, "data.json"))
        filters, _ = main.load_filters(data)
        for key, case in data["patterns"].items():
            prepared = main.prepare_case(key, case, filters)
            self.assertIn(prepared["expected"], ("Cross", "X"))

class TestMode2(unittest.TestCase):
    def setUp(self):
        self.data = main.load_json(os.path.join(ROOT, "data.json"))
        self.filters, _ = main.load_filters(self.data)

    def evaluate(self, key, case=None):
        if case is None:
            case = self.data["patterns"][key]
        return main.evaluate_case(key, case, self.filters)

    def test_data_json_expected_results(self):
        expected = {
            "size_5_1": ("FAIL", "UNDECIDED"),
            "size_5_2": ("PASS", "Cross"),
            "size_13_1": ("PASS", "X"),
            "size_13_2": ("FAIL", "UNDECIDED"),
            "size_25_1": ("FAIL", "UNDECIDED"),
            "size_25_2": ("PASS", "Cross"),
        }
        for key, (status, decision) in expected.items():
            with self.subTest(key=key):
                r = self.evaluate(key)
                self.assertEqual(r["status"], status)
                self.assertEqual(r["decision"], decision)

    def test_undecided_is_numeric_fail(self):
        r = self.evaluate("size_5_1")
        self.assertEqual(r["category"], main.FAIL_NUMERIC)

    def test_wrong_expected_is_logic_fail(self):
        case = dict(self.data["patterns"]["size_5_2"], expected="x")
        r = self.evaluate("size_5_2", case)
        self.assertEqual(r["status"], "FAIL")
        self.assertEqual(r["category"], main.FAIL_LOGIC)

    def test_size_mismatch_is_schema_fail(self):
        case = {"input": self.data["patterns"]["size_5_1"]["input"], "expected": "x"}
        r = self.evaluate("size_13_9", case)
        self.assertEqual(r["status"], "FAIL")
        self.assertEqual(r["category"], main.FAIL_SCHEMA)
        self.assertIn("크기 불일치", r["reason"])

    def run_mode2_text(self, path):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            results = main.run_mode2(path)
        return results, out.getvalue()

    def test_run_mode2_data_json(self):
        results, text = self.run_mode2_text(os.path.join(ROOT, "data.json"))
        self.assertEqual(len(results), 6)
        self.assertEqual(text.count("결과: PASS"), 3)
        self.assertEqual(text.count("결과: FAIL"), 3)

    def test_run_mode2_missing_file(self):
        results, text = self.run_mode2_text(os.path.join(ROOT, "no_such_file.json"))
        self.assertEqual(results, [])
        self.assertIn("파일을 찾을 수 없습니다", text)

    def test_run_mode2_broken_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "broken.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"filters": {,}')
            results, text = self.run_mode2_text(path)
        self.assertEqual(results, [])
        self.assertIn("JSON 형식 오류", text)

    def test_run_mode2_bad_data_continues(self):
        path = os.path.join(ROOT, "tests", "bad_data.json")
        results, text = self.run_mode2_text(path)
        statuses = {r["id"]: r["status"] for r in results}
        self.assertEqual(statuses["size_3_1"], "PASS")       # 정상 케이스는 계속 처리됨
        self.assertEqual(len(results), 5)
        self.assertEqual(sum(1 for r in results if r["status"] == "FAIL"), 4)

class TestBenchmark(unittest.TestCase):
    def test_make_cross_and_x(self):
        cross, x = main.make_cross(3), main.make_x(3)
        for r in range(3):
            for c in range(3):
                self.assertEqual(cross.get(r, c), CROSS_3[r][c])
                self.assertEqual(x.get(r, c), X_3[r][c])

    def test_generated_patterns_score(self):
        # n×n 십자가끼리 MAC = 칸 수 (2n-1), X끼리 = 2n-1 (n이 홀수일 때)
        for n in (5, 13, 25):
            self.assertEqual(main.mac(main.make_cross(n), main.make_cross(n)), 2 * n - 1)
            self.assertEqual(main.mac(main.make_x(n), main.make_x(n)), 2 * n - 1)

    def test_measure_returns_positive_ms(self):
        avg = main.measure_mac(main.make_cross(5), main.make_x(5), repeat=10)
        self.assertIsInstance(avg, float)
        self.assertGreater(avg, 0)

    def test_run_benchmark_sorted_sizes(self):
        pairs = {n: (main.make_cross(n), main.make_x(n)) for n in (25, 3, 13, 5)}
        rows = main.run_benchmark(pairs, repeat=10)
        self.assertEqual([n for n, _ in rows], [3, 5, 13, 25])

    def test_bigger_is_slower(self):
        rows = dict(main.run_benchmark(
            {n: (main.make_cross(n), main.make_x(n)) for n in (3, 25)}, repeat=10))
        self.assertGreater(rows[25], rows[3])

    def test_print_table(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main.print_benchmark([(3, 0.001), (25, 0.05)])
        text = out.getvalue()
        self.assertIn("크기(N×N)", text)
        self.assertIn("연산 횟수(N²)", text)
        self.assertIn("625", text)

    def test_mode2_prints_all_sizes(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main.run_mode2(os.path.join(ROOT, "data.json"))
        text = out.getvalue()
        for size in ("3x3", "5x5", "13x13", "25x25"):
            self.assertIn(size, text)
        self.assertIn("[성능 분석]", text)    
                 
if __name__ == "__main__":
    unittest.main()