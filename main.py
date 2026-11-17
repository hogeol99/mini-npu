"""Mini NPU Simulator

MAC(Multiply-Accumulate) 연산으로 십자가(Cross)와 X 패턴을 판별하고,
패턴 크기에 따른 연산 시간을 측정하는 콘솔 애플리케이션.

외부 라이브러리 없이 표준 라이브러리만 사용한다.
"""

import json
import time
import re
import math

# === 1. 상수 ===================================================================

EPSILON = 1e-9          # 점수 비교 허용오차
BENCH_REPEAT = 10       # 성능 측정 반복 횟수
BENCH_SIZES = (3, 5, 13, 25)  # 성능 측정 대상 크기
DATA_PATH = "data.json"

# compare() 결과를 모드 1 출력용 이름으로 바꾸는 표
MODE1_LABELS = {"A": "A", "B": "B", "TIE": "판정 불가"}

# === 2. 데이터 구조 (Matrix) ===================================================
class Matrix:
    """n×n 크기의 2차원 숫자 배열 (패턴과 필터 공용)."""

    def __init__(self, size):
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"행렬 크기는 1 이상의 정수여야 합니다: {size!r}")
        self.size = size
        self._data = [[0.0] * size for _ in range(size)]

    def _check_index(self, row, col):
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise IndexError(
                f"위치 ({row}, {col})가 {self.size}x{self.size} 범위를 벗어났습니다."
            )

    def get(self, row, col):
        self._check_index(row, col)
        return self._data[row][col]

    def set(self, row, col, value):
        self._check_index(row, col)
        self._data[row][col] = float(value)

    @classmethod
    def from_rows(cls, rows):
        """2차원 리스트로 Matrix를 만든다. 정사각형이 아니면 ValueError."""
        if not isinstance(rows, list) or len(rows) == 0:
            raise ValueError("행렬 데이터가 비어 있거나 리스트가 아닙니다.")
        n = len(rows)
        matrix = cls(n)
        for r, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != n:
                got = len(row) if isinstance(row, list) else type(row).__name__
                raise ValueError(f"{r + 1}행의 열 개수가 {n}이 아닙니다 (입력: {got}).")
            for c, value in enumerate(row):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"({r + 1}, {c + 1}) 위치 값이 숫자가 아닙니다: {value!r}")
                matrix.set(r, c, value)
        return matrix

    def __repr__(self):
        return f"Matrix({self.size}x{self.size})"

    def to_text(self):
        """콘솔 출력용 문자열."""
        lines = []
        for r in range(self.size):
            lines.append(" ".join(f"{self._data[r][c]:g}" for c in range(self.size)))
        return "\n".join(lines)

# === 3. 연산 (mac, compare) ====================================================
def mac(pattern, flt):
    """패턴과 필터를 같은 위치끼리 곱하고(Multiply) 모두 더한다(Accumulate).

    외부 라이브러리 없이 이중 반복문으로 직접 계산한다.
    연산 횟수(곱셈 수)는 N×N = N²이다.
    """
    if pattern.size != flt.size:
        raise ValueError(
            f"크기 불일치: 패턴 {pattern.size}x{pattern.size}, "
            f"필터 {flt.size}x{flt.size}"
        )
    n = pattern.size
    total = 0.0
    for r in range(n):
        for c in range(n):
            total += pattern.get(r, c) * flt.get(r, c)
    return total


def compare(score_a, score_b, eps=EPSILON):
    """두 점수를 허용오차(epsilon) 기준으로 비교한다.

    반환값: "A"(A가 큼), "B"(B가 큼), "TIE"(차이가 eps 미만이라 동점)
    """
    if abs(score_a - score_b) < eps:
        return "TIE"
    if score_a > score_b:
        return "A"
    return "B"


# === 4. 라벨 정규화 ============================================================

LABEL_CROSS = "Cross"
LABEL_X = "X"
LABEL_UNDECIDED = "UNDECIDED"

# 소문자로 바꾼 입력값 → 표준 라벨
LABEL_MAP = {
    "+": LABEL_CROSS,       # expected 값
    "cross": LABEL_CROSS,   # filter 키
    "x": LABEL_X,           # expected 값, filter 키 공통
}

# compare() 결과를 모드 2 출력용 이름으로 바꾸는 표 (A=Cross 필터, B=X 필터)

MODE2_LABELS = {
    "A": LABEL_CROSS,
    "B": LABEL_X,
    "TIE": LABEL_UNDECIDED
}

def normalize_label(raw):
    """여러 표기('+', 'x', 'cross', 'Cross', ' X ' 등)를 표준 라벨 Cross/X로 바꾼다.

    알 수 없는 값이면 ValueError.
    """
    if not isinstance(raw, str):
        raise ValueError(f"라벨이 문자열이 아닙니다: {raw!r}")
    key = raw.strip().lower()
    if key not in LABEL_MAP:
        raise ValueError(f"알 수 없는 라벨: {raw!r} (허용: '+', 'x', 'cross')")
    return LABEL_MAP[key]


# === 5. 입력 처리 (모드 1) =====================================================
def parse_row(line, n):
    """한 줄 문자열을 숫자 n개의 리스트로 바꾼다. 형식이 틀리면 ValueError."""
    tokens = line.split()
    if len(tokens) != n:
        raise ValueError(
            f"입력 형식 오류: 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요. "
            f"(입력된 개수: {len(tokens)})"
        )
    row = []
    for token in tokens:
        try:
            value = float(token)
        except ValueError:
            raise ValueError(
                f"입력 형식 오류: '{token}'은(는) 숫자가 아닙니다. "
                f"각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요."
            )
        if not math.isfinite(value):
            raise ValueError(f"입력 형식 오류: '{token}'은(는) 사용할 수 없는 값입니다.")
        row.append(value)
    return row


def read_matrix(title, n=3, input_func=input):
    """n×n 행렬을 한 줄씩 입력받는다. 틀린 줄은 그 줄만 다시 입력받는다."""
    print(f"\n[{title}] {n}줄을 입력하세요. (각 줄에 숫자 {n}개, 공백으로 구분)")
    rows = []
    while len(rows) < n:
        line = input_func(f"  {len(rows) + 1}행: ")
        try:
            rows.append(parse_row(line, n))
        except ValueError as e:
            print(f"  {e}")
    return Matrix.from_rows(rows)

# === 6. JSON 로드·검증 (모드 2) ================================================
FILTER_KEY_RE = re.compile(r"^size_(\d+)$")          # 예: size_13
PATTERN_KEY_RE = re.compile(r"^size_(\d+)_(\d+)$")  # 예: size_13_2


def load_json(path):
    """JSON 파일을 읽어 dict로 돌려준다. 실패하면 이유를 담은 ValueError."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"파일을 찾을 수 없습니다: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 형식 오류: {path} ({e.lineno}행 {e.colno}열)")
    if not isinstance(data, dict):
        raise ValueError("JSON 최상위 구조가 객체({})가 아닙니다.")
    return data


def load_filters(data):
    """data["filters"]를 읽어 {N: {"Cross": Matrix, "X": Matrix}}로 만든다.

    잘못된 필터 묶음은 건너뛰고, 그 이유를 errors 리스트에 남긴다.
    반환값: (filters, errors)
    """
    raw_filters = data.get("filters")
    if not isinstance(raw_filters, dict):
        raise ValueError("'filters' 항목이 없거나 객체가 아닙니다.")

    filters = {}
    errors = []
    for key, pair in raw_filters.items():
        try:
            match = FILTER_KEY_RE.match(key)
            if not match:
                raise ValueError("키 형식이 size_N이 아닙니다.")
            n = int(match.group(1))
            if not isinstance(pair, dict):
                raise ValueError("필터 묶음이 객체가 아닙니다.")
            loaded = {}
            for name, rows in pair.items():
                label = normalize_label(name)
                matrix = Matrix.from_rows(rows)
                if matrix.size != n:
                    raise ValueError(
                        f"'{name}' 필터 크기 {matrix.size}x{matrix.size}가 키의 N={n}과 다릅니다."
                    )
                loaded[label] = matrix
            for label in (LABEL_CROSS, LABEL_X):
                if label not in loaded:
                    raise ValueError(f"{label} 필터가 없습니다.")
            filters[n] = loaded
        except ValueError as e:
            errors.append(f"filters.{key}: {e}")
    return filters, errors


def parse_pattern_key(key):
    """'size_13_2' → (13, 2). 형식이 틀리면 ValueError."""
    match = PATTERN_KEY_RE.match(key)
    if not match:
        raise ValueError(f"패턴 키 형식이 size_N_idx가 아닙니다: {key!r}")
    return int(match.group(1)), int(match.group(2))


def prepare_case(key, case, filters):
    """패턴 1개를 검증하고 연산에 필요한 재료를 dict로 돌려준다.

    반환값: {"n", "pattern", "filters", "expected"}
    문제가 있으면 이유를 담은 ValueError (프로그램은 멈추지 않는다).
    """
    n, _ = parse_pattern_key(key)
    if n not in filters:
        raise ValueError(f"size_{n} 필터가 없습니다.")
    if not isinstance(case, dict):
        raise ValueError("패턴 항목이 객체가 아닙니다.")
    if "input" not in case:
        raise ValueError("'input' 항목이 없습니다.")
    if "expected" not in case:
        raise ValueError("'expected' 항목이 없습니다.")

    pattern = Matrix.from_rows(case["input"])
    if pattern.size != n:
        raise ValueError(
            f"크기 불일치: 패턴 {pattern.size}x{pattern.size}, 필터 {n}x{n}"
        )
    expected = normalize_label(case["expected"])
    return {
        "n": n,
        "pattern": pattern,
        "filters": filters[n],
        "expected": expected,
    }

# 실패 원인 분류 (README 분석과 같은 기준)
FAIL_SCHEMA = "데이터/스키마"
FAIL_NUMERIC = "수치 비교"
FAIL_LOGIC = "로직"


def evaluate_case(key, case, filters):
    """패턴 1개를 검증 → MAC → 판정 → expected 비교까지 하고 결과 dict를 돌려준다.

    어떤 문제가 있어도 예외를 밖으로 내보내지 않고 FAIL 결과로 바꾼다.
    """
    result = {
        "id": key, "status": "FAIL", "category": None, "reason": "",
        "cross": None, "x": None, "decision": None, "expected": None,
    }
    try:
        prepared = prepare_case(key, case, filters)
    except ValueError as e:
        result["category"] = FAIL_SCHEMA
        result["reason"] = str(e)
        return result

    try:
        pattern = prepared["pattern"]
        result["expected"] = prepared["expected"]
        result["cross"] = mac(pattern, prepared["filters"][LABEL_CROSS])
        result["x"] = mac(pattern, prepared["filters"][LABEL_X])
        result["decision"] = MODE2_LABELS[compare(result["cross"], result["x"])]
    except Exception as e:  # 예상하지 못한 오류도 케이스 단위로 처리
        result["category"] = FAIL_LOGIC
        result["reason"] = f"연산 중 오류: {e}"
        return result

    if result["decision"] == LABEL_UNDECIDED:
        result["category"] = FAIL_NUMERIC
        result["reason"] = (
            f"두 점수 차이가 epsilon({EPSILON:g}) 미만이라 판정 불가 "
            f"(expected {result['expected']})"
        )
    elif result["decision"] != result["expected"]:
        result["category"] = FAIL_LOGIC
        result["reason"] = f"판정 {result['decision']} ≠ expected {result['expected']}"
    else:
        result["status"] = "PASS"
    return result


def print_case_result(r):
    """케이스 1개 결과를 콘솔에 출력한다."""
    print(f"\n[{r['id']}]")
    if r["category"] == FAIL_SCHEMA:
        print(f"  결과: FAIL ({r['category']}: {r['reason']})")
        return
    print(f"  Cross 점수: {r['cross']}")
    print(f"  X 점수:     {r['x']}")
    print(f"  판정: {r['decision']} / expected: {r['expected']}")
    if r["status"] == "PASS":
        print("  결과: PASS")
    else:
        print(f"  결과: FAIL ({r['category']}: {r['reason']})")
# === 7. 성능 분석 ==============================================================

def make_cross(n):
    """가운데 행과 가운데 열이 1인 n×n 십자가 행렬을 만든다."""
    m = Matrix(n)
    mid = n // 2
    for i in range(n):
        m.set(mid, i, 1)
        m.set(i, mid, 1)
    return m


def make_x(n):
    """두 대각선이 1인 n×n X 행렬을 만든다."""
    m = Matrix(n)
    for i in range(n):
        m.set(i, i, 1)
        m.set(i, n - 1 - i, 1)
    return m


def measure_mac(pattern, flt, repeat=BENCH_REPEAT):
    """MAC 연산을 repeat번 반복 측정해 평균 시간(ms)을 돌려준다.

    입력/출력은 측정 구간에 넣지 않고, mac() 호출 구간만 잰다.
    """
    mac(pattern, flt)  # 워밍업 1회 (첫 호출의 준비 비용을 측정에서 제외)
    total = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        mac(pattern, flt)
        total += time.perf_counter() - start
    return total / repeat * 1000


def run_benchmark(pairs, repeat=BENCH_REPEAT):
    """pairs = {N: (pattern, filter)} 를 크기 순서대로 측정한다.

    반환값: [(N, 평균 ms), ...]
    """
    rows = []
    for n in sorted(pairs):
        pattern, flt = pairs[n]
        rows.append((n, measure_mac(pattern, flt, repeat)))
    return rows


def print_benchmark(rows, repeat=BENCH_REPEAT):
    """성능 분석 표를 출력한다."""
    print(f"\n[성능 분석] (크기별 MAC {repeat}회 반복 평균, I/O 제외)")
    print(f"{'크기(N×N)':<12}{'평균 시간(ms)':>16}{'연산 횟수(N²)':>16}{'1회당(ns)':>14}")
    print("-" * 58)
    for n, avg_ms in rows:
        ops = n * n
        per_op_ns = avg_ms * 1_000_000 / ops
        print(f"{f'{n}x{n}':<12}{avg_ms:>16.6f}{ops:>16}{per_op_ns:>14.1f}")


# === 8. 결과 리포트 ============================================================


# === 9. 모드 실행 / main() =====================================================

def run_mode1(input_func=input):
    """모드 1: 3×3 필터 A, B와 패턴을 입력받아 판정한다."""
    print("\n=== 모드 1: 사용자 입력 (3x3) ===")
    filter_a = read_matrix("필터 A", 3, input_func)
    filter_b = read_matrix("필터 B", 3, input_func)

    print("\n[저장 확인]")
    print("필터 A:")
    print(filter_a.to_text())
    print("필터 B:")
    print(filter_b.to_text())

    pattern = read_matrix("패턴", 3, input_func)

    # 연산 구간만 측정 (입력/출력 시간 제외)
    start = time.perf_counter()
    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    elapsed_ms = (time.perf_counter() - start) * 1000

    result = MODE1_LABELS[compare(score_a, score_b)]

    print("\n[결과]")
    print(f"필터 A 점수: {score_a}")
    print(f"필터 B 점수: {score_b}")
    print(f"연산 시간: {elapsed_ms:.6f} ms")
    print(f"판정 결과: {result}")
    
    print_benchmark(run_benchmark({3: (pattern, filter_a)}))


def run_mode2(path=DATA_PATH):
    """모드 2: data.json을 읽어 모든 패턴을 일괄 판정한다. 결과 리스트를 돌려준다."""
    print(f"\n=== 모드 2: data.json 분석 ({path}) ===")
    try:
        data = load_json(path)
        filters, filter_errors = load_filters(data)
    except ValueError as e:
        print(f"데이터를 불러오지 못했습니다: {e}")
        return []

    print(f"\n[필터 로드] 사용 가능한 크기: {', '.join(f'{n}x{n}' for n in sorted(filters)) or '없음'}")
    for err in filter_errors:
        print(f"  경고: {err}")

    patterns = data.get("patterns")
    if not isinstance(patterns, dict):
        print("'patterns' 항목이 없거나 객체가 아닙니다.")
        return []

    print("\n[패턴 판정]")
    results = []
    for key, case in patterns.items():
        r = evaluate_case(key, case, filters)
        print_case_result(r)
        results.append(r)
    pairs = {}
    for n in BENCH_SIZES:
        flt = filters[n][LABEL_CROSS] if n in filters else make_cross(n)
        pairs[n] = (make_cross(n), flt)
    print_benchmark(run_benchmark(pairs))
    return results

def main():
    print("=== Mini NPU Simulator ===")
    while True:
        print("\n[모드 선택]")
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")
        choice = input("선택(1 또는 2): ").strip()
        if choice == "1":
            run_mode1()
            break
        elif choice == "2":
            path = input(f"JSON 파일 경로 (Enter = {DATA_PATH}): ").strip() or DATA_PATH
            run_mode2(path)
            break
        else:
            print("1 또는 2를 입력하세요.")


if __name__ == "__main__":
    main()