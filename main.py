"""Mini NPU Simulator

MAC(Multiply-Accumulate) 연산으로 십자가(Cross)와 X 패턴을 판별하고,
패턴 크기에 따른 연산 시간을 측정하는 콘솔 애플리케이션.

외부 라이브러리 없이 표준 라이브러리만 사용한다.
"""

import json
import time


# === 1. 상수 ===================================================================

EPSILON = 1e-9          # 점수 비교 허용오차
BENCH_REPEAT = 10       # 성능 측정 반복 횟수
DATA_PATH = "data.json"


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


# === 4. 라벨 정규화 ============================================================


# === 5. 입력 처리 (모드 1) =====================================================


# === 6. JSON 로드·검증 (모드 2) ================================================


# === 7. 성능 분석 ==============================================================


# === 8. 결과 리포트 ============================================================


# === 9. 모드 실행 / main() =====================================================

def main():
    print("=== Mini NPU Simulator ===")
    print("[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    choice = input("선택(1 또는 2): ").strip()
    if choice == "1":
        print("모드 1은 아직 구현되지 않았습니다.")
    elif choice == "2":
        print("모드 2는 아직 구현되지 않았습니다.")
    else:
        print("1 또는 2를 입력하세요.")


if __name__ == "__main__":
    main()