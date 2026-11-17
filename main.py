"""Console entry point for the Mini NPU Simulator."""

from __future__ import annotations

from pathlib import Path
from typing import List

from npu import (
    EPSILON,
    DataSchemaError,
    MatrixValidationError,
    analyze_data,
    decide_label,
    generate_cross,
    generate_x,
    load_json_data,
    mac_score,
    mac_score_flat,
    measure_mac,
    measure_mac_flat,
    performance_rows,
)


DATA_PATH = Path(__file__).with_name("data.json")


def read_matrix(size: int, title: str) -> List[List[float]]:
    """Read a square matrix, restarting the entire matrix after a bad row."""

    while True:
        print("\n{} ({}줄 입력, 각 줄에 숫자 {}개를 공백으로 구분)".format(title, size, size))
        rows: List[List[float]] = []
        valid = True
        for row_number in range(1, size + 1):
            raw = input("  {}행: ".format(row_number)).strip()
            pieces = raw.split()
            if len(pieces) != size:
                valid = False
            else:
                try:
                    rows.append([float(piece) for piece in pieces])
                except ValueError:
                    valid = False
            if not valid:
                print("입력 형식 오류: 각 줄에 {}개의 숫자를 공백으로 구분해 입력하세요.".format(size))
                break
        if valid and len(rows) == size:
            return rows
        print("{} 입력을 처음부터 다시 받습니다.".format(title))


def print_performance(filters: dict) -> None:
    print("\n[성능 분석: MAC 함수 호출만 평균/10회 측정]")
    print("크기(N×N) | 평균 시간(ms) | 연산 횟수(N²)")
    print("-" * 47)
    for size, average_ms, operation_count in performance_rows(filters, repeats=10):
        print("{:>3}×{:<3} | {:>12.6f} | {:>13}".format(size, size, average_ms, operation_count))


def run_manual_mode() -> None:
    print("\n=== 사용자 입력 모드 (3×3) ===")
    filter_a = read_matrix(3, "필터 A")
    filter_b = read_matrix(3, "필터 B")
    print("필터 A와 필터 B를 저장했습니다.")
    pattern = read_matrix(3, "패턴")

    score_a = mac_score(pattern, filter_a)
    score_b = mac_score(pattern, filter_b)
    if abs(score_a - score_b) < EPSILON:
        decision = "UNDECIDED (판정 불가: |A-B| < 1e-9)"
    else:
        decision = "A" if score_a > score_b else "B"
    average_ms = measure_mac(pattern, filter_a, repeats=10)

    print("\n[MAC 결과]")
    print("A 점수: {:.10g}".format(score_a))
    print("B 점수: {:.10g}".format(score_b))
    print("연산 시간(평균/10회): {:.6f} ms".format(average_ms))
    print("판정: {}".format(decision))
    print("연산 횟수: 3² = 9")


def run_json_mode() -> None:
    print("\n=== data.json 분석 모드 ===")
    try:
        data = load_json_data(DATA_PATH)
        results = analyze_data(data)
    except (OSError, DataSchemaError) as error:
        print("data.json을 분석할 수 없습니다: {}".format(error))
        return

    print("\n[패턴 분석: 라벨 정규화 적용]")
    for result in results:
        print("\n--- {} ---".format(result.identifier))
        if result.cross_score is None:
            print("FAIL | 사유: {}".format(result.reason))
            continue
        print("Cross 점수: {:.10g}".format(result.cross_score))
        print("X 점수: {:.10g}".format(result.x_score))
        print("판정: {} | expected: {} | {}".format(result.actual, result.expected, "PASS" if result.passed else "FAIL"))
        if result.reason:
            print("사유: {}".format(result.reason))

    try:
        print_performance(data["filters"])
    except (DataSchemaError, MatrixValidationError, ValueError) as error:
        print("성능 분석을 수행할 수 없습니다: {}".format(error))

    passed = sum(1 for result in results if result.passed)
    failed = [result for result in results if not result.passed]
    print("\n[결과 요약]")
    print("총 테스트: {}개 | 통과: {}개 | 실패: {}개".format(len(results), passed, len(failed)))
    if failed:
        print("실패 케이스:")
        for result in failed:
            print("- {}: {}".format(result.identifier, result.reason))
    else:
        print("실패 케이스 없음: 라벨 정규화, 크기 검증, epsilon 정책을 모두 통과했습니다.")


def run_bonus_mode() -> None:
    print("\n=== 보너스: 자동 패턴 생성 + 1차원 MAC 비교 ===")
    raw_size = input("홀수 크기 N을 입력하세요 (3 이상): ").strip()
    try:
        size = int(raw_size)
        cross = generate_cross(size)
        x_shape = generate_x(size)
        score_matrix = mac_score(cross, x_shape)
        score_flat = mac_score_flat(cross, x_shape)
        matrix_ms = measure_mac(cross, x_shape, repeats=100)
        flat_ms = measure_mac_flat(cross, x_shape, repeats=100)
    except (ValueError, MatrixValidationError) as error:
        print("보너스 실행 오류: {}".format(error))
        return

    print("{}×{} Cross/X 패턴을 생성했습니다.".format(size, size))
    print("2차원 MAC 점수: {:.10g}".format(score_matrix))
    print("1차원 MAC 점수: {:.10g}".format(score_flat))
    print("2차원 평균(100회): {:.6f} ms".format(matrix_ms))
    print("1차원 평균(100회): {:.6f} ms".format(flat_ms))
    print("두 점수가 같은지: {}".format("예" if abs(score_matrix - score_flat) < EPSILON else "아니오"))


def main() -> None:
    print("=== Mini NPU Simulator ===")
    while True:
        print("\n[모드 선택]")
        print("1. 사용자 입력 (3×3)")
        print("2. data.json 분석")
        print("3. 보너스: 패턴 생성/1차원 MAC 비교")
        print("0. 종료")
        try:
            choice = input("선택: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n입력을 종료합니다.")
            return

        if choice == "1":
            run_manual_mode()
        elif choice == "2":
            run_json_mode()
        elif choice == "3":
            run_bonus_mode()
        elif choice == "0":
            print("Mini NPU Simulator를 종료합니다.")
            return
        else:
            print("메뉴 오류: 0, 1, 2, 3 중 하나를 입력하세요.")


if __name__ == "__main__":
    main()
