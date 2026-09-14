#!/usr/bin/env python3
"""
run_pipeline.py
Video Mask Inpainting Pipeline - CLI 진입점.

사용 예시:
    # 포인트 기반 트래킹
    python run_pipeline.py \\
        --input input.mp4 \\
        --output output.mp4 \\
        --prompt "a gravel road with green grass" \\
        --seed-points "[[320, 240]]" \\
        --frame-range "[0, 100]"

    # 박스 기반 트래킹
    python run_pipeline.py \\
        --input input.mp4 \\
        --output output.mp4 \\
        --prompt "clear blue sky" \\
        --seed-box "[100, 100, 500, 400]"

    # JSON 입력
    python run_pipeline.py \\
        --input input.mp4 \\
        --output output.mp4 \\
        --json-input '{"positive_prompt": "grass field", ...}'
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import load_config
from pipeline.orchestrator import PipelineOrchestrator


def setup_logging(verbose: bool = False) -> None:
    """로깅을 설정합니다."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def parse_args() -> argparse.Namespace:
    """커맨드라인 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="Video Mask Inpainting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 필수 인자
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="입력 비디오 파일 경로",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="출력 비디오 파일 경로",
    )

    # 프롬프트
    parser.add_argument(
        "--prompt", "-p",
        help="인페인팅 프롬프트 (positive)",
    )
    parser.add_argument(
        "--negative-prompt", "-np",
        default="",
        help="네거티브 프롬프트",
    )

    # 좌표
    parser.add_argument(
        "--seed-points",
        help='시드 포인트 좌표 JSON, 예: "[[320, 240]]"',
    )
    parser.add_argument(
        "--seed-box",
        help='시드 박스 좌표 JSON, 예: "[100, 100, 500, 400]"',
    )

    # 프레임 범위
    parser.add_argument(
        "--frame-range",
        help='처리할 프레임 범위 JSON, 예: "[0, 100]"',
    )

    # JSON 입력 (위 인자 대신 사용)
    parser.add_argument(
        "--json-input",
        help="전체 입력을 JSON 문자열로 제공",
    )

    # 설정
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="설정 파일 경로 (기본: config.yaml)",
    )
    parser.add_argument(
        "--backend",
        choices=["cogvideox", "wan21"],
        help="인페인팅 백엔드 (config.yaml 설정을 오버라이드)",
    )

    # 기타
    parser.add_argument(
        "--object-id",
        type=int,
        default=1,
        help="트래킹 객체 ID (기본: 1)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 로깅 활성화",
    )

    return parser.parse_args()


def main() -> int:
    """메인 함수."""
    args = parse_args()

    # 로깅 설정
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # 입력 파일 확인
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"입력 파일을 찾을 수 없습니다: {args.input}")
        return 1

    # 프롬프트 또는 JSON 입력 확인
    if args.json_input is None and args.prompt is None:
        logger.error("--prompt 또는 --json-input을 지정해야 합니다.")
        return 1

    if args.json_input is None:
        if args.seed_points is None and args.seed_box is None:
            logger.error("--seed-points 또는 --seed-box를 지정해야 합니다.")
            return 1

    # 설정 로드
    config = load_config(args.config, PROJECT_ROOT)

    # 백엔드 오버라이드
    if args.backend:
        config.inpainting.backend = args.backend
        logger.info(f"인페인팅 백엔드 오버라이드: {args.backend}")

    # 파이프라인 실행
    orchestrator = PipelineOrchestrator(config)

    try:
        result = orchestrator.run(
            input_video=args.input,
            output_video=args.output,
            prompt=args.prompt,
            seed_points=args.seed_points,
            seed_box=args.seed_box,
            frame_range=args.frame_range,
            negative_prompt=args.negative_prompt,
            object_id=args.object_id,
            json_input=args.json_input,
        )
        logger.info(f"성공! 출력 파일: {result}")
        return 0

    except Exception as e:
        logger.error(f"파이프라인 실패: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
