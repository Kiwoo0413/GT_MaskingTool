"""
pipeline/node01_prompt_parser.py
Node 01: 프롬프트 & 인텐트 파서.

사용자의 자연어 명령 또는 구조화된 입력을 파싱하여
파이프라인 실행에 필요한 구조화된 데이터를 생성합니다.

실행 엔진: CPU (VRAM 미사용)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class ParsedIntent:
    """파싱된 사용자 의도.

    Attributes
    ----------
    positive_prompt : str
        인페인팅 생성 프롬프트 (예: "a gravel road with green grass").
    negative_prompt : str
        네거티브 프롬프트.
    seed_coordinates : list
        초기 추적 키포인트 ``[[x, y], ...]`` 또는 바운딩 박스
        ``[x1, y1, x2, y2]``.
    coordinate_type : str
        좌표 유형: ``"point"`` 또는 ``"box"``.
    coordinate_labels : list[int]
        포인트 레이블 (1=positive, 0=negative). 포인트 타입에서만 사용.
    frame_range : tuple[int, int]
        편집 시작/종료 프레임 인덱스 ``(start, end)``.
        ``None``이면 전체 비디오.
    object_id : int
        트래킹 대상 객체 ID.
    """
    positive_prompt: str = ""
    negative_prompt: str = ""
    seed_coordinates: list = field(default_factory=list)
    coordinate_type: Literal["point", "box"] = "point"
    coordinate_labels: List[int] = field(default_factory=lambda: [1])
    frame_range: Optional[Tuple[int, int]] = None
    object_id: int = 1


def parse_json_input(json_str: str) -> ParsedIntent:
    """
    JSON 문자열에서 ParsedIntent를 파싱합니다.

    Expected JSON format::

        {
            "positive_prompt": "a gravel road",
            "negative_prompt": "blurry, distorted",
            "seed_coordinates": [[320, 240]],
            "coordinate_type": "point",
            "coordinate_labels": [1],
            "frame_range": [0, 100],
            "object_id": 1
        }

    Parameters
    ----------
    json_str : str
        JSON 문자열.

    Returns
    -------
    ParsedIntent
    """
    data = json.loads(json_str)
    intent = ParsedIntent()

    if "positive_prompt" in data:
        intent.positive_prompt = str(data["positive_prompt"])
    if "negative_prompt" in data:
        intent.negative_prompt = str(data["negative_prompt"])
    if "seed_coordinates" in data:
        intent.seed_coordinates = data["seed_coordinates"]
    if "coordinate_type" in data:
        ctype = data["coordinate_type"]
        if ctype in ("point", "box"):
            intent.coordinate_type = ctype
    if "coordinate_labels" in data:
        intent.coordinate_labels = [int(l) for l in data["coordinate_labels"]]
    if "frame_range" in data:
        fr = data["frame_range"]
        if fr is not None and len(fr) == 2:
            intent.frame_range = (int(fr[0]), int(fr[1]))
    if "object_id" in data:
        intent.object_id = int(data["object_id"])

    return intent


def parse_cli_args(
    prompt: str,
    seed_points: Optional[str] = None,
    seed_box: Optional[str] = None,
    frame_range: Optional[str] = None,
    negative_prompt: str = "",
    object_id: int = 1,
) -> ParsedIntent:
    """
    CLI 인자에서 ParsedIntent를 생성합니다.

    Parameters
    ----------
    prompt : str
        인페인팅 positive 프롬프트.
    seed_points : str, optional
        포인트 좌표 JSON 문자열, 예: ``"[[320, 240]]"``.
    seed_box : str, optional
        박스 좌표 JSON 문자열, 예: ``"[100, 100, 500, 400]"``.
    frame_range : str, optional
        프레임 범위 JSON 문자열, 예: ``"[0, 100]"``.
    negative_prompt : str
        네거티브 프롬프트.
    object_id : int
        객체 ID.

    Returns
    -------
    ParsedIntent
    """
    intent = ParsedIntent(
        positive_prompt=prompt,
        negative_prompt=negative_prompt,
        object_id=object_id,
    )

    if seed_points is not None:
        coords = json.loads(seed_points)
        intent.seed_coordinates = coords
        intent.coordinate_type = "point"
        # 기본 레이블: 모든 포인트가 positive
        intent.coordinate_labels = [1] * len(coords)
    elif seed_box is not None:
        coords = json.loads(seed_box)
        intent.seed_coordinates = coords
        intent.coordinate_type = "box"
    else:
        raise ValueError(
            "시드 좌표가 필요합니다. --seed-points 또는 --seed-box를 지정하세요."
        )

    if frame_range is not None:
        fr = json.loads(frame_range)
        if len(fr) == 2:
            intent.frame_range = (int(fr[0]), int(fr[1]))

    return intent


def validate_intent(intent: ParsedIntent) -> None:
    """
    ParsedIntent의 유효성을 검사합니다.

    Parameters
    ----------
    intent : ParsedIntent

    Raises
    ------
    ValueError
        유효성 검사 실패 시.
    """
    if not intent.positive_prompt:
        raise ValueError("인페인팅 프롬프트(positive_prompt)가 비어 있습니다.")

    if not intent.seed_coordinates:
        raise ValueError("시드 좌표(seed_coordinates)가 비어 있습니다.")

    if intent.coordinate_type == "point":
        for pt in intent.seed_coordinates:
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                raise ValueError(
                    f"포인트 좌표는 [x, y] 형식이어야 합니다: {pt}"
                )
        if len(intent.coordinate_labels) != len(intent.seed_coordinates):
            raise ValueError(
                f"포인트 수({len(intent.seed_coordinates)})와 "
                f"레이블 수({len(intent.coordinate_labels)})가 일치하지 않습니다."
            )
    elif intent.coordinate_type == "box":
        if len(intent.seed_coordinates) != 4:
            raise ValueError(
                f"박스 좌표는 [x1, y1, x2, y2] 형식이어야 합니다: "
                f"{intent.seed_coordinates}"
            )

    if intent.frame_range is not None:
        start, end = intent.frame_range
        if start < 0 or end < start:
            raise ValueError(
                f"유효하지 않은 프레임 범위: ({start}, {end})"
            )

    logger.info(f"Intent 유효성 검사 통과: {intent}")


def run_node01(
    prompt: Optional[str] = None,
    seed_points: Optional[str] = None,
    seed_box: Optional[str] = None,
    frame_range: Optional[str] = None,
    negative_prompt: str = "",
    object_id: int = 1,
    json_input: Optional[str] = None,
) -> ParsedIntent:
    """
    Node 01을 실행합니다.

    Parameters
    ----------
    prompt : str, optional
        인페인팅 프롬프트 (json_input 미사용 시 필수).
    seed_points : str, optional
        포인트 좌표 JSON.
    seed_box : str, optional
        박스 좌표 JSON.
    frame_range : str, optional
        프레임 범위 JSON.
    negative_prompt : str
        네거티브 프롬프트.
    object_id : int
        객체 ID.
    json_input : str, optional
        JSON 전체 입력 문자열 (이 값이 있으면 다른 인자 무시).

    Returns
    -------
    ParsedIntent
    """
    logger.info("=" * 60)
    logger.info("Node 01: Prompt & Intent Parser 시작")
    logger.info("=" * 60)

    if json_input is not None:
        intent = parse_json_input(json_input)
    else:
        if prompt is None:
            raise ValueError("prompt 또는 json_input이 필요합니다.")
        intent = parse_cli_args(
            prompt=prompt,
            seed_points=seed_points,
            seed_box=seed_box,
            frame_range=frame_range,
            negative_prompt=negative_prompt,
            object_id=object_id,
        )

    validate_intent(intent)

    logger.info(f"  Positive Prompt : {intent.positive_prompt}")
    logger.info(f"  Negative Prompt : {intent.negative_prompt}")
    logger.info(f"  Coordinate Type : {intent.coordinate_type}")
    logger.info(f"  Seed Coordinates: {intent.seed_coordinates}")
    logger.info(f"  Frame Range     : {intent.frame_range}")
    logger.info(f"  Object ID       : {intent.object_id}")
    logger.info("Node 01 완료")

    return intent
