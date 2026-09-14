"""
griptape_nodes/prompt_parser_node.py
Node 01: Prompt & Intent Parser for Griptape Nodes Desktop.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure pipeline module is accessible
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode


class VideoMaskPromptParserNode(DataNode):
    """
    Node 01: Prompt & Intent Parser
    사용자의 자연어 편집 명령이나 수동 좌표를 파싱하여
    인페인팅 프롬프트, 시드 키포인트, 바운딩 박스, 대상 프레임 범위를 추출합니다.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Inputs
        self.add_parameter(
            Parameter(
                name="instruction",
                type="str",
                default_value="",
                tooltip="자연어 편집 지시문 (예: '3초 부근에 지나가는 빨간 차를 지우고 도로로 채워줘')",
                display_name="Instruction",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="default_prompt",
                type="str",
                default_value="clean background, seamless texture, highly detailed",
                tooltip="자연어에서 프롬프트가 없을 때 사용할 기본 인페인팅 프롬프트",
                display_name="Default Prompt",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="seed_coords",
                type="str",
                default_value="",
                tooltip="수동 시드 좌표 (예: '640,360' 또는 '[[640, 360]]')",
                display_name="Seed Coordinates",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="box_coords",
                type="str",
                default_value="",
                tooltip="바운딩 박스 (예: '100,100,300,300')",
                display_name="Box Coordinates",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="frame_range",
                type="str",
                default_value="",
                tooltip="편집 프레임 범위 (예: '0,50')",
                display_name="Frame Range",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # Outputs
        self.add_parameter(
            Parameter(
                name="prompt",
                type="str",
                tooltip="파싱된 인페인팅 대상 프롬프트",
                display_name="Inpainting Prompt",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="negative_prompt",
                type="str",
                tooltip="네거티브 프롬프트",
                display_name="Negative Prompt",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="out_seed_coords",
                type="str",
                tooltip="추출된 시드 좌표",
                display_name="Seed Coords Out",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="out_box_coords",
                type="str",
                tooltip="추출된 바운딩 박스",
                display_name="Box Coords Out",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="out_frame_range",
                type="str",
                tooltip="추출된 프레임 범위",
                display_name="Frame Range Out",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        from pipeline.node01_prompt_parser import run_node01

        instruction = self.get_parameter_value("instruction") or ""
        default_prompt = self.get_parameter_value("default_prompt") or ""
        seed_coords = self.get_parameter_value("seed_coords") or ""
        box_coords = self.get_parameter_value("box_coords") or ""
        frame_range = self.get_parameter_value("frame_range") or ""

        intent = run_node01(
            instruction=instruction,
            default_prompt=default_prompt,
            seed_coords=seed_coords if seed_coords else None,
            box_coords=box_coords if box_coords else None,
            frame_range=frame_range if frame_range else None,
        )

        # Convert coords to string for output ports
        seed_str = ""
        if intent.seed_points:
            seed_str = ",".join(f"{pt[0]},{pt[1]}" for pt in intent.seed_points)

        box_str = ""
        if intent.seed_box:
            box_str = ",".join(str(int(c)) for c in intent.seed_box)

        range_str = ""
        if intent.frame_range:
            range_str = f"{intent.frame_range[0]},{intent.frame_range[1]}"

        self.set_parameter_value("prompt", intent.target_prompt)
        self.set_parameter_value("negative_prompt", intent.negative_prompt)
        self.set_parameter_value("out_seed_coords", seed_str)
        self.set_parameter_value("out_box_coords", box_str)
        self.set_parameter_value("out_frame_range", range_str)
