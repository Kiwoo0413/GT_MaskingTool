"""
griptape_nodes/pipeline_all_in_one_node.py
All-in-One Video Mask Inpainting Pipeline Node for Griptape Nodes Desktop.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode


class VideoMaskInpaintingAllInOneNode(DataNode):
    """
    Video Mask Inpainting Pipeline (All-in-One)
    비디오 파일, 편집 지시문/프롬프트, 시드 키포인트를 입력받아
    내부에서 6단계 파이프라인(분해 -> SAM 2 추적 -> 페더링 -> DiT 생성 -> 합성)을
    16GB VRAM 격리 원칙 하에 한 번에 완수하는 올인원 노드입니다.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Inputs
        self.add_parameter(
            Parameter(
                name="input_video",
                type="str",
                default_value="",
                tooltip="입력 비디오 파일의 절대 경로",
                display_name="Input Video Path",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
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
                name="prompt",
                type="str",
                default_value="clean background, seamless texture, ultra realistic",
                tooltip="인페인팅 대상 프롬프트 (Positive Prompt)",
                display_name="Prompt",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="negative_prompt",
                type="str",
                default_value="",
                tooltip="네거티브 프롬프트",
                display_name="Negative Prompt",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="seed_coords",
                type="str",
                default_value="",
                tooltip="시드 키포인트 좌표 (예: '640,360')",
                display_name="Seed Coords",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="box_coords",
                type="str",
                default_value="",
                tooltip="바운딩 박스 (예: '100,100,300,300')",
                display_name="Box Coords",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="frame_range",
                type="str",
                default_value="",
                tooltip="편집 대상 프레임 범위 (예: '0,50')",
                display_name="Frame Range",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="backend",
                type="str",
                default_value="cogvideox",
                tooltip="인페인팅 백엔드 ('cogvideox' 또는 'wan2.1')",
                display_name="Backend",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="output_video",
                type="str",
                default_value="",
                tooltip="최종 저장 비디오 경로 (.mp4, 비워두면 자동 생성)",
                display_name="Output Video Path",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # Outputs
        self.add_parameter(
            Parameter(
                name="final_video_path",
                type="str",
                tooltip="완성된 최종 비디오 파일 경로",
                display_name="Final Video Path",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="status",
                type="str",
                tooltip="파이프라인 전체 완료 상태",
                display_name="Status",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        from pipeline.config import PipelineConfig, load_config
        from pipeline.orchestrator import PipelineOrchestrator

        video_val = self.get_parameter_value("input_video")
        if hasattr(video_val, "value"):
            input_video = str(video_val.value)
        else:
            input_video = str(video_val or "").strip()

        if not input_video:
            raise ValueError("input_video is required.")

        instruction = (self.get_parameter_value("instruction") or "").strip()
        prompt = (self.get_parameter_value("prompt") or "").strip()
        neg_prompt = (self.get_parameter_value("negative_prompt") or "").strip()
        seed_coords = (self.get_parameter_value("seed_coords") or "").strip()
        box_coords = (self.get_parameter_value("box_coords") or "").strip()
        frame_range = (self.get_parameter_value("frame_range") or "").strip()
        backend = (self.get_parameter_value("backend") or "cogvideox").strip().lower()
        out_video = (self.get_parameter_value("output_video") or "").strip()

        if not out_video:
            out_video = str(Path(tempfile.gettempdir()) / "video_mask_inpainted_final.mp4")

        # Load or create pipeline config
        cfg_file = WORKSPACE_ROOT / "config.yaml"
        if cfg_file.exists():
            config = load_config(str(cfg_file), WORKSPACE_ROOT)
        else:
            config = PipelineConfig()

        config.inpainting.backend = backend
        orchestrator = PipelineOrchestrator(config=config)

        final_path = orchestrator.run(
            input_video=input_video,
            output_video=out_video,
            prompt=instruction or prompt or "clean background",
            seed_points=seed_coords or None,
            seed_box=box_coords or None,
            frame_range=frame_range or None,
            negative_prompt=neg_prompt,
        )

        self.set_parameter_value("final_video_path", final_path)
        self.set_parameter_value("status", f"Complete: {final_path}")
