"""
griptape_nodes/frame_decomposer_node.py
Node 02: Frame Decomposer & Preprocessor for Griptape Nodes Desktop.
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


class VideoFrameDecomposerNode(DataNode):
    """
    Node 02: Frame Decomposer & Preprocessor
    입력 비디오를 지정된 해상도(기본 720p)와 FPS로 정규화하여
    개별 프레임 시퀀스로 디컴포즈합니다.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Inputs
        self.add_parameter(
            Parameter(
                name="video_path",
                type="str",
                default_value="",
                tooltip="입력 비디오 파일의 절대 경로 또는 URL",
                display_name="Video Path",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="target_resolution",
                type="int",
                default_value=720,
                tooltip="최대 해상도 높이 (기본: 720p, VRAM 절약)",
                display_name="Target Resolution",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="normalize_fps",
                type="bool",
                default_value=True,
                tooltip="FPS를 24fps로 정규화할지 여부",
                display_name="Normalize to 24 FPS",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="custom_output_dir",
                type="str",
                default_value="",
                tooltip="프레임 저장 디렉토리 (비워두면 자동 생성)",
                display_name="Custom Output Dir",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # Outputs
        self.add_parameter(
            Parameter(
                name="frames_dir",
                type="str",
                tooltip="추출된 프레임 시퀀스 폴더 경로",
                display_name="Frames Directory",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="fps",
                type="float",
                tooltip="출력 비디오 FPS",
                display_name="FPS",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="total_frames",
                type="int",
                tooltip="추출된 총 프레임 수",
                display_name="Total Frames",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="resolution",
                type="str",
                tooltip="프레임 해상도 (Width x Height)",
                display_name="Resolution",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        from pipeline.node02_frame_decomposer import run_node02

        video_path_val = self.get_parameter_value("video_path")
        # Handle artifact objects if passed
        if hasattr(video_path_val, "value"):
            video_path = str(video_path_val.value)
        else:
            video_path = str(video_path_val or "").strip()

        if not video_path:
            raise ValueError("video_path is required.")

        target_res = int(self.get_parameter_value("target_resolution") or 720)
        norm_fps = bool(self.get_parameter_value("normalize_fps"))
        custom_dir = self.get_parameter_value("custom_output_dir") or ""

        if custom_dir:
            out_dir = Path(custom_dir)
        else:
            out_dir = Path(tempfile.mkdtemp(prefix="gt_frames_"))
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = run_node02(
            video_path=video_path,
            output_dir=str(out_dir),
            target_resolution=target_res,
            normalize_fps=norm_fps,
        )

        self.set_parameter_value("frames_dir", meta.frames_dir)
        self.set_parameter_value("fps", float(meta.fps))
        self.set_parameter_value("total_frames", int(meta.frame_count))
        self.set_parameter_value("resolution", f"{meta.width}x{meta.height}")
