"""
griptape_nodes/compositor_node.py
Node 06: Alpha Composite & Audio Merge for Griptape Nodes Desktop.
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


class VideoCompositorNode(DataNode):
    """
    Node 06: Alpha Composite & Assembly
    원본 비디오와 생성된 인페인팅 프레임을 페더링 마스크를 이용해
    픽셀 단위 알파 블렌딩하고, FFmpeg NVENC 하드웨어 가속 및
    원본 오디오 트랙을 결합하여 최종 MP4 비디오를 완성합니다.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Inputs
        self.add_parameter(
            Parameter(
                name="orig_video_path",
                type="str",
                default_value="",
                tooltip="원본 비디오 파일 경로 (오디오 트랙 및 해상도 결합용)",
                display_name="Original Video Path",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="generated_frames_dir",
                type="str",
                default_value="",
                tooltip="Node 05에서 생성된 인페인팅 프레임 폴더",
                display_name="Generated Frames Dir",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="masks_dir",
                type="str",
                default_value="",
                tooltip="Node 04에서 생성된 페더링 마스크 폴더",
                display_name="Feathered Masks Dir",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="output_video_path",
                type="str",
                default_value="",
                tooltip="최종 출력 비디오 파일 경로 (.mp4, 비워두면 자동 생성)",
                display_name="Output Video Path",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="codec",
                type="str",
                default_value="h264_nvenc",
                tooltip="비디오 인코더 ('h264_nvenc' 또는 'libx264')",
                display_name="Video Codec",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="crf",
                type="int",
                default_value=18,
                tooltip="인코딩 화질 CRF 값 (기본: 18)",
                display_name="CRF Quality",
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
                tooltip="합성 및 렌더링 완료 상태",
                display_name="Status",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        from pipeline.config import CompositeConfig
        from pipeline.node06_compositor import run_node06

        orig_video_val = self.get_parameter_value("orig_video_path")
        if hasattr(orig_video_val, "value"):
            orig_video = str(orig_video_val.value)
        else:
            orig_video = str(orig_video_val or "").strip()

        gen_frames_dir = (self.get_parameter_value("generated_frames_dir") or "").strip()
        masks_dir = (self.get_parameter_value("masks_dir") or "").strip()
        out_video = (self.get_parameter_value("output_video_path") or "").strip()
        codec = self.get_parameter_value("codec") or "h264_nvenc"
        crf = int(self.get_parameter_value("crf") or 18)

        if not orig_video or not gen_frames_dir or not masks_dir:
            raise ValueError("orig_video_path, generated_frames_dir, and masks_dir are all required.")

        if not out_video:
            out_video = str(Path(tempfile.gettempdir()) / "video_mask_inpainted_output.mp4")

        cfg = CompositeConfig(
            video_codec=codec,
            crf=crf,
        )

        final_path = run_node06(
            orig_video=orig_video,
            gen_frames_dir=gen_frames_dir,
            masks_dir=masks_dir,
            output_path=out_video,
            config=cfg,
        )

        self.set_parameter_value("final_video_path", final_path)
        self.set_parameter_value("status", f"Successfully rendered: {final_path}")
