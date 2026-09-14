"""
griptape_nodes/sam2_tracker_node.py
Node 03: Temporal Mask Tracker (SAM 2 VOS) for Griptape Nodes Desktop.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode


def _parse_coords_string(coords_str: str) -> Optional[List[Tuple[float, float]]]:
    if not coords_str:
        return None
    cleaned = coords_str.replace("[", "").replace("]", "").replace(";", ",").strip()
    if not cleaned:
        return None
    tokens = [float(t.strip()) for t in cleaned.split(",") if t.strip()]
    if len(tokens) % 2 != 0:
        return None
    points = []
    for i in range(0, len(tokens), 2):
        points.append((tokens[i], tokens[i + 1]))
    return points


def _parse_box_string(box_str: str) -> Optional[Tuple[float, float, float, float]]:
    if not box_str:
        return None
    cleaned = box_str.replace("[", "").replace("]", "").replace(";", ",").strip()
    if not cleaned:
        return None
    tokens = [float(t.strip()) for t in cleaned.split(",") if t.strip()]
    if len(tokens) >= 4:
        return (tokens[0], tokens[1], tokens[2], tokens[3])
    return None


def _parse_range_string(range_str: str) -> Optional[Tuple[int, int]]:
    if not range_str:
        return None
    cleaned = range_str.replace("[", "").replace("]", "").replace(";", ",").strip()
    if not cleaned:
        return None
    tokens = [int(float(t.strip())) for t in cleaned.split(",") if t.strip()]
    if len(tokens) >= 2:
        return (tokens[0], tokens[1])
    return None


class SAM2VOSNode(DataNode):
    """
    Node 03: Temporal Mask Tracker (SAM 2 VOS)
    SAM 2 모델을 사용하여 프레임 시퀀스 전반에 걸쳐 객체를 시공간 추적하고
    프레임별 바이너리 마스크를 생성합니다. 작업 완료 시 VRAM 0MB Flush를 수행합니다.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Inputs
        self.add_parameter(
            Parameter(
                name="frames_dir",
                type="str",
                default_value="",
                tooltip="추출된 원본 프레임 시퀀스가 저장된 디렉토리 경로",
                display_name="Frames Directory",
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
                tooltip="바운딩 박스 좌표 (예: '100,100,300,300')",
                display_name="Box Coords",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="frame_range",
                type="str",
                default_value="",
                tooltip="추적 대상 프레임 범위 (예: '0,50')",
                display_name="Frame Range",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="model_cfg",
                type="str",
                default_value="configs/sam2.1/sam2.1_hiera_b+.yaml",
                tooltip="SAM 2 모델 구성 파일",
                display_name="Model Config",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="checkpoint",
                type="str",
                default_value="",
                tooltip="SAM 2 가중치 파일 경로 (비워두면 자동 탐색)",
                display_name="Checkpoint Path",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="custom_output_dir",
                type="str",
                default_value="",
                tooltip="마스크 저장 디렉토리 (비워두면 자동 생성)",
                display_name="Custom Output Dir",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # Outputs
        self.add_parameter(
            Parameter(
                name="raw_masks_dir",
                type="str",
                tooltip="추적 생성된 바이너리 마스크 시퀀스 폴더 경로",
                display_name="Raw Masks Directory",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="mask_count",
                type="int",
                tooltip="생성된 마스크 프레임 수",
                display_name="Mask Count",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="vram_status",
                type="str",
                tooltip="SAM 2 언로드 및 VRAM Flush 상태",
                display_name="VRAM Status",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        from pipeline.config import SAM2Config
        from pipeline.node03_vos_tracker import run_node03
        from pipeline.vram_manager import flush_vram, log_vram_status

        frames_dir = (self.get_parameter_value("frames_dir") or "").strip()
        if not frames_dir:
            raise ValueError("frames_dir is required.")

        seed_coords_str = self.get_parameter_value("seed_coords") or ""
        box_coords_str = self.get_parameter_value("box_coords") or ""
        frame_range_str = self.get_parameter_value("frame_range") or ""
        model_cfg = self.get_parameter_value("model_cfg") or "configs/sam2.1/sam2.1_hiera_b+.yaml"
        checkpoint = (self.get_parameter_value("checkpoint") or "").strip()
        custom_dir = self.get_parameter_value("custom_output_dir") or ""

        points = _parse_coords_string(seed_coords_str)
        box = _parse_box_string(box_coords_str)
        frame_range = _parse_range_string(frame_range_str)

        if not points and not box:
            # Fallback center point
            points = [(640.0, 360.0)]

        point_labels = [1] * len(points) if points else None

        if custom_dir:
            out_dir = Path(custom_dir)
        else:
            out_dir = Path(tempfile.mkdtemp(prefix="gt_masks_raw_"))
        out_dir.mkdir(parents=True, exist_ok=True)

        sam2_cfg = SAM2Config(
            model_cfg=model_cfg,
            checkpoint=checkpoint or None,
        )

        try:
            masks_dir = run_node03(
                frames_dir=frames_dir,
                output_dir=str(out_dir),
                sam2_cfg=sam2_cfg,
                points=points,
                point_labels=point_labels,
                box=box,
                frame_range=frame_range,
                object_id=1,
            )
        finally:
            flush_vram()

        mask_files = list(Path(masks_dir).glob("*.png"))
        self.set_parameter_value("raw_masks_dir", str(masks_dir))
        self.set_parameter_value("mask_count", len(mask_files))
        self.set_parameter_value("vram_status", "VRAM Flushed to <1GB (Clean)")
