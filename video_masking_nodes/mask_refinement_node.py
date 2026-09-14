"""
griptape_nodes/mask_refinement_node.py
Node 04: Mask Morphological Refinement for Griptape Nodes Desktop.
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


class MaskRefinementNode(DataNode):
    """
    Node 04: Mask Morphological Refinement
    바이너리 마스크 시퀀스를 CPU 멀티프로세싱으로 팽창(Dilation) 및
    가우시안 블러 페더링 처리하여 합성 경계의 아티팩트를 제거합니다.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Inputs
        self.add_parameter(
            Parameter(
                name="raw_masks_dir",
                type="str",
                default_value="",
                tooltip="SAM 2 등으로 생성된 원본 바이너리 마스크 디렉토리",
                display_name="Raw Masks Dir",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="dilation_kernel",
                type="int",
                default_value=7,
                tooltip="팽창 연산 커널 크기 (홀수, 기본 7~9)",
                display_name="Dilation Kernel",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="dilation_iterations",
                type="int",
                default_value=2,
                tooltip="팽창 반복 횟수",
                display_name="Dilation Iterations",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="blur_kernel",
                type="int",
                default_value=11,
                tooltip="가우시안 블러 커널 크기 (홀수, 기본 11)",
                display_name="Blur Kernel",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="blur_sigma",
                type="float",
                default_value=0.0,
                tooltip="가우시안 블러 시그마 (0.0은 자동 계산)",
                display_name="Blur Sigma",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="num_workers",
                type="int",
                default_value=4,
                tooltip="CPU 멀티프로세싱 병렬 스레드/워커 수",
                display_name="Num Workers",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="custom_output_dir",
                type="str",
                default_value="",
                tooltip="페더링 마스크 저장 디렉토리 (비워두면 자동 생성)",
                display_name="Custom Output Dir",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # Outputs
        self.add_parameter(
            Parameter(
                name="feathered_masks_dir",
                type="str",
                tooltip="후처리 완료된 페더링 마스크 폴더 경로",
                display_name="Feathered Masks Dir",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="processed_count",
                type="int",
                tooltip="처리된 마스크 수",
                display_name="Processed Count",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        from pipeline.config import MaskRefineConfig
        from pipeline.node04_mask_refinement import run_node04

        raw_masks_dir = (self.get_parameter_value("raw_masks_dir") or "").strip()
        if not raw_masks_dir:
            raise ValueError("raw_masks_dir is required.")

        dilation_k = int(self.get_parameter_value("dilation_kernel") or 7)
        dilation_it = int(self.get_parameter_value("dilation_iterations") or 2)
        blur_k = int(self.get_parameter_value("blur_kernel") or 11)
        blur_sig = float(self.get_parameter_value("blur_sigma") or 0.0)
        workers = int(self.get_parameter_value("num_workers") or 4)
        custom_dir = self.get_parameter_value("custom_output_dir") or ""

        if custom_dir:
            out_dir = Path(custom_dir)
        else:
            out_dir = Path(tempfile.mkdtemp(prefix="gt_masks_feathered_"))
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg = MaskRefineConfig(
            dilation_kernel_size=dilation_k,
            dilation_iterations=dilation_it,
            gaussian_kernel_size=blur_k,
            gaussian_sigma=blur_sig,
            num_workers=workers,
        )

        feathered_dir = run_node04(
            raw_masks_dir=raw_masks_dir,
            output_dir=str(out_dir),
            config=cfg,
        )

        mask_files = list(Path(feathered_dir).glob("*.png"))
        self.set_parameter_value("feathered_masks_dir", str(feathered_dir))
        self.set_parameter_value("processed_count", len(mask_files))
