"""
griptape_nodes/video_dit_node.py
Node 05: Generative DiT Inpainting for Griptape Nodes Desktop.
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


class VideoDiTInpainterNode(DataNode):
    """
    Node 05: Generative DiT Inpainting
    CogVideoX (FP8) 또는 Wan2.1 확산 모델을 사용하여 마스킹된 영역을
    프롬프트에 맞게 비디오 단위로 자연스럽게 인페인팅합니다.
    16GB VRAM 제약을 위해 CPU Offload와 VAE Tiling을 필수로 활성화합니다.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Inputs
        self.add_parameter(
            Parameter(
                name="frames_dir",
                type="str",
                default_value="",
                tooltip="원본 비디오 프레임 폴더 경로",
                display_name="Frames Directory",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="masks_dir",
                type="str",
                default_value="",
                tooltip="페더링 마스크 시퀀스 폴더 경로",
                display_name="Masks Directory",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="prompt",
                type="str",
                default_value="clean background, seamless texture, ultra realistic",
                tooltip="인페인팅 대상 프롬프트 (Positive Prompt)",
                display_name="Inpainting Prompt",
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
                name="backend",
                type="str",
                default_value="cogvideox",
                tooltip="인페인팅 백엔드 모델 ('cogvideox' 또는 'wan2.1')",
                display_name="Model Backend",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="chunk_size",
                type="int",
                default_value=49,
                tooltip="슬라이딩 윈도우 청크 크기 (기본: 49 프레임)",
                display_name="Chunk Size",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="guidance_scale",
                type="float",
                default_value=6.0,
                tooltip="CFG 가이던스 스케일 (기본: 6.0)",
                display_name="Guidance Scale",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="num_inference_steps",
                type="int",
                default_value=30,
                tooltip="디노이징 스텝 수 (기본: 30)",
                display_name="Inference Steps",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="custom_output_dir",
                type="str",
                default_value="",
                tooltip="생성 프레임 저장 디렉토리 (비워두면 자동 생성)",
                display_name="Custom Output Dir",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # Outputs
        self.add_parameter(
            Parameter(
                name="generated_frames_dir",
                type="str",
                tooltip="인페인팅 생성된 프레임 시퀀스 폴더 경로",
                display_name="Generated Frames Dir",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="generated_count",
                type="int",
                tooltip="생성된 프레임 수",
                display_name="Generated Count",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="vram_status",
                type="str",
                tooltip="DiT 언로드 및 VRAM Flush 상태",
                display_name="VRAM Status",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        from pipeline.config import InpaintingConfig
        from pipeline.node05_dit_inpainter import run_node05
        from pipeline.vram_manager import flush_vram

        # Pre-flush VRAM before DiT allocation
        flush_vram()

        frames_dir = (self.get_parameter_value("frames_dir") or "").strip()
        masks_dir = (self.get_parameter_value("masks_dir") or "").strip()
        if not frames_dir or not masks_dir:
            raise ValueError("Both frames_dir and masks_dir are required.")

        prompt = self.get_parameter_value("prompt") or "clean background"
        neg_prompt = self.get_parameter_value("negative_prompt") or ""
        backend = (self.get_parameter_value("backend") or "cogvideox").strip().lower()
        chunk_size = int(self.get_parameter_value("chunk_size") or 49)
        guidance = float(self.get_parameter_value("guidance_scale") or 6.0)
        steps = int(self.get_parameter_value("num_inference_steps") or 30)
        custom_dir = self.get_parameter_value("custom_output_dir") or ""

        if custom_dir:
            out_dir = Path(custom_dir)
        else:
            out_dir = Path(tempfile.mkdtemp(prefix="gt_generated_frames_"))
        out_dir.mkdir(parents=True, exist_ok=True)

        inpainting_cfg = InpaintingConfig(
            backend=backend,
            chunk_size=chunk_size,
        )
        # Update active backend parameters
        active_sub_cfg = inpainting_cfg.active_config
        active_sub_cfg.guidance_scale = guidance
        active_sub_cfg.num_inference_steps = steps
        if neg_prompt:
            active_sub_cfg.negative_prompt = neg_prompt

        try:
            gen_dir = run_node05(
                frames_dir=frames_dir,
                masks_dir=masks_dir,
                prompt=prompt,
                output_dir=str(out_dir),
                config=inpainting_cfg,
            )
        finally:
            flush_vram()

        gen_files = list(Path(gen_dir).glob("*.png"))
        self.set_parameter_value("generated_frames_dir", str(gen_dir))
        self.set_parameter_value("generated_count", len(gen_files))
        self.set_parameter_value("vram_status", "DiT Finished & VRAM Flushed")
