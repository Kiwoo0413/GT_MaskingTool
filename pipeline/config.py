"""
pipeline/config.py
파이프라인 설정 관리 모듈.

YAML 설정 파일을 파싱하고 dataclass 기반의 타입 안전한
설정 객체를 제공합니다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Dataclass 정의
# ---------------------------------------------------------------------------

@dataclass
class WorkspaceConfig:
    """워크스페이스 경로 설정."""
    base_dir: str = "./workspace"
    frames_dir: str = "frames"
    masks_raw_dir: str = "masks_raw"
    masks_feathered_dir: str = "masks_feathered"
    generated_frames_dir: str = "generated_frames"
    composite_dir: str = "composite"

    def resolve(self, project_root: Path) -> None:
        """상대 경로를 절대 경로로 변환."""
        base = Path(self.base_dir)
        if not base.is_absolute():
            base = project_root / base
        self.base_dir = str(base)

    def get_path(self, subdir: str) -> Path:
        """서브디렉토리의 전체 경로를 반환하고, 없으면 생성."""
        p = Path(self.base_dir) / subdir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def frames_path(self) -> Path:
        return self.get_path(self.frames_dir)

    @property
    def masks_raw_path(self) -> Path:
        return self.get_path(self.masks_raw_dir)

    @property
    def masks_feathered_path(self) -> Path:
        return self.get_path(self.masks_feathered_dir)

    @property
    def generated_frames_path(self) -> Path:
        return self.get_path(self.generated_frames_dir)

    @property
    def composite_path(self) -> Path:
        return self.get_path(self.composite_dir)


@dataclass
class PreprocessingConfig:
    """비디오 전처리 설정 (Node 02)."""
    max_resolution: int = 1280
    target_short_side: int = 720
    normalize_fps: bool = True
    target_fps: int = 24
    frame_format: Literal["jpg", "png"] = "jpg"
    jpeg_quality: int = 95


@dataclass
class SAM2Config:
    """SAM 2 VOS 트래킹 설정 (Node 03)."""
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    checkpoint: str = "checkpoints/sam2.1_hiera_base_plus.pt"
    checkpoint_url: str = (
        "https://dl.fbaipublicfiles.com/segment_anything_2/"
        "092824/sam2.1_hiera_base_plus.pt"
    )
    auto_download: bool = True
    device: str = "cuda"
    dtype: str = "bfloat16"


@dataclass
class MaskRefinementConfig:
    """마스크 후처리 설정 (Node 04)."""
    dilation_kernel_size: int = 7
    dilation_iterations: int = 2
    gaussian_kernel_size: int = 11
    gaussian_sigma: float = 0.0
    num_workers: int = 4


@dataclass
class CogVideoXConfig:
    """CogVideoX 인페인팅 설정."""
    model_id: str = "THUDM/CogVideoX-5b-I2V"
    dtype: str = "float16"
    quantization: Literal["fp8", "nf4", "none"] = "fp8"
    num_inference_steps: int = 50
    guidance_scale: float = 6.0
    chunk_frames: int = 49
    enable_cpu_offload: bool = True
    enable_vae_tiling: bool = True


@dataclass
class Wan21Config:
    """Wan2.1 인페인팅 설정."""
    model_id: str = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    dtype: str = "float16"
    quantization: Literal["nf4", "fp8", "none"] = "nf4"
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    chunk_frames: int = 49
    enable_cpu_offload: bool = True
    enable_vae_tiling: bool = True


@dataclass
class InpaintingConfig:
    """DiT 인페인팅 설정 (Node 05)."""
    backend: Literal["cogvideox", "wan21"] = "cogvideox"
    cogvideox: CogVideoXConfig = field(default_factory=CogVideoXConfig)
    wan21: Wan21Config = field(default_factory=Wan21Config)

    @property
    def active_config(self) -> CogVideoXConfig | Wan21Config:
        """현재 선택된 백엔드의 설정을 반환."""
        if self.backend == "cogvideox":
            return self.cogvideox
        return self.wan21


@dataclass
class AssemblyConfig:
    """최종 조립 설정 (Node 06)."""
    encoder: str = "h264_nvenc"
    pixel_format: str = "yuv420p"
    crf: int = 18
    preset: str = "p4"


@dataclass
class ErrorHandlingConfig:
    """에러 처리 설정."""
    max_oom_retries: int = 3
    resolution_scale_factor: float = 0.8
    min_resolution: int = 480


@dataclass
class VRAMConfig:
    """VRAM 관리 설정."""
    max_idle_vram_mb: int = 1024
    force_flush: bool = True


@dataclass
class PipelineConfig:
    """파이프라인 전체 설정."""
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    sam2: SAM2Config = field(default_factory=SAM2Config)
    mask_refinement: MaskRefinementConfig = field(default_factory=MaskRefinementConfig)
    inpainting: InpaintingConfig = field(default_factory=InpaintingConfig)
    assembly: AssemblyConfig = field(default_factory=AssemblyConfig)
    error_handling: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)
    vram: VRAMConfig = field(default_factory=VRAMConfig)

    def resolve_paths(self, project_root: Path) -> None:
        """모든 상대 경로를 프로젝트 루트 기준 절대 경로로 변환."""
        self.workspace.resolve(project_root)

        # SAM 2 체크포인트 경로 해석
        ckpt = Path(self.sam2.checkpoint)
        if not ckpt.is_absolute():
            self.sam2.checkpoint = str(project_root / ckpt)


# ---------------------------------------------------------------------------
# 설정 로더
# ---------------------------------------------------------------------------

def _apply_dict(target, source: dict) -> None:
    """딕셔너리 값을 dataclass 인스턴스에 재귀적으로 적용."""
    for key, value in source.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _apply_dict(current, value)
        else:
            setattr(target, key, value)


def load_config(
    config_path: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> PipelineConfig:
    """
    YAML 설정 파일에서 PipelineConfig를 로드합니다.

    Parameters
    ----------
    config_path : str, optional
        설정 파일 경로. None이면 기본값 사용.
    project_root : Path, optional
        프로젝트 루트 디렉토리. None이면 현재 작업 디렉토리 사용.

    Returns
    -------
    PipelineConfig
        파싱된 파이프라인 설정 객체.
    """
    cfg = PipelineConfig()

    if project_root is None:
        project_root = Path.cwd()

    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = project_root / config_file
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            if raw:
                _apply_dict(cfg, raw)

    cfg.resolve_paths(project_root)
    return cfg
