"""
pipeline/node05_dit_inpainter.py
Node 05: Video DiT Inpainter.

확산 트랜스포머(Diffusion Transformer) 기반 비디오 인페인팅.
CogVideoX-5B-I2V 또는 Wan2.1-I2V 백엔드를 지원합니다.

필수 최적화: FP8/NF4 양자화, CPU Offload, Tiled VAE
실행 엔진: GPU (FP8/NF4)
필수: 작업 완료 후 VRAM 완전 해제
"""

from __future__ import annotations

import gc
import logging
import shutil
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from .config import InpaintingConfig, WorkspaceConfig
from .node01_prompt_parser import ParsedIntent
from .node02_frame_decomposer import VideoMetadata
from .vram_manager import flush_vram, log_vram_status, vram_scope, get_torch_dtype

logger = logging.getLogger(__name__)


def _load_frames_as_pil(
    frames_dir: Path,
    indices: Optional[List[int]] = None,
) -> List[Image.Image]:
    """프레임 시퀀스를 PIL 이미지 리스트로 로드합니다."""
    frame_files = sorted(frames_dir.glob("*.*"))
    frame_files = [f for f in frame_files if f.suffix.lower() in (".jpg", ".png")]

    if indices is not None:
        frame_files = [frame_files[i] for i in indices if i < len(frame_files)]

    images = []
    for fpath in frame_files:
        img = Image.open(fpath).convert("RGB")
        images.append(img)

    return images


def _load_masks_as_pil(
    masks_dir: Path,
    indices: Optional[List[int]] = None,
) -> List[Image.Image]:
    """마스크 시퀀스를 PIL 이미지 리스트로 로드합니다."""
    mask_files = sorted(masks_dir.glob("*.png"))

    if indices is not None:
        mask_files = [mask_files[i] for i in indices if i < len(mask_files)]

    masks = []
    for fpath in mask_files:
        mask = Image.open(fpath).convert("L")
        masks.append(mask)

    return masks


def _save_generated_frames(
    frames: List[Image.Image],
    output_dir: Path,
    start_idx: int = 0,
) -> None:
    """생성된 프레임을 PNG로 저장합니다."""
    for i, frame in enumerate(frames):
        fname = f"{start_idx + i:05d}.png"
        frame.save(output_dir / fname)


def _run_cogvideox_inpainting(
    frames: List[Image.Image],
    masks: List[Image.Image],
    prompt: str,
    negative_prompt: str,
    config: InpaintingConfig,
    output_dir: Path,
    chunk_start: int = 0,
) -> List[Image.Image]:
    """
    CogVideoX-5B-I2V 기반 인페인팅을 실행합니다.

    Returns
    -------
    list[Image.Image]
        생성된 프레임 리스트.
    """
    from diffusers import CogVideoXImageToVideoPipeline

    cfg = config.cogvideox
    dtype = get_torch_dtype(cfg.dtype)

    logger.info(f"  CogVideoX 모델 로드: {cfg.model_id}")
    logger.info(f"  양자화: {cfg.quantization}, dtype: {cfg.dtype}")

    # 모델 로드
    pipe_kwargs = {
        "torch_dtype": dtype,
    }

    # 양자화 설정
    if cfg.quantization == "nf4":
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )
            pipe_kwargs["quantization_config"] = quantization_config
        except ImportError:
            logger.warning("bitsandbytes 미설치 — NF4 양자화 건너뜀")

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        cfg.model_id, **pipe_kwargs
    )

    # 최적화 적용
    if cfg.enable_cpu_offload:
        pipe.enable_model_cpu_offload()
        logger.info("  CPU Offload 활성화됨")

    if cfg.enable_vae_tiling:
        pipe.vae.enable_tiling()
        logger.info("  VAE Tiling 활성화됨")

    log_vram_status("CogVideoX 모델 로드 후")

    # 추론
    # CogVideoX I2V: 첫 프레임 이미지를 조건으로 비디오 생성
    # 인페인팅 워크플로우: 마스크 영역에 프롬프트 기반 콘텐츠 생성
    try:
        # 첫 프레임을 조건 이미지로 사용
        condition_image = frames[0].copy()

        # 마스크 영역을 첫 프레임에 적용 (마스크된 영역 제거)
        mask_np = np.array(masks[0].convert("L"))
        cond_np = np.array(condition_image)
        # 마스크 영역을 주변 색상의 평균으로 대체 (단순 인페인팅 힌트)
        mask_bool = mask_np > 127
        if np.any(mask_bool):
            # 마스크 외부 영역의 평균 색상으로 채우기
            bg_color = cond_np[~mask_bool].mean(axis=0).astype(np.uint8)
            cond_np[mask_bool] = bg_color
            condition_image = Image.fromarray(cond_np)

        logger.info(
            f"  추론 시작: steps={cfg.num_inference_steps}, "
            f"guidance={cfg.guidance_scale}"
        )

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=condition_image,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            num_frames=min(len(frames), cfg.chunk_frames),
        )

        generated_frames = []
        if hasattr(output, "frames") and output.frames is not None:
            # output.frames는 보통 list[list[PIL.Image]]
            for frame_batch in output.frames:
                if isinstance(frame_batch, list):
                    generated_frames.extend(frame_batch)
                elif isinstance(frame_batch, Image.Image):
                    generated_frames.append(frame_batch)
                else:
                    # tensor인 경우
                    generated_frames.append(
                        Image.fromarray(
                            (frame_batch.cpu().numpy() * 255).astype(np.uint8)
                        )
                    )

        logger.info(f"  추론 완료: {len(generated_frames)} frames 생성됨")
        return generated_frames

    finally:
        # 모델 해제
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


def _run_wan21_inpainting(
    frames: List[Image.Image],
    masks: List[Image.Image],
    prompt: str,
    negative_prompt: str,
    config: InpaintingConfig,
    output_dir: Path,
    chunk_start: int = 0,
) -> List[Image.Image]:
    """
    Wan2.1-I2V 기반 인페인팅을 실행합니다.

    Returns
    -------
    list[Image.Image]
        생성된 프레임 리스트.
    """
    from diffusers import WanImageToVideoPipeline

    cfg = config.wan21
    dtype = get_torch_dtype(cfg.dtype)

    logger.info(f"  Wan2.1 모델 로드: {cfg.model_id}")
    logger.info(f"  양자화: {cfg.quantization}, dtype: {cfg.dtype}")

    pipe_kwargs = {
        "torch_dtype": dtype,
    }

    # NF4 양자화
    if cfg.quantization == "nf4":
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )
            pipe_kwargs["quantization_config"] = quantization_config
        except ImportError:
            logger.warning("bitsandbytes 미설치 — NF4 양자화 건너뜀")

    pipe = WanImageToVideoPipeline.from_pretrained(
        cfg.model_id, **pipe_kwargs
    )

    if cfg.enable_cpu_offload:
        pipe.enable_model_cpu_offload()
        logger.info("  CPU Offload 활성화됨")

    if cfg.enable_vae_tiling:
        pipe.vae.enable_tiling()
        logger.info("  VAE Tiling 활성화됨")

    log_vram_status("Wan2.1 모델 로드 후")

    try:
        condition_image = frames[0].copy()

        logger.info(
            f"  추론 시작: steps={cfg.num_inference_steps}, "
            f"guidance={cfg.guidance_scale}"
        )

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=condition_image,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            num_frames=min(len(frames), cfg.chunk_frames),
        )

        generated_frames = []
        if hasattr(output, "frames") and output.frames is not None:
            for frame_batch in output.frames:
                if isinstance(frame_batch, list):
                    generated_frames.extend(frame_batch)
                elif isinstance(frame_batch, Image.Image):
                    generated_frames.append(frame_batch)

        logger.info(f"  추론 완료: {len(generated_frames)} frames 생성됨")
        return generated_frames

    finally:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


def run_node05(
    intent: ParsedIntent,
    metadata: VideoMetadata,
    workspace: WorkspaceConfig,
    config: InpaintingConfig,
    max_idle_vram_mb: float = 1024.0,
) -> Path:
    """
    Node 05를 실행합니다.

    DiT 기반 비디오 인페인팅을 수행하여 마스크 영역을
    프롬프트 기반 콘텐츠로 생성합니다.

    Parameters
    ----------
    intent : ParsedIntent
        인페인팅 프롬프트 포함.
    metadata : VideoMetadata
        비디오 메타데이터.
    workspace : WorkspaceConfig
        워크스페이스 설정.
    config : InpaintingConfig
        인페인팅 설정.
    max_idle_vram_mb : float
        VRAM flush 후 최대 허용 VRAM (MB).

    Returns
    -------
    Path
        생성된 프레임 디렉토리 경로.
    """
    logger.info("=" * 60)
    logger.info("Node 05: Video DiT Inpainter 시작")
    logger.info("=" * 60)

    frames_dir = workspace.frames_path
    masks_dir = workspace.masks_feathered_path
    gen_dir = workspace.generated_frames_path

    # 기존 결과 정리
    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)

    # 프레임 & 마스크 로드
    frames = _load_frames_as_pil(frames_dir)
    masks = _load_masks_as_pil(masks_dir)

    if not frames:
        raise RuntimeError(f"프레임이 없습니다: {frames_dir}")
    if not masks:
        raise RuntimeError(f"마스크가 없습니다: {masks_dir}")

    # 프레임 수 맞추기
    min_count = min(len(frames), len(masks))
    frames = frames[:min_count]
    masks = masks[:min_count]

    logger.info(f"  입력: {min_count} frames, 백엔드: {config.backend}")

    # 청크 단위 처리
    active_cfg = config.active_config
    chunk_size = active_cfg.chunk_frames
    total_chunks = (min_count + chunk_size - 1) // chunk_size

    log_vram_status("Node 05 시작 전")

    all_generated: List[Image.Image] = []

    with vram_scope("DiT Inpainting", max_idle_vram_mb, flush_before=True, flush_after=True):
        for chunk_idx in range(total_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, min_count)
            chunk_frames = frames[start:end]
            chunk_masks = masks[start:end]

            logger.info(
                f"  청크 {chunk_idx + 1}/{total_chunks}: "
                f"frames [{start}:{end}]"
            )

            if config.backend == "cogvideox":
                gen_frames = _run_cogvideox_inpainting(
                    frames=chunk_frames,
                    masks=chunk_masks,
                    prompt=intent.positive_prompt,
                    negative_prompt=intent.negative_prompt,
                    config=config,
                    output_dir=gen_dir,
                    chunk_start=start,
                )
            elif config.backend == "wan21":
                gen_frames = _run_wan21_inpainting(
                    frames=chunk_frames,
                    masks=chunk_masks,
                    prompt=intent.positive_prompt,
                    negative_prompt=intent.negative_prompt,
                    config=config,
                    output_dir=gen_dir,
                    chunk_start=start,
                )
            else:
                raise ValueError(f"지원하지 않는 백엔드: {config.backend}")

            all_generated.extend(gen_frames)

            # 청크 간 VRAM 정리
            flush_vram(max_idle_vram_mb)

    # 생성된 프레임 저장
    # 생성된 프레임 수가 원본과 다를 수 있음 → 원본 수에 맞춰 리사이즈/패딩
    for i, gen_frame in enumerate(all_generated):
        if i >= min_count:
            break
        # 원본과 동일한 크기로 리사이즈
        target_size = (metadata.processed_width, metadata.processed_height)
        if gen_frame.size != target_size:
            gen_frame = gen_frame.resize(target_size, Image.LANCZOS)
        gen_frame.save(gen_dir / f"{i:05d}.png")

    # 생성 프레임이 부족한 경우 마지막 프레임으로 패딩
    if len(all_generated) < min_count:
        logger.warning(
            f"  생성 프레임 부족: {len(all_generated)}/{min_count} — "
            f"마지막 프레임으로 패딩"
        )
        last_frame = all_generated[-1] if all_generated else frames[-1]
        target_size = (metadata.processed_width, metadata.processed_height)
        if last_frame.size != target_size:
            last_frame = last_frame.resize(target_size, Image.LANCZOS)
        for i in range(len(all_generated), min_count):
            last_frame.save(gen_dir / f"{i:05d}.png")

    log_vram_status("Node 05 종료 후")

    gen_count = len(list(gen_dir.glob("*.png")))
    logger.info(f"  생성 완료: {gen_count} frames -> {gen_dir}")
    logger.info("Node 05 완료")

    return gen_dir
