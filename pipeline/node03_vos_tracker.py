"""
pipeline/node03_vos_tracker.py
Node 03: Temporal Mask Tracker (SAM 2 VOS).

SAM 2의 비디오 세그멘테이션 기능으로 시드 좌표에서
시공간 마스크를 생성하고 비디오 전체로 전파합니다.

실행 엔진: GPU (BF16)
필수: 작업 완료 후 VRAM 완전 해제
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .config import SAM2Config, WorkspaceConfig
from .node01_prompt_parser import ParsedIntent
from .node02_frame_decomposer import VideoMetadata
from .vram_manager import flush_vram, log_vram_status, vram_scope, get_torch_dtype

logger = logging.getLogger(__name__)


def _ensure_checkpoint(config: SAM2Config) -> str:
    """
    SAM 2 체크포인트가 존재하는지 확인하고, 없으면 다운로드합니다.

    Returns
    -------
    str
        체크포인트 파일 경로.
    """
    ckpt_path = Path(config.checkpoint)

    if ckpt_path.exists():
        logger.info(f"  SAM 2 체크포인트 발견: {ckpt_path}")
        return str(ckpt_path)

    if not config.auto_download:
        raise FileNotFoundError(
            f"SAM 2 체크포인트를 찾을 수 없습니다: {ckpt_path}\n"
            f"다운로드 URL: {config.checkpoint_url}"
        )

    # 자동 다운로드
    logger.info(f"  SAM 2 체크포인트 다운로드 중: {config.checkpoint_url}")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    import urllib.request
    urllib.request.urlretrieve(config.checkpoint_url, str(ckpt_path))
    logger.info(f"  다운로드 완료: {ckpt_path}")

    return str(ckpt_path)


def _save_mask(
    mask: np.ndarray,
    frame_idx: int,
    output_dir: Path,
) -> None:
    """바이너리 마스크를 PNG로 저장합니다."""
    # mask: (H, W) bool or float -> 0/255 uint8
    if mask.dtype == bool:
        mask_uint8 = (mask.astype(np.uint8)) * 255
    elif mask.max() <= 1.0:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask.astype(np.uint8)

    fname = f"{frame_idx:05d}.png"
    cv2.imwrite(str(output_dir / fname), mask_uint8)


def run_node03(
    intent: ParsedIntent,
    metadata: VideoMetadata,
    workspace: WorkspaceConfig,
    config: SAM2Config,
    max_idle_vram_mb: float = 1024.0,
) -> Path:
    """
    Node 03을 실행합니다.

    SAM 2 비디오 예측기로 시드 좌표에서 시작하여
    비디오 전체 프레임에 걸쳐 마스크를 전파합니다.

    Parameters
    ----------
    intent : ParsedIntent
        파싱된 사용자 의도 (시드 좌표 포함).
    metadata : VideoMetadata
        비디오 메타데이터.
    workspace : WorkspaceConfig
        워크스페이스 설정.
    config : SAM2Config
        SAM 2 모델 설정.
    max_idle_vram_mb : float
        VRAM flush 후 최대 허용 VRAM (MB).

    Returns
    -------
    Path
        원본(raw) 마스크 시퀀스 디렉토리 경로.
    """
    logger.info("=" * 60)
    logger.info("Node 03: Temporal Mask Tracker (SAM 2 VOS) 시작")
    logger.info("=" * 60)

    frames_dir = workspace.frames_path
    masks_dir = workspace.masks_raw_path

    # 기존 마스크 정리
    import shutil
    if masks_dir.exists():
        shutil.rmtree(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # 체크포인트 확인
    checkpoint = _ensure_checkpoint(config)

    log_vram_status("Node 03 시작 전")

    # VRAM 격리 스코프 내에서 SAM 2 실행
    with vram_scope("SAM2 Tracking", max_idle_vram_mb, flush_before=True, flush_after=True):
        _run_sam2_tracking(
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            intent=intent,
            metadata=metadata,
            config=config,
            checkpoint=checkpoint,
        )

    log_vram_status("Node 03 종료 후")

    # 생성된 마스크 수 확인
    mask_count = len(list(masks_dir.glob("*.png")))
    logger.info(f"  마스크 생성 완료: {mask_count} frames -> {masks_dir}")
    logger.info("Node 03 완료")

    return masks_dir


def _run_sam2_tracking(
    frames_dir: Path,
    masks_dir: Path,
    intent: ParsedIntent,
    metadata: VideoMetadata,
    config: SAM2Config,
    checkpoint: str,
) -> None:
    """SAM 2 모델 로드 및 트래킹 실행 (VRAM 스코프 내)."""

    # SAM 2 import
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError:
        raise ImportError(
            "sam2 패키지를 찾을 수 없습니다. 설치하세요:\n"
            "pip install git+https://github.com/facebookresearch/sam2.git"
        )

    device = config.device
    dtype = get_torch_dtype(config.dtype)

    # CUDA autocast 설정
    if device == "cuda" and torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # 모델 빌드
    logger.info(f"  SAM 2 모델 로드: cfg={config.model_cfg}, ckpt={checkpoint}")
    predictor = build_sam2_video_predictor(
        config.model_cfg, checkpoint, device=device
    )
    log_vram_status("SAM 2 모델 로드 후")

    try:
        # autocast 컨텍스트
        with torch.autocast(device, dtype=dtype):
            # 비디오 상태 초기화
            logger.info(f"  init_state: {frames_dir}")
            state = predictor.init_state(video_path=str(frames_dir))

            # 시드 좌표 주입
            if intent.coordinate_type == "point":
                points = np.array(intent.seed_coordinates, dtype=np.float32)
                labels = np.array(intent.coordinate_labels, dtype=np.int32)
                logger.info(
                    f"  포인트 주입: {points.tolist()}, labels={labels.tolist()}"
                )
                _, _, initial_masks = predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=intent.object_id,
                    points=points,
                    labels=labels,
                )
            elif intent.coordinate_type == "box":
                box = np.array(intent.seed_coordinates, dtype=np.float32)
                logger.info(f"  박스 주입: {box.tolist()}")
                _, _, initial_masks = predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=intent.object_id,
                    box=box,
                )

            # 초기 마스크 저장
            if initial_masks is not None and len(initial_masks) > 0:
                mask_np = (initial_masks[0] > 0.0).cpu().numpy().squeeze()
                _save_mask(mask_np, 0, masks_dir)

            # 비디오 전체로 마스크 전파
            logger.info("  마스크 전파 중...")
            for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
                for i, obj_id in enumerate(obj_ids):
                    if obj_id == intent.object_id:
                        mask_np = (masks[i] > 0.0).cpu().numpy().squeeze()
                        _save_mask(mask_np, frame_idx, masks_dir)

            logger.info("  마스크 전파 완료")

    finally:
        # 필수 Exit Protocol - 스코프 종료 전 명시적 삭제
        logger.info("  SAM 2 모델 해제 중...")
        del predictor
        if "state" in dir():
            del state
        gc.collect()
        torch.cuda.empty_cache()
        log_vram_status("SAM 2 모델 해제 후")
