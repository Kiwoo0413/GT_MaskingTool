"""
pipeline/node04_mask_refinement.py
Node 04: Mask Morphological Refinement.

바이너리 마스크에 팽창(dilation)과 가우시안 블러를 적용하여
자연스러운 경계 페더링을 생성합니다.

실행 엔진: CPU / OpenCV 멀티프로세싱 (VRAM 미사용)
"""

from __future__ import annotations

import logging
import shutil
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from .config import MaskRefinementConfig, WorkspaceConfig

logger = logging.getLogger(__name__)


def _refine_single_mask(args: Tuple[str, str, int, int, int, float]) -> str:
    """
    단일 마스크 파일을 형태학적 후처리합니다.

    Parameters
    ----------
    args : tuple
        (input_path, output_path, dilation_kernel, dilation_iters,
         gaussian_kernel, gaussian_sigma)

    Returns
    -------
    str
        출력 파일 경로.
    """
    (
        input_path,
        output_path,
        dilation_kernel_size,
        dilation_iterations,
        gaussian_kernel_size,
        gaussian_sigma,
    ) = args

    # 마스크 읽기 (그레이스케일)
    mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"마스크 파일을 읽을 수 없습니다: {input_path}")

    # 이진화 (안전장치)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 1. Dilation: 객체 경계 잔여물 완전 차단
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilation_kernel_size, dilation_kernel_size),
    )
    mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)

    # 2. Gaussian Blur: 자연스러운 경계 블렌딩 (페더링)
    mask = cv2.GaussianBlur(
        mask,
        (gaussian_kernel_size, gaussian_kernel_size),
        gaussian_sigma,
    )

    # float 정규화 (0.0 ~ 1.0) 후 다시 0-255 uint8으로 저장
    # 페더링 마스크는 연속값이므로 그대로 저장
    cv2.imwrite(output_path, mask)

    # Red 컬러 마스크 생성 (R 채널에 마스크, B/G는 0)
    if len(args) > 6 and args[6]:
        red_output_path = args[6]
        h, w = mask.shape[:2]
        red_matte = np.zeros((h, w, 3), dtype=np.uint8)
        red_matte[:, :, 2] = mask  # BGR: Red channel
        cv2.imwrite(red_output_path, red_matte)

    return output_path


def _validate_mask_temporal_consistency(
    masks_dir: Path,
    threshold: float = 0.3,
) -> List[int]:
    """
    연속 프레임 간 마스크의 시간적 일관성을 검증합니다.

    급격한 마스크 변화(드리프팅 후보)를 감지합니다.

    Parameters
    ----------
    masks_dir : Path
        마스크 디렉토리.
    threshold : float
        IoU 급락 임계값 (이 비율 이하로 IoU가 떨어지면 경고).

    Returns
    -------
    list[int]
        드리프팅이 의심되는 프레임 인덱스 목록.
    """
    mask_files = sorted(masks_dir.glob("*.png"))
    if len(mask_files) < 2:
        return []

    suspicious_frames = []
    prev_mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
    if prev_mask is None:
        return []
    _, prev_mask = cv2.threshold(prev_mask, 127, 255, cv2.THRESH_BINARY)
    prev_area = np.sum(prev_mask > 0)

    for i in range(1, len(mask_files)):
        curr_mask = cv2.imread(str(mask_files[i]), cv2.IMREAD_GRAYSCALE)
        if curr_mask is None:
            continue
        _, curr_mask = cv2.threshold(curr_mask, 127, 255, cv2.THRESH_BINARY)
        curr_area = np.sum(curr_mask > 0)

        if prev_area > 0:
            # 면적 기반 간이 일관성 체크
            intersection = np.sum((prev_mask > 0) & (curr_mask > 0))
            union = np.sum((prev_mask > 0) | (curr_mask > 0))
            iou = intersection / union if union > 0 else 0.0

            if iou < threshold:
                suspicious_frames.append(i)
                logger.warning(
                    f"  프레임 {i}: IoU={iou:.3f} (임계값 {threshold} 미만) "
                    f"— 마스크 드리프팅 의심"
                )

        prev_mask = curr_mask
        prev_area = curr_area

    return suspicious_frames


def run_node04(
    workspace: WorkspaceConfig,
    config: MaskRefinementConfig,
    red_masks_dir: Optional[Path] = None,
) -> Tuple[Path, Optional[Path]]:
    """
    Node 04를 실행합니다.

    바이너리 마스크를 형태학적 후처리(팽창 + 가우시안 블러)하여
    자연스러운 페더링 마스크 및 Red 컬러 마스크를 생성합니다.

    Parameters
    ----------
    workspace : WorkspaceConfig
        워크스페이스 설정.
    config : MaskRefinementConfig
        마스크 후처리 설정.
    red_masks_dir : Optional[Path]
        Red 컬러 마스크 저장 디렉토리 (선택).

    Returns
    -------
    Tuple[Path, Optional[Path]]
        (페더링 마스크 디렉토리 경로, Red 컬러 마스크 디렉토리 경로)
    """
    logger.info("=" * 60)
    logger.info("Node 04: Mask Morphological Refinement 시작")
    logger.info("=" * 60)

    masks_raw_dir = workspace.masks_raw_path
    masks_feathered_dir = workspace.masks_feathered_path

    # 기존 결과 정리
    if masks_feathered_dir.exists():
        shutil.rmtree(masks_feathered_dir)
    masks_feathered_dir.mkdir(parents=True, exist_ok=True)

    if red_masks_dir is not None:
        if red_masks_dir.exists():
            shutil.rmtree(red_masks_dir)
        red_masks_dir.mkdir(parents=True, exist_ok=True)

    # 마스크 파일 목록
    mask_files = sorted(masks_raw_dir.glob("*.png"))
    if not mask_files:
        raise RuntimeError(f"마스크 파일이 없습니다: {masks_raw_dir}")

    logger.info(f"  입력 마스크: {len(mask_files)} files")
    logger.info(
        f"  Dilation: kernel={config.dilation_kernel_size}x"
        f"{config.dilation_kernel_size}, "
        f"iterations={config.dilation_iterations}"
    )
    logger.info(
        f"  Gaussian Blur: kernel={config.gaussian_kernel_size}x"
        f"{config.gaussian_kernel_size}, "
        f"sigma={config.gaussian_sigma}"
    )

    # 멀티프로세싱으로 마스크 후처리
    tasks = []
    for mask_file in mask_files:
        output_file = masks_feathered_dir / mask_file.name
        red_output_file = (red_masks_dir / mask_file.name) if red_masks_dir else None
        tasks.append((
            str(mask_file),
            str(output_file),
            config.dilation_kernel_size,
            config.dilation_iterations,
            config.gaussian_kernel_size,
            config.gaussian_sigma,
            str(red_output_file) if red_output_file else None,
        ))

    num_workers = min(config.num_workers, cpu_count(), len(tasks))
    logger.info(f"  CPU 워커 수: {num_workers}")

    if num_workers > 1:
        with Pool(num_workers) as pool:
            pool.map(_refine_single_mask, tasks)
    else:
        # 단일 프로세스 (디버깅 용이)
        for task in tasks:
            _refine_single_mask(task)

    # 시간적 일관성 검증
    logger.info("  시간적 일관성 검증 중...")
    suspicious = _validate_mask_temporal_consistency(masks_feathered_dir)
    if suspicious:
        logger.warning(
            f"  드리프팅 의심 프레임 {len(suspicious)}개: {suspicious[:10]}..."
        )
    else:
        logger.info("  시간적 일관성 양호")

    logger.info(f"  페더링 마스크 생성 완료 -> {masks_feathered_dir}")
    if red_masks_dir:
        logger.info(f"  Red 컬러 마스크 생성 완료 -> {red_masks_dir}")
    logger.info("Node 04 완료")

    return masks_feathered_dir, red_masks_dir
