"""
pipeline/node06_compositor.py
Node 06: Alpha Composite & Final Assembly.

원본 프레임과 생성된 프레임을 마스크 기반 알파 블렌딩하고,
FFmpeg NVENC으로 최종 비디오를 인코딩합니다.

실행 엔진: CPU (OpenCV/NumPy) + FFmpeg (NVENC GPU 인코딩)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import AssemblyConfig, WorkspaceConfig
from .node02_frame_decomposer import VideoMetadata

logger = logging.getLogger(__name__)


def _alpha_composite_frame(
    original: np.ndarray,
    generated: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    단일 프레임을 알파 합성합니다.

    Output = Original × (1 - Mask) + Generated × Mask

    Parameters
    ----------
    original : np.ndarray
        원본 프레임 (H, W, 3) uint8.
    generated : np.ndarray
        생성된 프레임 (H, W, 3) uint8.
    mask : np.ndarray
        페더링 마스크 (H, W) uint8 (0~255).

    Returns
    -------
    np.ndarray
        합성된 프레임 (H, W, 3) uint8.
    """
    # 마스크를 float 0.0~1.0으로 변환
    alpha = mask.astype(np.float32) / 255.0

    # 3채널로 확장
    if alpha.ndim == 2:
        alpha = alpha[:, :, np.newaxis]

    # 크기 일치 확인 & 리사이즈
    h, w = original.shape[:2]
    if generated.shape[:2] != (h, w):
        generated = cv2.resize(generated, (w, h), interpolation=cv2.INTER_LANCZOS4)
    if alpha.shape[:2] != (h, w):
        alpha_2d = alpha.squeeze()
        alpha_2d = cv2.resize(alpha_2d, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha = alpha_2d[:, :, np.newaxis]

    # 알파 블렌딩
    orig_f = original.astype(np.float32)
    gen_f = generated.astype(np.float32)

    composite = orig_f * (1.0 - alpha) + gen_f * alpha
    composite = np.clip(composite, 0, 255).astype(np.uint8)

    return composite


def _compose_all_frames(
    frames_dir: Path,
    generated_dir: Path,
    masks_dir: Path,
    composite_dir: Path,
) -> int:
    """
    모든 프레임을 알파 합성합니다.

    Returns
    -------
    int
        합성된 프레임 수.
    """
    # 프레임 파일 목록
    frame_exts = (".jpg", ".png")
    frame_files = sorted(
        f for f in frames_dir.iterdir()
        if f.suffix.lower() in frame_exts
    )
    gen_files = sorted(generated_dir.glob("*.png"))
    mask_files = sorted(masks_dir.glob("*.png"))

    # 최소 수 맞춤
    count = min(len(frame_files), len(gen_files), len(mask_files))
    if count == 0:
        raise RuntimeError("합성할 프레임이 없습니다.")

    logger.info(f"  합성 프레임 수: {count}")

    for i in range(count):
        original = cv2.imread(str(frame_files[i]))
        generated = cv2.imread(str(gen_files[i]))
        mask = cv2.imread(str(mask_files[i]), cv2.IMREAD_GRAYSCALE)

        if original is None:
            raise RuntimeError(f"원본 프레임 읽기 실패: {frame_files[i]}")
        if generated is None:
            raise RuntimeError(f"생성 프레임 읽기 실패: {gen_files[i]}")
        if mask is None:
            raise RuntimeError(f"마스크 읽기 실패: {mask_files[i]}")

        composite = _alpha_composite_frame(original, generated, mask)

        output_path = composite_dir / f"{i:05d}.png"
        cv2.imwrite(str(output_path), composite)

    return count


def _encode_video_ffmpeg(
    composite_dir: Path,
    input_video: str,
    output_path: str,
    fps: float,
    has_audio: bool,
    config: AssemblyConfig,
) -> None:
    """
    FFmpeg으로 합성 프레임을 비디오로 인코딩합니다.

    NVENC H.264 인코딩 + 원본 오디오 스트림 결합.
    """
    cmd = [
        "ffmpeg", "-y",
        "-r", str(fps),
        "-i", str(composite_dir / "%05d.png"),
    ]

    # 원본 비디오에서 오디오 가져오기
    if has_audio:
        cmd.extend(["-i", input_video])

    cmd.extend(["-map", "0:v"])

    if has_audio:
        cmd.extend(["-map", "1:a?"])

    # 인코더 설정
    cmd.extend(["-c:v", config.encoder])
    cmd.extend(["-pix_fmt", config.pixel_format])

    # 인코더별 추가 옵션
    if "nvenc" in config.encoder:
        cmd.extend(["-preset", config.preset])
        cmd.extend(["-rc", "vbr"])
        cmd.extend(["-cq", str(config.crf)])
    else:
        # libx264 fallback
        cmd.extend(["-crf", str(config.crf)])
        cmd.extend(["-preset", "medium"])

    # 오디오 코덱 (복사)
    if has_audio:
        cmd.extend(["-c:a", "copy"])

    cmd.append(output_path)

    logger.info(f"  FFmpeg 명령: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10분 타임아웃
        )
        if result.returncode != 0:
            logger.error(f"FFmpeg stderr:\n{result.stderr}")
            # NVENC 실패 시 libx264 fallback
            if "nvenc" in config.encoder:
                logger.warning("  NVENC 인코딩 실패 — libx264으로 fallback")
                fallback_config = AssemblyConfig(
                    encoder="libx264",
                    pixel_format=config.pixel_format,
                    crf=config.crf,
                    preset="medium",
                )
                _encode_video_ffmpeg(
                    composite_dir, input_video, output_path,
                    fps, has_audio, fallback_config,
                )
                return
            raise RuntimeError(
                f"FFmpeg 인코딩 실패 (return code: {result.returncode})"
            )
        logger.info(f"  인코딩 완료: {output_path}")

    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg을 찾을 수 없습니다. FFmpeg을 설치하고 PATH에 추가하세요."
        )


def run_node06(
    output_video: str,
    metadata: VideoMetadata,
    workspace: WorkspaceConfig,
    config: AssemblyConfig,
) -> str:
    """
    Node 06을 실행합니다.

    원본/생성 프레임을 마스크 기반으로 합성하고,
    FFmpeg으로 최종 비디오를 인코딩합니다.

    Parameters
    ----------
    output_video : str
        출력 비디오 파일 경로.
    metadata : VideoMetadata
        비디오 메타데이터 (FPS, 오디오 정보 등).
    workspace : WorkspaceConfig
        워크스페이스 설정.
    config : AssemblyConfig
        인코딩 설정.

    Returns
    -------
    str
        최종 출력 비디오 경로.
    """
    logger.info("=" * 60)
    logger.info("Node 06: Alpha Composite & Final Assembly 시작")
    logger.info("=" * 60)

    composite_dir = workspace.composite_path

    # 기존 결과 정리
    if composite_dir.exists():
        shutil.rmtree(composite_dir)
    composite_dir.mkdir(parents=True, exist_ok=True)

    # 1. 알파 합성
    logger.info("  Step 1: 알파 합성 수행 중...")
    frame_count = _compose_all_frames(
        frames_dir=workspace.frames_path,
        generated_dir=workspace.generated_frames_path,
        masks_dir=workspace.masks_feathered_path,
        composite_dir=composite_dir,
    )

    # 2. FFmpeg 인코딩
    logger.info("  Step 2: FFmpeg 인코딩 중...")
    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _encode_video_ffmpeg(
        composite_dir=composite_dir,
        input_video=metadata.input_path,
        output_path=str(output_path),
        fps=metadata.processed_fps,
        has_audio=metadata.has_audio,
        config=config,
    )

    # 결과 확인
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  최종 출력: {output_path} ({size_mb:.1f}MB)")
    else:
        raise RuntimeError(f"출력 파일이 생성되지 않았습니다: {output_path}")

    logger.info(f"  합성 프레임: {frame_count}, FPS: {metadata.processed_fps}")
    logger.info("Node 06 완료")

    return str(output_path)
