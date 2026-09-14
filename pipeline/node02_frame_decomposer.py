"""
pipeline/node02_frame_decomposer.py
Node 02: Frame Decomposer & Preprocessor.

원본 비디오를 프레임 시퀀스로 분해하고 해상도/FPS를 정규화합니다.

실행 엔진: CPU / OpenCV (VRAM 미사용)
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import PreprocessingConfig, WorkspaceConfig

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """비디오 메타데이터.

    Attributes
    ----------
    original_width : int
        원본 너비.
    original_height : int
        원본 높이.
    processed_width : int
        처리 후 너비.
    processed_height : int
        처리 후 높이.
    original_fps : float
        원본 FPS.
    processed_fps : float
        처리 후 FPS.
    total_frames : int
        총 프레임 수.
    has_audio : bool
        오디오 트랙 존재 여부.
    input_path : str
        원본 비디오 경로.
    """
    original_width: int = 0
    original_height: int = 0
    processed_width: int = 0
    processed_height: int = 0
    original_fps: float = 0.0
    processed_fps: float = 0.0
    total_frames: int = 0
    has_audio: bool = False
    input_path: str = ""


def _compute_scaled_size(
    width: int,
    height: int,
    max_resolution: int,
    target_short_side: int,
) -> Tuple[int, int]:
    """
    해상도 스케일링 크기를 계산합니다.

    너비 또는 높이가 max_resolution을 초과하면
    비율을 유지하면서 target_short_side 기준으로 스케일다운합니다.
    결과 크기는 항상 2의 배수로 맞춥니다 (비디오 인코딩 호환).

    Returns
    -------
    tuple[int, int]
        (new_width, new_height)
    """
    if max(width, height) <= max_resolution:
        # 해상도가 제한 이내이면 변경 없음
        return width, height

    # 짧은 쪽을 target_short_side로 맞추고 비율 유지
    if width < height:
        scale = target_short_side / width
    else:
        scale = target_short_side / height

    new_w = int(width * scale)
    new_h = int(height * scale)

    # 2의 배수로 정렬
    new_w = new_w - (new_w % 2)
    new_h = new_h - (new_h % 2)

    return new_w, new_h


def _check_audio_stream(video_path: str) -> bool:
    """FFprobe 없이 간단히 오디오 존재 여부를 확인합니다."""
    try:
        import subprocess
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        return "audio" in result.stdout.lower()
    except Exception:
        # ffprobe가 없거나 실패하면 True로 가정 (안전)
        logger.warning("ffprobe 실행 실패 — 오디오 존재로 가정합니다.")
        return True


def run_node02(
    input_video: str,
    workspace: WorkspaceConfig,
    config: PreprocessingConfig,
    frame_range: Optional[Tuple[int, int]] = None,
) -> VideoMetadata:
    """
    Node 02를 실행합니다.

    비디오를 프레임 시퀀스로 분해하고, 필요 시 해상도/FPS를 조정합니다.

    Parameters
    ----------
    input_video : str
        입력 비디오 파일 경로.
    workspace : WorkspaceConfig
        워크스페이스 설정.
    config : PreprocessingConfig
        전처리 설정.
    frame_range : tuple[int, int], optional
        처리할 프레임 범위. None이면 전체.

    Returns
    -------
    VideoMetadata
        비디오 메타데이터.
    """
    logger.info("=" * 60)
    logger.info("Node 02: Frame Decomposer & Preprocessor 시작")
    logger.info("=" * 60)

    input_path = Path(input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"입력 비디오를 찾을 수 없습니다: {input_video}")

    # 출력 디렉토리 준비
    frames_dir = workspace.frames_path
    # 기존 프레임 정리
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 비디오 열기
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {input_video}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"  원본: {orig_w}x{orig_h} @ {orig_fps:.2f}fps, {orig_total} frames")

    # 해상도 결정
    proc_w, proc_h = _compute_scaled_size(
        orig_w, orig_h,
        config.max_resolution,
        config.target_short_side,
    )
    need_resize = (proc_w != orig_w) or (proc_h != orig_h)

    if need_resize:
        logger.info(f"  스케일다운: {orig_w}x{orig_h} -> {proc_w}x{proc_h}")

    # FPS 결정
    proc_fps = orig_fps
    if config.normalize_fps and orig_fps != config.target_fps:
        proc_fps = float(config.target_fps)
        logger.info(f"  FPS 정규화: {orig_fps:.2f} -> {proc_fps:.2f}")

    # FPS 변환 시 프레임 샘플링 간격 계산
    if config.normalize_fps and orig_fps > 0 and proc_fps != orig_fps:
        frame_interval = orig_fps / proc_fps
    else:
        frame_interval = 1.0

    # 프레임 추출 범위 결정
    start_frame = 0
    end_frame = orig_total
    if frame_range is not None:
        start_frame = frame_range[0]
        end_frame = min(frame_range[1] + 1, orig_total)  # end inclusive -> exclusive
    
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 프레임 추출
    frame_ext = config.frame_format
    saved_count = 0
    source_idx = start_frame
    next_sample = 0.0

    while source_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # FPS 정규화 시 샘플링
        if source_idx - start_frame >= next_sample:
            # 리사이즈
            if need_resize:
                frame = cv2.resize(
                    frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA
                )

            # 저장
            fname = f"{saved_count:05d}.{frame_ext}"
            fpath = frames_dir / fname

            if frame_ext == "jpg":
                cv2.imwrite(
                    str(fpath), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality],
                )
            else:
                cv2.imwrite(str(fpath), frame)

            saved_count += 1
            next_sample += frame_interval

        source_idx += 1

    cap.release()

    # 오디오 확인
    has_audio = _check_audio_stream(str(input_path))

    metadata = VideoMetadata(
        original_width=orig_w,
        original_height=orig_h,
        processed_width=proc_w,
        processed_height=proc_h,
        original_fps=orig_fps,
        processed_fps=proc_fps,
        total_frames=saved_count,
        has_audio=has_audio,
        input_path=str(input_path),
    )

    logger.info(f"  처리 완료: {saved_count} frames -> {frames_dir}")
    logger.info(f"  해상도: {proc_w}x{proc_h}, FPS: {proc_fps}, 오디오: {has_audio}")
    logger.info("Node 02 완료")

    return metadata
