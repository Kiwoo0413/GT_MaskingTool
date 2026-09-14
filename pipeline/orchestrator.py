"""
pipeline/orchestrator.py
파이프라인 오케스트레이터.

6개 노드의 순차 실행을 관리하고, VRAM 격리 규칙을 강제하며,
Self-Healing 로직(OOM 재시도, 해상도 축소)을 구현합니다.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from .config import PipelineConfig, load_config
from .node01_prompt_parser import ParsedIntent, run_node01
from .node02_frame_decomposer import VideoMetadata, run_node02
from .node03_vos_tracker import run_node03
from .node04_mask_refinement import run_node04
from .node05_dit_inpainter import run_node05
from .node06_compositor import run_node06
from .vram_manager import flush_vram, log_vram_status

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    비디오 마스크 인페인팅 파이프라인 오케스트레이터.

    6개 노드를 순차적으로 실행하며, VRAM 메모리 격리를
    강제합니다.

    Parameters
    ----------
    config : PipelineConfig
        파이프라인 설정.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._metadata: Optional[VideoMetadata] = None
        self._intent: Optional[ParsedIntent] = None

    def run(
        self,
        input_video: str,
        output_video: str,
        prompt: Optional[str] = None,
        seed_points: Optional[str] = None,
        seed_box: Optional[str] = None,
        frame_range: Optional[str] = None,
        negative_prompt: str = "",
        object_id: int = 1,
        json_input: Optional[str] = None,
    ) -> str:
        """
        전체 파이프라인을 실행합니다.

        Parameters
        ----------
        input_video : str
            입력 비디오 파일 경로.
        output_video : str
            출력 비디오 파일 경로.
        prompt : str, optional
            인페인팅 프롬프트.
        seed_points : str, optional
            시드 포인트 좌표 JSON.
        seed_box : str, optional
            시드 박스 좌표 JSON.
        frame_range : str, optional
            프레임 범위 JSON.
        negative_prompt : str
            네거티브 프롬프트.
        object_id : int
            트래킹 객체 ID.
        json_input : str, optional
            JSON 전체 입력.

        Returns
        -------
        str
            최종 출력 비디오 경로.
        """
        total_start = time.time()

        logger.info("=" * 70)
        logger.info(" Video Mask Inpainting Pipeline 시작")
        logger.info("=" * 70)
        logger.info(f"  입력: {input_video}")
        logger.info(f"  출력: {output_video}")

        try:
            # ─── Node 01: Prompt & Intent Parsing (CPU) ───────────
            t0 = time.time()
            self._intent = run_node01(
                prompt=prompt,
                seed_points=seed_points,
                seed_box=seed_box,
                frame_range=frame_range,
                negative_prompt=negative_prompt,
                object_id=object_id,
                json_input=json_input,
            )
            logger.info(f"  [Node 01 소요시간: {time.time() - t0:.1f}s]")

            # ─── Node 02: Frame Decompose & Scaling (CPU) ─────────
            t0 = time.time()
            self._metadata = run_node02(
                input_video=input_video,
                workspace=self.config.workspace,
                config=self.config.preprocessing,
                frame_range=self._intent.frame_range,
            )
            logger.info(f"  [Node 02 소요시간: {time.time() - t0:.1f}s]")

            # ─── Node 03: VOS Tracking (SAM 2, GPU) ──────────────
            # OOM Self-Healing 루프
            t0 = time.time()
            self._run_with_oom_retry(
                self._run_node03,
                node_name="Node 03 (SAM 2 VOS)",
            )
            logger.info(f"  [Node 03 소요시간: {time.time() - t0:.1f}s]")

            # ▲ VRAM 격리 경계: SAM 2 모델 완전 해제됨 ▲

            # ─── Node 04: Mask Morphological Post-Processing (CPU) ─
            t0 = time.time()
            run_node04(
                workspace=self.config.workspace,
                config=self.config.mask_refinement,
            )
            logger.info(f"  [Node 04 소요시간: {time.time() - t0:.1f}s]")

            # ─── Node 05: Generative DiT Inpainting (GPU) ────────
            t0 = time.time()
            self._run_with_oom_retry(
                self._run_node05,
                node_name="Node 05 (DiT Inpainting)",
            )
            logger.info(f"  [Node 05 소요시간: {time.time() - t0:.1f}s]")

            # ▲ VRAM 격리 경계: DiT 모델 완전 해제됨 ▲

            # ─── Node 06: Alpha Composite & Audio Merge (FFmpeg) ──
            t0 = time.time()
            result = run_node06(
                output_video=output_video,
                metadata=self._metadata,
                workspace=self.config.workspace,
                config=self.config.assembly,
            )
            logger.info(f"  [Node 06 소요시간: {time.time() - t0:.1f}s]")

        except Exception as e:
            logger.error(f"파이프라인 실행 실패: {e}", exc_info=True)
            # 최종 VRAM 정리
            flush_vram()
            raise

        total_elapsed = time.time() - total_start
        logger.info("=" * 70)
        logger.info(f" 파이프라인 완료! 총 소요시간: {total_elapsed:.1f}s")
        logger.info(f" 출력 파일: {result}")
        logger.info("=" * 70)

        return result

    def _run_node03(self) -> None:
        """Node 03 실행 (OOM retry wrapper에서 호출)."""
        run_node03(
            intent=self._intent,
            metadata=self._metadata,
            workspace=self.config.workspace,
            config=self.config.sam2,
            max_idle_vram_mb=float(self.config.vram.max_idle_vram_mb),
        )

    def _run_node05(self) -> None:
        """Node 05 실행 (OOM retry wrapper에서 호출)."""
        run_node05(
            intent=self._intent,
            metadata=self._metadata,
            workspace=self.config.workspace,
            config=self.config.inpainting,
            max_idle_vram_mb=float(self.config.vram.max_idle_vram_mb),
        )

    def _run_with_oom_retry(
        self,
        run_fn,
        node_name: str,
    ) -> None:
        """
        OOM 발생 시 해상도를 축소하고 재실행하는 Self-Healing 래퍼.

        Parameters
        ----------
        run_fn : callable
            실행할 노드 함수.
        node_name : str
            노드 이름 (로깅용).
        """
        max_retries = self.config.error_handling.max_oom_retries
        scale_factor = self.config.error_handling.resolution_scale_factor
        min_res = self.config.error_handling.min_resolution

        for attempt in range(max_retries + 1):
            try:
                run_fn()
                return  # 성공
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                error_msg = str(e).lower()
                is_oom = (
                    "out of memory" in error_msg
                    or "cuda" in error_msg and "memory" in error_msg
                    or isinstance(e, torch.cuda.OutOfMemoryError)
                )

                if not is_oom or attempt >= max_retries:
                    raise

                # OOM 발생 — 해상도 축소 후 재시도
                old_w = self._metadata.processed_width
                old_h = self._metadata.processed_height

                new_w = int(old_w * scale_factor)
                new_h = int(old_h * scale_factor)

                # 2의 배수로 정렬
                new_w = new_w - (new_w % 2)
                new_h = new_h - (new_h % 2)

                if min(new_w, new_h) < min_res:
                    logger.error(
                        f"  해상도가 최소 제한({min_res}) 미만으로 축소됨 — "
                        f"재시도 중단"
                    )
                    raise

                logger.warning(
                    f"  [{node_name}] CUDA OOM 발생! "
                    f"해상도 축소: {old_w}x{old_h} -> {new_w}x{new_h} "
                    f"(시도 {attempt + 1}/{max_retries})"
                )

                # VRAM 완전 정리
                flush_vram()

                # 해상도 업데이트
                self._metadata.processed_width = new_w
                self._metadata.processed_height = new_h

                # 프레임 리사이즈 (Node 02 재실행)
                logger.info("  프레임 재스케일링 중...")
                self._rescale_frames(new_w, new_h)

        raise RuntimeError(f"{node_name}: 최대 재시도 횟수 초과")

    def _rescale_frames(self, new_w: int, new_h: int) -> None:
        """기존 프레임을 새 해상도로 리사이즈합니다."""
        import cv2

        frames_dir = self.config.workspace.frames_path
        frame_files = sorted(
            f for f in frames_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".png")
        )

        for fpath in frame_files:
            img = cv2.imread(str(fpath))
            if img is not None:
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(fpath), img)

        logger.info(f"  {len(frame_files)} 프레임 리사이즈 완료: {new_w}x{new_h}")


# torch import는 _run_with_oom_retry에서 사용하므로 top-level 유지
import torch
