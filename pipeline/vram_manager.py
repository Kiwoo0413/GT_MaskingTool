"""
pipeline/vram_manager.py
VRAM 메모리 관리 유틸리티.

순차 메모리 격리(Sequential Memory Isolation) 전략의 핵심 모듈.
SAM 2와 Video DiT 모델이 VRAM에 동시 상주하지 않도록 보장합니다.
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from typing import Callable, Optional, TypeVar

import torch

logger = logging.getLogger(__name__)

T = TypeVar("T")


def get_vram_usage_mb() -> float:
    """
    현재 GPU VRAM 사용량을 MB 단위로 반환합니다.

    Returns
    -------
    float
        현재 VRAM 사용량 (MB). CUDA를 사용할 수 없으면 0.0.
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 * 1024)


def get_vram_reserved_mb() -> float:
    """
    현재 GPU VRAM 예약량을 MB 단위로 반환합니다.

    Returns
    -------
    float
        현재 VRAM 예약량 (MB).
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_reserved() / (1024 * 1024)


def flush_vram(max_idle_mb: float = 1024.0) -> float:
    """
    VRAM을 정리하고 점유율을 최소화합니다.

    가비지 컬렉터 실행 후 CUDA 캐시를 비웁니다.
    결과적으로 VRAM 점유율이 ``max_idle_mb`` 미만이어야 합니다.

    Parameters
    ----------
    max_idle_mb : float
        목표 최대 잔여 VRAM (MB). 기본 1024 MB (1 GB).

    Returns
    -------
    float
        flush 후 VRAM 사용량 (MB).

    Raises
    ------
    RuntimeWarning
        VRAM 사용량이 ``max_idle_mb``를 초과하면 경고.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    usage = get_vram_usage_mb()
    reserved = get_vram_reserved_mb()
    logger.info(
        f"VRAM flush 완료: allocated={usage:.1f}MB, reserved={reserved:.1f}MB"
    )

    if usage > max_idle_mb:
        import warnings
        warnings.warn(
            f"VRAM flush 후에도 {usage:.1f}MB 사용 중 "
            f"(목표: {max_idle_mb:.1f}MB 미만). "
            f"메모리 누수 가능성이 있습니다.",
            RuntimeWarning,
            stacklevel=2,
        )

    return usage


def force_delete(*objects) -> None:
    """
    객체들을 명시적으로 삭제하고 VRAM을 해제합니다.

    모델이나 상태 객체를 전달하면 참조를 제거한 뒤
    가비지 컬렉션과 CUDA 캐시 정리를 수행합니다.

    Parameters
    ----------
    *objects
        삭제할 객체들.
    """
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def vram_scope(
    scope_name: str = "unnamed",
    max_idle_mb: float = 1024.0,
    flush_before: bool = True,
    flush_after: bool = True,
):
    """
    VRAM 격리 스코프를 제공하는 컨텍스트 매니저.

    스코프 진입 시 VRAM을 flush하고, 종료 시 다시 flush합니다.
    GPU 집약적인 노드(Node 03, Node 05) 실행 시 사용합니다.

    Parameters
    ----------
    scope_name : str
        스코프 이름 (로깅용).
    max_idle_mb : float
        flush 후 최대 허용 VRAM (MB).
    flush_before : bool
        스코프 진입 전 flush 여부.
    flush_after : bool
        스코프 종료 후 flush 여부.

    Example
    -------
    >>> with vram_scope("SAM2 Tracking"):
    ...     predictor = load_sam2_model()
    ...     masks = run_tracking(predictor)
    ...     del predictor  # 스코프 내에서 명시적 삭제 권장
    """
    if flush_before:
        logger.info(f"[{scope_name}] VRAM 스코프 진입 - 사전 flush 수행")
        flush_vram(max_idle_mb)

    try:
        yield
    finally:
        if flush_after:
            logger.info(f"[{scope_name}] VRAM 스코프 종료 - 사후 flush 수행")
            flush_vram(max_idle_mb)


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """
    문자열을 torch.dtype으로 변환합니다.

    Parameters
    ----------
    dtype_str : str
        "float16", "bfloat16", "float32" 등.

    Returns
    -------
    torch.dtype
    """
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype_str_lower = dtype_str.lower()
    if dtype_str_lower not in dtype_map:
        raise ValueError(
            f"지원하지 않는 dtype: {dtype_str!r}. "
            f"사용 가능: {list(dtype_map.keys())}"
        )
    return dtype_map[dtype_str_lower]


def log_vram_status(label: str = "") -> None:
    """현재 VRAM 상태를 로깅합니다."""
    if not torch.cuda.is_available():
        logger.info(f"[VRAM {label}] CUDA 사용 불가")
        return

    allocated = get_vram_usage_mb()
    reserved = get_vram_reserved_mb()
    props = torch.cuda.get_device_properties(0)
    total_bytes = getattr(props, "total_memory", getattr(props, "total_mem", 0))
    total = total_bytes / (1024 * 1024)
    free = total - allocated

    logger.info(
        f"[VRAM {label}] "
        f"allocated={allocated:.0f}MB | "
        f"reserved={reserved:.0f}MB | "
        f"free≈{free:.0f}MB | "
        f"total={total:.0f}MB"
    )
