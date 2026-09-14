# Video Mask Inpainting Pipeline
"""
RTX 4080 (16GB VRAM) 환경을 위한 비디오 마스크 인페인팅 파이프라인.

순차 메모리 격리(Sequential Memory Isolation) 전략으로
SAM 2 VOS 트래킹과 DiT 인페인팅을 안전하게 실행합니다.
"""

__version__ = "0.1.0"
