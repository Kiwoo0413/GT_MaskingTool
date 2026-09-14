"""
tests/test_pipeline.py
파이프라인 기본 단위 테스트.

GPU/모델 의존성 없이 CPU에서 실행 가능한 테스트만 포함합니다.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ===========================================================================
# Config 테스트
# ===========================================================================

class TestConfig:
    """config.py 테스트."""

    def test_default_config(self):
        from pipeline.config import PipelineConfig
        cfg = PipelineConfig()
        assert cfg.workspace.base_dir == "./workspace"
        assert cfg.preprocessing.max_resolution == 1280
        assert cfg.inpainting.backend == "cogvideox"

    def test_load_config_default(self):
        from pipeline.config import load_config
        cfg = load_config(config_path=None, project_root=PROJECT_ROOT)
        assert cfg is not None
        assert cfg.preprocessing.target_fps == 24

    def test_load_config_from_yaml(self):
        from pipeline.config import load_config
        config_path = PROJECT_ROOT / "config.yaml"
        if config_path.exists():
            cfg = load_config(str(config_path), PROJECT_ROOT)
            assert cfg.sam2.model_cfg == "configs/sam2.1/sam2.1_hiera_b+.yaml"

    def test_active_inpainting_config(self):
        from pipeline.config import PipelineConfig
        cfg = PipelineConfig()
        cfg.inpainting.backend = "cogvideox"
        active = cfg.inpainting.active_config
        assert active.model_id == "THUDM/CogVideoX-5b-I2V"

        cfg.inpainting.backend = "wan21"
        active = cfg.inpainting.active_config
        assert "Wan" in active.model_id

    def test_resolve_paths(self):
        from pipeline.config import PipelineConfig
        cfg = PipelineConfig()
        cfg.resolve_paths(PROJECT_ROOT)
        assert Path(cfg.workspace.base_dir).is_absolute()


# ===========================================================================
# Node 01 테스트
# ===========================================================================

class TestNode01:
    """node01_prompt_parser.py 테스트."""

    def test_parse_json_input(self):
        from pipeline.node01_prompt_parser import parse_json_input
        json_str = json.dumps({
            "positive_prompt": "a gravel road",
            "negative_prompt": "blurry",
            "seed_coordinates": [[320, 240]],
            "coordinate_type": "point",
            "coordinate_labels": [1],
            "frame_range": [0, 100],
            "object_id": 1,
        })
        intent = parse_json_input(json_str)
        assert intent.positive_prompt == "a gravel road"
        assert intent.negative_prompt == "blurry"
        assert intent.seed_coordinates == [[320, 240]]
        assert intent.frame_range == (0, 100)

    def test_parse_cli_args_points(self):
        from pipeline.node01_prompt_parser import parse_cli_args
        intent = parse_cli_args(
            prompt="replace with grass",
            seed_points="[[100, 200], [300, 400]]",
            frame_range="[10, 50]",
        )
        assert intent.positive_prompt == "replace with grass"
        assert intent.coordinate_type == "point"
        assert len(intent.seed_coordinates) == 2
        assert intent.coordinate_labels == [1, 1]
        assert intent.frame_range == (10, 50)

    def test_parse_cli_args_box(self):
        from pipeline.node01_prompt_parser import parse_cli_args
        intent = parse_cli_args(
            prompt="clear sky",
            seed_box="[50, 50, 400, 300]",
        )
        assert intent.coordinate_type == "box"
        assert len(intent.seed_coordinates) == 4

    def test_validate_intent_success(self):
        from pipeline.node01_prompt_parser import ParsedIntent, validate_intent
        intent = ParsedIntent(
            positive_prompt="test prompt",
            seed_coordinates=[[100, 200]],
            coordinate_type="point",
            coordinate_labels=[1],
        )
        validate_intent(intent)  # 예외 없으면 성공

    def test_validate_intent_empty_prompt(self):
        from pipeline.node01_prompt_parser import ParsedIntent, validate_intent
        intent = ParsedIntent(
            positive_prompt="",
            seed_coordinates=[[100, 200]],
        )
        with pytest.raises(ValueError, match="프롬프트"):
            validate_intent(intent)

    def test_validate_intent_empty_coords(self):
        from pipeline.node01_prompt_parser import ParsedIntent, validate_intent
        intent = ParsedIntent(
            positive_prompt="test",
            seed_coordinates=[],
        )
        with pytest.raises(ValueError, match="시드 좌표"):
            validate_intent(intent)

    def test_run_node01_with_json(self):
        from pipeline.node01_prompt_parser import run_node01
        json_str = json.dumps({
            "positive_prompt": "a dirt path",
            "seed_coordinates": [[150, 250]],
            "coordinate_type": "point",
            "coordinate_labels": [1],
        })
        intent = run_node01(json_input=json_str)
        assert intent.positive_prompt == "a dirt path"


# ===========================================================================
# Node 02 테스트
# ===========================================================================

class TestNode02:
    """node02_frame_decomposer.py 테스트 (간접)."""

    def test_compute_scaled_size_no_change(self):
        from pipeline.node02_frame_decomposer import _compute_scaled_size
        w, h = _compute_scaled_size(1280, 720, 1280, 720)
        assert (w, h) == (1280, 720)

    def test_compute_scaled_size_downscale(self):
        from pipeline.node02_frame_decomposer import _compute_scaled_size
        w, h = _compute_scaled_size(1920, 1080, 1280, 720)
        assert w <= 1280
        assert h <= 1280
        # 2의 배수 확인
        assert w % 2 == 0
        assert h % 2 == 0

    def test_compute_scaled_size_portrait(self):
        from pipeline.node02_frame_decomposer import _compute_scaled_size
        w, h = _compute_scaled_size(1080, 1920, 1280, 720)
        assert w % 2 == 0
        assert h % 2 == 0


# ===========================================================================
# Node 04 테스트
# ===========================================================================

class TestNode04:
    """node04_mask_refinement.py 테스트."""

    def test_refine_single_mask(self):
        import cv2
        from pipeline.node04_mask_refinement import _refine_single_mask

        with tempfile.TemporaryDirectory() as tmpdir:
            # 테스트 마스크 생성
            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[30:70, 30:70] = 255  # 중앙 사각형
            input_path = os.path.join(tmpdir, "input.png")
            output_path = os.path.join(tmpdir, "output.png")
            cv2.imwrite(input_path, mask)

            # 후처리 실행
            result = _refine_single_mask((
                input_path, output_path,
                7, 2, 11, 0.0,
            ))

            assert os.path.exists(result)
            refined = cv2.imread(result, cv2.IMREAD_GRAYSCALE)
            assert refined is not None
            assert refined.shape == (100, 100)

            # 팽창으로 마스크 영역이 커졌는지 확인
            original_area = np.sum(mask > 0)
            refined_area = np.sum(refined > 0)
            assert refined_area >= original_area

    def test_temporal_consistency(self):
        import cv2
        from pipeline.node04_mask_refinement import (
            _validate_mask_temporal_consistency,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # 일관된 마스크 시퀀스
            for i in range(5):
                mask = np.zeros((100, 100), dtype=np.uint8)
                mask[30:70, 30 + i:70 + i] = 255  # 약간 이동
                cv2.imwrite(str(tmppath / f"{i:05d}.png"), mask)

            suspicious = _validate_mask_temporal_consistency(tmppath, threshold=0.3)
            assert len(suspicious) == 0  # 일관된 시퀀스


# ===========================================================================
# Node 06 테스트
# ===========================================================================

class TestNode06:
    """node06_compositor.py 테스트."""

    def test_alpha_composite_frame(self):
        from pipeline.node06_compositor import _alpha_composite_frame

        original = np.zeros((100, 100, 3), dtype=np.uint8)
        original[:] = [255, 0, 0]  # 빨간색

        generated = np.zeros((100, 100, 3), dtype=np.uint8)
        generated[:] = [0, 255, 0]  # 초록색

        # 전체 마스크 (완전 대체)
        mask = np.full((100, 100), 255, dtype=np.uint8)
        result = _alpha_composite_frame(original, generated, mask)
        np.testing.assert_array_almost_equal(
            result.astype(float),
            generated.astype(float),
            decimal=0,
        )

        # 빈 마스크 (변경 없음)
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = _alpha_composite_frame(original, generated, mask)
        np.testing.assert_array_equal(result, original)

        # 50% 마스크 (블렌딩)
        mask = np.full((100, 100), 128, dtype=np.uint8)
        result = _alpha_composite_frame(original, generated, mask)
        # 중간값 기대
        assert np.all(result[:, :, 0] > 0)  # R > 0 (원본 기여)
        assert np.all(result[:, :, 1] > 0)  # G > 0 (생성 기여)


# ===========================================================================
# VRAM Manager 테스트
# ===========================================================================

class TestVRAMManager:
    """vram_manager.py 테스트."""

    def test_get_torch_dtype(self):
        import torch
        from pipeline.vram_manager import get_torch_dtype

        assert get_torch_dtype("float16") == torch.float16
        assert get_torch_dtype("bfloat16") == torch.bfloat16
        assert get_torch_dtype("float32") == torch.float32
        assert get_torch_dtype("fp16") == torch.float16
        assert get_torch_dtype("bf16") == torch.bfloat16

    def test_get_torch_dtype_invalid(self):
        from pipeline.vram_manager import get_torch_dtype
        with pytest.raises(ValueError):
            get_torch_dtype("invalid_dtype")

    def test_flush_vram(self):
        from pipeline.vram_manager import flush_vram
        # CUDA가 없어도 에러 없이 실행되어야 함
        usage = flush_vram()
        assert isinstance(usage, float)

    def test_vram_scope(self):
        from pipeline.vram_manager import vram_scope
        # CUDA 없이도 에러 없이 실행되어야 함
        with vram_scope("test_scope"):
            pass  # 정상 종료


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
