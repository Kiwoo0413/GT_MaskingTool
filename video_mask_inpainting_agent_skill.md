# Skill: Video Mask Inpainting Pipeline (RTX 4080 / 16GB VRAM)

## 1. 개요 및 목적
본 Skill은 사용자의 자연어 명령 또는 좌표 입력을 바탕으로 영상 내 특정 객체를 시공간 마스킹(Spatio-temporal Masking)하고, 확산 모델(Diffusion Transformer)을 통해 인페인팅(Inpainting) 편집을 완수하는 비디오 편집 에이전트 전용 실행 가이드라인이다.
하드웨어 제약(VRAM 16GB)을 고려하여 **순차 메모리 격리(Sequential Memory Isolation)** 전략을 엄격히 준수한다.

---

## 2. 하드웨어 환경 및 제약 조건
* **GPU**: NVIDIA GeForce RTX 4080 (VRAM 16GB GDDR6X)
* **메모리 보호 규칙 (절대 원칙)**:
  1. `SAM 2`와 `Video DiT` 모델을 VRAM에 동시에 상주시키지 않는다.
  2. 한 노드의 작업이 끝나면 가비지 컬렉터(`gc.collect()`) 및 `torch.cuda.empty_cache()`를 호출하여 VRAM 점유율을 1GB 미만으로 초기화한 뒤 다음 모델을 로드한다.
  3. 작업 기본 해상도는 `1280x720` (720p) 또는 `832x480` 이하로 제한한다.
  4. 비디오 생성 모델은 `FP8 / NF4` 양자화와 `CPU Offload`, `Tiled VAE`를 필수로 활성화한다.

---

## 3. 노드 기반 파이프라인 구조 (Node Architecture)

```text
[Node 01: Prompt & Intent Parsing] (CPU/LLM)
       │
       ▼
[Node 02: Video Decompose & Scaling] (CPU/OpenCV)
       │
       ▼
[Node 03: VOS Tracking (SAM 2)] ──▶ [VRAM 0MB Flush: unload model]
       │
       ▼
[Node 04: Mask Morphological Post-Processing] (CPU/OpenCV)
       │
       ▼
[Node 05: Generative DiT Inpainting] (FP8 / CPU Offload / Tiled VAE)
       │
       ▼
[Node 06: Alpha Composite & Audio Merge] (FFmpeg / NVENC)
```

---

## 4. 노드별 상세 실행 규격

### Node 01. Prompt & Intent Parser
* **Input**: 사용자 자연어 명령 (예: "3초 부근에 지나가는 빨간 차를 지우고 자갈길로 채워줘")
* **Output**:
  * `target_prompt`: 인페인팅 대상 프롬프트 (Positive/Negative)
  * `seed_coordinates`: 초기 추적 키포인트 `[[x, y]]` 또는 바운딩 박스 `[x1, y1, x2, y2]`
  * `frame_range`: 편집 시작/종료 프레임 인덱스
* **실행 엔진**: LLM 파서 (VRAM 미사용)

### Node 02. Frame Decomposer & Preprocessor
* **Input**: 원본 비디오 파일 (`input.mp4`)
* **Output**: 프레임 시퀀스 폴더 (`/workspace/frames/%05d.png`)
* **동작 규격**:
  * 해상도 검사: 너비/높이가 1280을 초과할 경우 비율 유지 후 720p 스케일다운.
  * FPS 유지 또는 24fps로 정규화.

### Node 03. Temporal Mask Tracker (SAM 2 VOS)
* **Input**: 프레임 시퀀스, 시드 좌표/박스
* **모델**: `sam2_hiera_base_plus` (FP16/BF16)
* **동작 규격**:
  1. `init_state(video_path)` 호출
  2. 첫 프레임에 포인트/박스 주입 후 비디오 시퀀스 전체로 마스크 전파
  3. 프레임별 바이너리 마스크 저장 (`/workspace/masks_raw/%05d.png`)
* **필수 후처리 (Exit Protocol)**:
  ```python
  del predictor
  del state
  import gc, torch
  gc.collect()
  torch.cuda.empty_cache()
  ```

### Node 04. Mask Morphological Refinement
* **Input**: 바이너리 마스크 시퀀스 (`/workspace/masks_raw`)
* **Output**: 페더링 마스크 시퀀스 (`/workspace/masks_feathered`)
* **알고리즘**:
  * **Dilation**: Kernel Size 7x7 ~ 9x9 (객체 경계 잔여물 완전 차단)
  * **Gaussian Blur**: Kernel Size 11x11, Sigma=0 (자연스러운 경계 블렌딩)
* **리소스**: CPU 멀티프로세싱 처리

### Node 05. Video DiT Inpainter
* **Input**: 원본 프레임 시퀀스, 페더링 마스크, 인페인팅 프롬프트
* **백엔드 옵션**: CogVideoX-5B-I2V (FP8) 또는 Wan2.1-I2V (NF4/FP8)
* **필수 최적화 파라미터**:
  * `enable_model_cpu_offload()` 적용
  * `enable_vae_tiling()` 활성화 (고해상도 VAE 디코딩 OOM 방지)
  * 추론 청크 단위: 49프레임 단위 슬라이딩 윈도우
* **Output**: 생성된 패치 프레임 시퀀스 (`/workspace/generated_frames`)

### Node 06. Alpha Composite & Final Assembly
* **Input**: 원본 프레임, 생성 프레임, 페더링 마스크, 원본 오디오
* **동작 규격**:
  1. $Output = Original \times (1 - Mask) + Generated \times Mask$ 공식으로 픽셀 블렌딩
  2. FFmpeg NVENC H.264 인코딩 및 원본 오디오 스트림 결합:
  ```bash
  ffmpeg -y -r {fps} -i /workspace/composite/%05d.png -i input.mp4 \
         -map 0:v -map 1:a? -c:v h264_nvenc -pix_fmt yuv420p output.mp4
  ```

---

## 5. 에러 처리 및 Fallback 전략 (Self-Healing)
1. **CUDA OOM (Out of Memory) 발생 시**:
   * 즉시 현재 프로세스 중단 후 캐시 flush.
   * 프레임 해상도를 $0.8\times$ 단계로 축소 (예: 720p -> 540p -> 480p).
   * 재생성 파이프라인 재실행.
2. **트래킹 마스크 유실 (Drifting) 발생 시**:
   * SAM 2 프롬프트 포인트를 단일 프레임이 아닌 중간 키프레임(예: 30f, 60f)에 멀티 포인트로 재주입.