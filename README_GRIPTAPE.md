# Griptape Nodes: Video Mask & Inpainting Library

Griptape Nodes Desktop 전용 **비디오 시공간 객체 마스킹(SAM 2) & 확산 모델 인페인팅(CogVideoX / Wan2.1)** 커스텀 노드 라이브러리입니다.  
16GB VRAM(RTX 4080 등) 환경에서 OOM(메모리 부족) 없이 동작하도록 **순차 메모리 격리(VRAM 0MB Flush)** 전략이 내장되어 있습니다.

---

## 1. 제공 노드 구성

### 🚀 All-in-One 편의 노드
* **`Mask Inpainting (All-in-One)`**:
  * 비디오 경로, 프롬프트(또는 자연어 지시문), 시드 좌표만 입력하면 6개 단계를 내부에서 순차 실행하여 최종 완성 비디오(.mp4)를 출력합니다.

### 🧩 6단계 모듈러 노드 (자유로운 파이프라인 구성용)
1. **`Node 01: Prompt & Intent Parser`**: 자연어 명령 분석, 타겟 프롬프트, 시드 키포인트, 바운딩 박스, 프레임 범위 추출
2. **`Node 02: Frame Decomposer`**: 720p 해상도 스케일링 및 24fps 정규화 후 프레임 시퀀스 추출
3. **`Node 03: SAM 2 Temporal Tracker`**: SAM 2 시공간 객체 추적 & 바이너리 마스크 생성 (**0MB VRAM Flush** 내장)
4. **`Node 04: Mask Refinement`**: CPU 멀티프로세싱 마스크 팽창(Dilation) 및 가우시안 블러 페더링
5. **`Node 05: Video DiT Inpainter`**: CogVideoX-5B (FP8) 또는 Wan2.1 확산 모델 인페인팅 (CPU Offload + Tiled VAE)
6. **`Node 06: Alpha Compositor & Assembler`**: 알파 블렌딩 & NVENC H.264 하드웨어 가속 비디오/오디오 결합

---

## 2. 친구/동료에게 공유 및 설치 가이드 (Sharing & Installation)

### [1단계] 저장소 복제 (Git Clone)
Griptape 라이브러리 폴더(`D:\AI\GripTape\libraries\` 또는 원하는 위치)에 클론합니다:
```bash
git clone https://github.com/Kiwoo0413/GT_MaskingTool.git griptape-nodes-library-masking
```

### [2단계] 의존성 패키지 설치
Griptape 가상환경 또는 시스템 Python 환경에서 패키지를 설치합니다:
```bash
cd griptape-nodes-library-masking
pip install -r requirements.txt
```
*핵심 패키지*: `torch>=2.4.0`, `diffusers>=0.30.0`, `opencv-python>=4.8.0`, `ffmpeg-python`, `numpy`, `pillow`

### [3단계] Griptape Nodes Desktop에 라이브러리 등록
1. 설정 파일 열기: `%APPDATA%\Griptape Nodes\xdg_config_home\griptape_nodes\griptape_nodes_config.json`
2. `libraries_to_register` 배열에 라이브러리의 `griptape_nodes_library.json` 경로 추가:
   ```json
   "libraries_to_register": [
     "D:\\AI\\GripTape\\libraries\\griptape-nodes-library-standard\\griptape_nodes_library.json",
     "D:\\AI\\GripTape\\libraries\\griptape-nodes-library-masking\\griptape_nodes_library.json"
   ]
   ```

### [4단계] GUI에서 새로고침
1. **Griptape Nodes Desktop** 앱을 엽니다.
2. 좌측 하단의 **`Refresh Libraries`** 버튼을 클릭합니다. (또는 우측 상단 `Engine` -> `Restart`)
3. 사이드바의 **`LIBRARIES`** 및 **`NODES`** 탭에 **`Video Mask & Inpainting`** 카테고리가 나타납니다! 🎉

---

## 3. 워크플로우 Step-by-Step 가이드

Griptape 캔버스에서 `+ Create New Workflow`를 클릭한 후, 아래 두 가지 코스 중 원하는 방식을 선택해 노드를 연결합니다.

---

### 🚀 코스 A: 초간단 "All-in-One" 1개 노드 워크플로우

복잡한 선 연결 없이 단 1개의 노드로 전체 파이프라인을 실행합니다.

```text
┌────────────────────────────────────────────────────────┐
│             Mask Inpainting (All-in-One)               │
│                                                        │
│ [입력 설정]                                            │
│ • Input Video Path : "D:\test\input.mp4"               │
│ • Instruction      : "빨간 자동차 지우고 도로로 채워줘"  │
│ • Seed Coords      : "640,360"                         │
│ • Backend          : "cogvideox"                       │
│                                                        │
│                                [출력 포트]             │
│                                Final Video Path ───────▶ (완성된 MP4 경로)
│                                Red Overlay Video Path ─▶ (빨간색 마스크 검수 MP4)
│                                Masks Dir ──────────────▶ (알파 마스크 폴더)
└────────────────────────────────────────────────────────┘
```

#### Step-by-Step 설정:
1. **노드 추가**: 캔버스 빈 곳을 더블클릭하거나 좌측 노드 목록에서 **`Mask Inpainting (All-in-One)`**을 추가합니다.
2. **파라미터 입력**:
   * **`Input Video Path`**: 원본 비디오 파일의 절대 경로 (예: `D:\videos\input.mp4`)
   * **`Instruction`**: 자연어 편집 지시문 (예: `"빨간 차 지우고 도로 바닥으로 채워줘"`)
   * **`Seed Coords`**: 편집 대상 객체의 중심 좌표 (예: `640,360`)
   * **`Backend`**: 사용할 생성 모델 선택 (`cogvideox` 또는 `wan2.1`)
   * **`Output Video Path`**: 저장 경로 (비워두면 임시 폴더에 자동 생성)
3. **실행**: 상단의 **Run** 버튼을 누르면 내부에서 6단계가 순차 실행되며, **`Final Video Path`**(완성 영상)와 함께 **`Red Overlay Video Path`**(빨간색 마스크 검수 비디오)가 동시에 출력됩니다!

---

### 🧩 코스 B: 세부 제어가 가능한 "6단계 모듈러" 연결 워크플로우

중간 결과물(마스크, 생성 프레임)을 직접 확인하거나 각 단계를 정밀 제어할 때 사용하는 표준 방식입니다.

#### 🌐 전체 연결 와이어맵 (Connection Diagram)

```text
[Node 01: Prompt Parser]
   ├─ prompt ─────────────────────────────────────────────────────────┐
   ├─ negative_prompt ──────────────────────────────────────────────┐ │
   └─ out_seed_coords ──┐                                           │ │
                        │                                           │ │
[Node 02: Frame Decomposer]                                         │ │
   └─ frames_dir ───────┼─────────────┐                             │ │
                        ▼             ▼                             │ │
               [Node 03: SAM 2]       │                             │ │
                  └─ raw_masks_dir    │                             │ │
                        │             │                             │ │
                        ▼             │                             │ │
               [Node 04: Mask Refine] │                             │ │
                  └─ feathered_masks ─┼──────────────┐              │ │
                                      ▼              ▼              ▼ ▼
                            [Node 05: Video DiT] ──▶ [Node 06: Compositor]
                               generated_frames         └─ Final Video (.mp4)
```

---

#### 1단계. Node 01: Prompt & Intent Parser
* **역할**: 자연어 명령에서 인페인팅 대상 프롬프트와 객체 좌표를 자동 추출합니다.
* **입력 설정 (Properties)**:
  * `Instruction`: `"3초 부근에 지나가는 빨간 차 지우고 도로로 채워줘"`
  * `Seed Coordinates`: *(선택)* 수동 좌표 직접 입력 시 `"640,360"`
* **출력 포트 (Outputs)**:
  * `prompt` ➔ **Node 05의 `prompt`**로 연결
  * `negative_prompt` ➔ **Node 05의 `negative_prompt`**로 연결
  * `out_seed_coords` ➔ **Node 03의 `seed_coords`**로 연결
  * `out_box_coords` ➔ **Node 03의 `box_coords`**로 연결
  * `out_frame_range` ➔ **Node 03의 `frame_range`**로 연결

---

#### 2단계. Node 02: Frame Decomposer
* **역할**: 원본 비디오를 16GB VRAM에 맞게 720p로 정규화하고 개별 이미지 프레임 시퀀스로 분해합니다.
* **입력 설정 (Properties)**:
  * `Video Path`: 원본 영상 파일 경로 (예: `D:\videos\input.mp4`)
  * `Target Resolution`: `720` (기본값 권장 - VRAM 최적화)
  * `Normalize to 24 FPS`: `True`
* **출력 포트 (Outputs)**:
  * `frames_dir` ➔ **Node 03의 `frames_dir`** 및 **Node 05의 `frames_dir`** 2곳으로 연결

---

#### 3단계. Node 03: SAM 2 Temporal Tracker
* **역할**: 시드 좌표의 객체를 비디오 전 프레임에 걸쳐 추적하고, 종료 시 **0MB VRAM Flush**를 수행합니다.
* **입력 포트 연결 (Inputs)**:
  * `Frames Directory` ⬅ **Node 02의 `frames_dir`** 연결
  * `Seed Coords` ⬅ **Node 01의 `out_seed_coords`** 연결
  * `Box Coords` ⬅ **Node 01의 `out_box_coords`** 연결
  * `Frame Range` ⬅ **Node 01의 `out_frame_range`** 연결
* **출력 포트 (Outputs)**:
  * `raw_masks_dir` ➔ **Node 04의 `raw_masks_dir`**로 연결
  * `vram_status`: VRAM 1GB 미만 정리 완료 상태 확인용

---

#### 4단계. Node 04: Mask Refinement
* **역할**: CPU 멀티프로세싱으로 마스크 경계를 팽창(Dilation) 및 가우시안 페더링하여 합성 시 경계 아티팩트를 방지합니다.
* **입력 포트 연결 (Inputs)**:
  * `Raw Masks Dir` ⬅ **Node 03의 `raw_masks_dir`** 연결
  * `Dilation Kernel`: `7` (기본값)
  * `Blur Kernel`: `11` (기본값)
* **출력 포트 (Outputs)**:
  * `feathered_masks_dir` ➔ **Node 05의 `masks_dir`** 및 **Node 06의 `masks_dir`** 2곳으로 연결 (Grayscale)
  * `red_masks_dir` ➔ **순수 Red 컬러로 채색된 알파 마스크 시퀀스 폴더** (R 채널 매트 검수용)

---

#### 5단계. Node 05: Video DiT Inpainter
* **역할**: VRAM 격리 후 CogVideoX (FP8) 모델로 마스크 영역을 새로운 프레임으로 자연스럽게 생성합니다.
* **입력 포트 연결 (Inputs)**:
  * `Frames Directory` ⬅ **Node 02의 `frames_dir`** 연결
  * `Masks Directory` ⬅ **Node 04의 `feathered_masks_dir`** 연결
  * `Inpainting Prompt` ⬅ **Node 01의 `prompt`** 연결
  * `Negative Prompt` ⬅ **Node 01의 `negative_prompt`** 연결
* **속성 설정 (Properties)**:
  * `Model Backend`: `cogvideox` (또는 `wan2.1`)
  * `Guidance Scale`: `6.0`
  * `Inference Steps`: `30`
* **출력 포트 (Outputs)**:
  * `generated_frames_dir` ➔ **Node 06의 `generated_frames_dir`**로 연결

---

#### 6단계. Node 06: Alpha Compositor & Assembler
* **역할**: 인페인팅 프레임과 원본 프레임을 알파 마스크로 합성하고 NVENC H.264 하드웨어 가속으로 최종 비디오를 인코딩합니다.
* **입력 포트 연결 (Inputs)**:
  * `Original Video Path`: 원본 영상 파일 경로 (`D:\videos\input.mp4`)
  * `Generated Frames Dir` ⬅ **Node 05의 `generated_frames_dir`** 연결
  * `Feathered Masks Dir` ⬅ **Node 04의 `feathered_masks_dir`** 연결
  * `Frames Directory`: *(선택)* **Node 02의 `frames_dir`** 연결 (Red Overlay 비디오 생성 시)
* **속성 설정 (Properties)**:
  * `Output Video Path`: 최종 저장 경로 (예: `D:\output.mp4`, 비워두면 자동 생성)
  * `Video Codec`: `h264_nvenc`
* **출력 포트 (Outputs)**:
  * `final_video_path` ➔ **최종 완성된 비디오 파일의 경로** 출력 🎬
  * `red_overlay_video_path` ➔ **마스크 영역이 빨간색(Red)으로 오버레이된 검수용 비디오 파일 경로** 출력 🔴

---

## 4. 하드웨어 권장 사항
* **GPU**: NVIDIA RTX 4080 (16GB VRAM) 이상
* **해상도**: 1280x720 (720p) 이하 권장
* **메모리 보호**: 노드 전환 시 VRAM Flush가 자동 실행되므로 장시간 실행 시에도 VRAM 누수가 발생하지 않습니다.
