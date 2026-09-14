# Video Mask & Inpainting Nodes for Griptape

Griptape Nodes Desktop 전용 **비디오 시공간 객체 마스킹(SAM 2) & 확산 모델 인페인팅(CogVideoX / Wan2.1)** 커스텀 노드 라이브러리입니다.  
16GB VRAM(RTX 4080 등) 환경에서 OOM(메모리 부족) 없이 동작하도록 **순차 메모리 격리(VRAM 0MB Flush)** 전략이 내장되어 있습니다.

---

## 🌟 주요 기능

* **🚀 Mask Inpainting (All-in-One)**: 단일 노드에서 영상 경로, 프롬프트, 시드 좌표만으로 6단계 파이프라인 자동 실행
* **🧩 6단계 모듈러 노드**:
  1. `Node 01: Prompt & Intent Parser` (자연어 편집 지시문/좌표 파싱)
  2. `Node 02: Frame Decomposer` (720p 스케일링 & 프레임 추출)
  3. `Node 03: SAM 2 Temporal Tracker` (시공간 추적 & **0MB VRAM Flush**)
  4. `Node 04: Mask Refinement` (팽창 및 가우시안 페더링)
  5. `Node 05: Video DiT Inpainter` (CogVideoX FP8 / Wan2.1 비디오 인페인팅)
  6. `Node 06: Alpha Compositor` (알파 블렌딩 & NVENC H.264 하드웨어 인코딩)

---

## 📦 설치 및 Griptape Nodes 연동

### 1. 저장소 복제 (Clone)
Griptape 라이브러리 폴더(`D:\AI\GripTape\libraries\` 또는 원하는 위치)에 클론합니다:
```bash
git clone https://github.com/Kiwoo0413/GT_MaskingTool.git griptape-nodes-library-masking
```

### 2. 의존성 설치
```bash
cd griptape-nodes-library-masking
pip install -r requirements.txt
```

### 3. Griptape Nodes Desktop 등록
`%APPDATA%\Griptape Nodes\xdg_config_home\griptape_nodes\griptape_nodes_config.json`의 `libraries_to_register`에 라이브러리 경로를 추가합니다:
```json
"libraries_to_register": [
  "D:\\AI\\GripTape\\libraries\\griptape-nodes-library-masking\\griptape_nodes_library.json"
]
```

### 4. Griptape Nodes Desktop에서 새로고침
앱 좌측 하단의 **`Refresh Libraries`** 버튼을 클릭(또는 엔진 `Restart`)하면 `LIBRARIES`에 `Video Mask & Inpainting` 카테고리가 나타납니다.

---

## 📖 워크플로우 Step-by-Step 가이드

Griptape 캔버스에서 `+ Create New Workflow`를 클릭한 후, 아래 두 가지 코스 중 원하는 방식으로 워크플로우를 구성합니다.

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
3. **실행**: 상단의 **Run** 버튼을 누르면 내부에서 6단계가 순차 실행되며, **`Final Video Path`** 포트로 완성된 영상 경로가 출력됩니다.

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
  * `feathered_masks_dir` ➔ **Node 05의 `masks_dir`** 및 **Node 06의 `masks_dir`** 2곳으로 연결

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
* **속성 설정 (Properties)**:
  * `Output Video Path`: 최종 저장 경로 (예: `D:\output.mp4`, 비워두면 자동 생성)
  * `Video Codec`: `h264_nvenc`
* **출력 포트 (Outputs)**:
  * `final_video_path` ➔ **최종 완성된 비디오 파일의 경로**가 출력됩니다! 🎬

---

## 4. 하드웨어 권장 사항
* **GPU**: NVIDIA RTX 4080 (16GB VRAM) 이상
* **해상도**: 1280x720 (720p) 이하 권장
* **메모리 보호**: 노드 전환 시 VRAM Flush가 자동 실행되므로 장시간 실행 시에도 VRAM 누수가 발생하지 않습니다.

---

## 📄 라이선스
Apache 2.0 License
