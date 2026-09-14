# Griptape Nodes: Video Mask & Inpainting Library

Griptape Nodes Desktop 전용 **비디오 시공간 객체 마스킹(SAM 2) & 확산 모델 인페인팅(CogVideoX / Wan2.1)** 커스텀 노드 라이브러리입니다.  
16GB VRAM(RTX 4080 등) 환경에서 OOM(메모리 부족) 없이 동작하도록 **순차 메모리 격리(VRAM 0MB Flush)** 전략이 내장되어 있습니다.

---

## 1. 제공 노드 구성

### 🚀 All-in-One 편의 노드
* **`Mask Inpainting (All-in-One)`**:
  * 비디오 경로, 프롬프트(또는 자연어 지시문), 시드 좌표만 입력하면 6개 단계를 내부에서 순차 실행하여 최종 완성 비디오(.mp4)를 출력합니다.

### 🧩 6단계 모듈러 노드 (자유로운 파이프라인 구성용)
1. **`Node 01: Prompt & Intent Parser`**
   * 자연어 명령(예: "3초 부근에 지나가는 빨간 차 지우고 도로로 채워줘")을 분석하여 타겟 프롬프트, 시드 키포인트, 바운딩 박스, 대상 프레임 범위를 추출합니다.
2. **`Node 02: Frame Decomposer`**
   * 입력 비디오를 720p 해상도 스케일링 및 24fps 정규화하여 프레임 시퀀스로 분해합니다.
3. **`Node 03: SAM 2 Temporal Tracker`**
   * Meta SAM 2 모델을 사용하여 비디오 전체 프레임에서 객체를 시공간 추적하고 바이너리 마스크를 생성합니다. 작업 완료 시 **VRAM 0MB Flush**를 자동 수행합니다.
4. **`Node 04: Mask Refinement`**
   * CPU 멀티프로세싱을 통해 마스크 팽창(Dilation) 및 가우시안 블러 페더링을 적용하여 합성 경계 아티팩트를 방지합니다.
5. **`Node 05: Video DiT Inpainter`**
   * CogVideoX-5B (FP8) 또는 Wan2.1 확산 모델을 사용하여 마스킹된 영역을 프롬프트에 맞춰 자연스럽게 인페인팅합니다. (CPU Offload + Tiled VAE 적용)
6. **`Node 06: Alpha Compositor & Assembler`**
   * 원본 비디오와 인페인팅 프레임을 알파 블렌딩하고, FFmpeg NVENC H.264 하드웨어 가속 및 원본 오디오 스트림을 결합하여 최종 비디오를 인코딩합니다.

---

## 2. 친구/동료에게 공유 및 설치 가이드 (Sharing & Installation)

이 라이브러리를 다른 사람(친구, 팀원)의 Griptape Nodes Desktop 환경에 연동하는 방법은 매우 간단합니다.

### [1단계] 라이브러리 폴더 복사
배포 폴더(`griptape-nodes-library-masking` 또는 `GT_MaskingTool`)를 친구의 PC에 전달합니다.  
추천 경로: `D:\AI\GripTape\libraries\griptape-nodes-library-masking`

### [2단계] 의존성 패키지 설치
Griptape 엔진 가상환경 또는 시스템 Python 환경에서 아래 패키지를 설치합니다:
```bash
cd griptape-nodes-library-masking
pip install -r requirements.txt
```
*핵심 패키지*: `torch>=2.4.0`, `diffusers>=0.30.0`, `opencv-python>=4.8.0`, `ffmpeg-python`, `numpy`, `pillow`

### [3단계] Griptape Nodes Desktop에 라이브러리 등록
1. Griptape Nodes 설정 파일 열기:
   * 윈도우 경로: `%APPDATA%\Griptape Nodes\xdg_config_home\griptape_nodes\griptape_nodes_config.json`
2. `libraries_to_register` 배열에 라이브러리의 `griptape_nodes_library.json` 경로를 추가합니다:
   ```json
   "libraries_to_register": [
     "D:\\AI\\GripTape\\libraries\\griptape-nodes-library-standard\\griptape_nodes_library.json",
     "D:\\AI\\GripTape\\libraries\\griptape-nodes-library-masking\\griptape_nodes_library.json"
   ]
   ```

### [4단계] GUI에서 새로고침
1. **Griptape Nodes Desktop** 앱을 엽니다.
2. 좌측 하단의 **`Refresh Libraries`** 버튼을 클릭합니다.
   * (필요 시 우측 상단의 `Engine` -> `Restart` 클릭)
3. 좌측 패널의 **`LIBRARIES`** 및 **`NODES`** 탭에 **`Video Mask & Inpainting`** 카테고리가 나타납니다! 🎉

---

## 3. 워크플로우 구성 가이드

### 방법 1: 초간단 올인원 워크플로우
1. 캔버스 빈 곳에서 더블클릭 후 **`Mask Inpainting (All-in-One)`** 노드를 추가합니다.
2. 입력 파라미터 설정:
   * **Input Video Path**: 원본 영상 절대 경로 (예: `D:\input.mp4`)
   * **Instruction**: `"빨간 자동차 지우고 아스팔트 바닥으로 채워줘"`
   * **Seed Coords**: `"640,360"` (객체 중심 좌표)
3. 실행 버튼을 누르면 인페인팅된 최종 비디오 경로가 **`Final Video Path`** 포트로 출력됩니다.

### 방법 2: 6단계 모듈러 연결 워크플로우
* `Prompt Parser` (출력: prompt, coords)
  ⬇
* `Frame Decomposer` (입력: video_path ➔ 출력: frames_dir, fps)
  ⬇
* `SAM 2 Temporal Tracker` (입력: frames_dir, coords ➔ 출력: raw_masks_dir)
  ⬇
* `Mask Refinement` (입력: raw_masks_dir ➔ 출력: feathered_masks_dir)
  ⬇
* `Video DiT Inpainter` (입력: frames_dir, feathered_masks_dir, prompt ➔ 출력: gen_frames_dir)
  ⬇
* `Alpha Compositor` (입력: orig_video, gen_frames_dir, feathered_masks_dir ➔ 출력: final_video_path)

---

## 4. 하드웨어 권장 사항
* **GPU**: NVIDIA RTX 4080 (16GB VRAM) 이상
* **해상도**: 1280x720 (720p) 이하 권장
* **메모리 보호**: 노드 전환 시 VRAM Flush가 자동 실행되므로 장시간 실행 시에도 VRAM 누수가 발생하지 않습니다.
