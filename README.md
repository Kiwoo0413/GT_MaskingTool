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

## 📄 라이선스
Apache 2.0 License
