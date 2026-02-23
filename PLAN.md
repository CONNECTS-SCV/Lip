# Lip 프로젝트 진행 현황

## 개요
Lip = CHEM 엔진의 standalone CLI 추출본.
REINVENT4 기반 분자 생성 → 도킹/스코어링 → 최적화 파이프라인.

---

## ✅ 완료된 수정 (서버 배포 필요)

### 1. REINVENT4 TOML 생성 전면 수정 (`lip/generator/reinvent.py`)
- `[stage.scoring] type = "geometric_mean"` 추가
- `[[...endpoint]]` 블록 패턴으로 변경 (CHEM 동일)
- ExternalProcess 동일 executable+args 그룹핑
- `[parameters]` 필드: `use_checkpoint`, `purge_memories`, `unique_sequences`, `randomize_smiles`
- `[diversity_filter]` 필드: `bucket_size = 25`, `minscore = 0.4`
- `[[stage]]`에 `termination = "simple"` 추가
- `chkpt_file` 항상 설정
- `TanimotoSimilarity` → `TanimotoDistance` + `use_features: True`
- `external_process_component`에 `property` 파라미터 추가
- `ReinventGenerator` → `ReinventWrapper` 네이밍 수정
- `"DAP"` → `"dap"` (대소문자 수정)

### 2. Docking 전면 수정 (`lip/scoring/docking.py`)
- **Vina grid map 1회 사전 계산** — `__init__`에서 receptor 로드 + map 계산, 분자별 재사용 (기존: 매 분자마다 재생성)
- **`set_ligand_from_file()`** — temp 파일 기반 (기존: `set_ligand_from_string` 버그)
- **obabel `-xrh`** — 수소 추가 플래그 (기존: `-xr`만)
- **`prepare_ligand_pdbqt()` fallback** — `ETKDGv3()` 실패 시 `EmbedParameters()` fallback (기존: None 반환)
- **환경변수** — `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` 설정
- **멀티프로세싱** — `_init_worker` / `_dock_one` 패턴으로 병렬 도킹
- **`get_cached_receptor_pdbqt()`** — CHEM 동일 캐시 패턴 (temp 디렉토리 기반)
- **`[LIP-DOCK]` stderr 로그** — receptor 경로, SMILES 수, 성공/실패 통계, 점수 범위 출력
- **`analyze_pose()` 반환 타입** — `analysis.get()` → `analysis.total_count` (InteractionReport dataclass)

### 3. Shape scoring 프로토콜 수정 (`lip/scoring/shape.py`)
- `sys.path.insert(0, _PROJECT_ROOT)` 추가 (ExternalProcess에서 lip 패키지 인식)
- stdin: JSON → plain text SMILES (REINVENT4 ExternalProcess 프로토콜)
- stdout redirect로 stray output 방지

### 4. Loop 수정 (`lip/loop.py`)
- **Docking 점수 정규화 반전 수정** — `normalize_score(r.score, low=t_high, high=t_low)` (기존: high/low 반전으로 -12→0, -6→1이었음. 수정 후: -12→1, -6→0)
- **Manual mode 파라미터 통일** — `n_rounds` → `n_steps`, `n_molecules_per_round` → `batch_size` (사용자 설정값 양쪽 모드 동일 적용)

### 5. Config 수정 (`lip/config.py`)
- `mode: str = "manual"` (기존: `"managed"`)
- `checkpoint_every: int = 1` (기존: `5`)

### 6. lipScript.tsx 수정 (`Curieus/front/scripts/lipScript.tsx`)
- shape constraint → `constraintLines`에 `type: shape` 추가
- `pocket2mol:` + `paths:` 섹션 생성
- docking `enabled: false` else 블록 추가
- `generator:` 섹션 추가 (`device: cuda`, `batch_size: 50`)
- `filter:` 섹션 추가
- `tpsa/hbd/hba/rotatable_bonds` component_builder 매핑 완료

### 7. 기존 Feature (이전 완료)
- Feature 1: 포켓 자동 탐지 ✅
- Feature 2: 상호작용 분석 통합 ✅
- Feature 3: 합성경로 분석 후처리 ✅

---

## 🚨 서버 배포 시 필수 조치

### gemmi 패키지 설치
- **문제**: 서버 테스트 결과 ALL docking이 `No module named 'gemmi'`로 실패
- **원인**: `meeko` (SMILES→PDBQT 변환) 내부에서 `gemmi`를 import하는데 서버에 미설치
- **해결**: `pip install gemmi` (requirements.txt에 추가 완료)

---

## 🔍 서버에서 확인 필요 사항

### Docking 점수 확인 (gemmi 설치 후)
- 수정 전: `docking_raw=0.0`, `docking_score=0.000010` (모든 도킹 실패)
- 수정 후 예상: `docking_raw=-8.5`, `docking_score=0.4~0.8` (정상 바인딩)
- stderr에 `[LIP-DOCK]` 로그로 성공/실패 통계 확인 가능

### 파라미터 동기화 확인
- `n_steps=10` 설정 시 managed/manual 모두 10스텝/라운드
- `batch_size=100` 설정 시 양쪽 모두 100개/스텝

---

## 아키텍처

```
사용자 → lipScript.tsx (YAML 생성) → 서버
서버 → python -m lip run --config config.yaml --device cuda
  └─ runner.py
      ├─ 포켓 자동 탐지 (pocket.py)
      ├─ Pocket2Mol (shape constraint 있을 때)
      ├─ OptimizationLoop
      │   ├─ managed mode: REINVENT4 staged_learning (TOML → subprocess)
      │   │   └─ ExternalProcess: docking.py (Vina), shape.py (RDKit)
      │   └─ manual mode: generate → filter → score → select 반복
      │       └─ VinaDockingScorer 직접 호출
      └─ 합성 분석 (AiZynthFinder)
```

---

## 파일별 역할

| 파일 | 역할 |
|------|------|
| `lip/config.py` | 설정 시스템 (YAML + CLI merge) |
| `lip/runner.py` | 파이프라인 오케스트레이션 |
| `lip/loop.py` | RL 최적화 루프 (managed/manual) |
| `lip/generator/reinvent.py` | REINVENT4 래퍼 + TOML 생성 |
| `lip/component_builder.py` | 제약조건 → REINVENT4 ScoringComponent 변환 |
| `lip/scoring/docking.py` | Vina 도킹 + ExternalProcess CLI |
| `lip/scoring/shape.py` | Shape similarity + ExternalProcess CLI |
| `lip/scoring/interactions.py` | 단백질-리간드 상호작용 분석 |
| `lip/scoring/synthesis.py` | 합성경로 분석 (AiZynthFinder) |
| `lip/utils/math.py` | REINVENT4 transform dict 빌더 |
| `lip/utils/pocket.py` | 포켓 탐지 (fpocket, 리간드 추출) |

---

## 환경 요구사항

| 환경 | Python | 용도 |
|------|--------|------|
| `lip` (메인) | 3.10+ | REINVENT4, Vina, RDKit, Meeko |
| `Pocket2Mol` (별도 conda) | 3.9 | Pocket2Mol 분자 생성 |

### 핵심 패키지
- REINVENT4: `pip install git+https://github.com/MolecularAI/REINVENT4.git`
- Pocket2Mol: Git 클론 (`/home/connects/SCV_Models/Models/Pocket2Mol`)
- AutoDock Vina: `pip install vina`
- Meeko: `pip install meeko`
- obabel: conda 또는 시스템 설치
