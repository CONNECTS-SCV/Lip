# CHEM → Lip 끊어진 연결 복구 계획

## Context

Lip 프로젝트는 CHEM(원본 플랫폼)의 `engine/` 코어 로직을 추출한 CLI 도구입니다.
추출 과정에서 3가지 핵심 기능의 연결이 끊어졌습니다:

1. **포켓 자동 탐지** — `pocket.py`에 `detect_pockets()`, `extract_ligands()` 구현됨, 파이프라인 미연결
2. **상호작용 분석** — `interactions.py`에 `analyze_pose()` 구현됨, 도킹 후 호출 안 됨
3. **합성경로 분석** — `synthesis.py`에 `analyze_synthesis()` 구현됨, 파이프라인 미연결

이 세 기능을 Lip의 파이프라인에 연결하여 CHEM과 동등한 기능을 CLI 환경에서 제공합니다.

---

## Feature 1: 포켓 자동 탐지

`--pocket-center`를 지정하지 않으면 PDB에서 자동으로 포켓 중심을 탐지합니다.

### 1-1. `lip/config.py` — `auto_pocket` 필드 추가

`LipConfig` 클래스에 `auto_pocket: bool = True` 추가.

`_from_dict()`과 `merge_cli()`의 top-level scalars에 `"auto_pocket"` 추가.

### 1-2. `lip/__main__.py` — CLI 플래그 추가

`_add_common_args()`에 추가:
```python
parser.add_argument("--no-auto-pocket", dest="auto_pocket",
                    action="store_false", default=None)
```

### 1-3. `lip/runner.py` — 자동 탐지 로직

`run()` 함수에서 config 저장 후, Pocket2Mol 전에 호출:
```python
if pocket_center == [0.0, 0.0, 0.0] and config.auto_pocket and config.receptor_pdb:
    _auto_detect_pocket(config)
    config.save_yaml(output_dir / "config.yaml")  # 탐지 결과 반영
```

`_auto_detect_pocket(config)` 함수:
1. `extract_ligands(receptor_pdb)` 시도 → 공결정 리간드 center 사용 (가장 신뢰성 높음)
2. 리간드 없으면 `detect_pockets(receptor_pdb, fpocket_bin=config.paths.fpocket)` → druggability 최고 포켓
3. 둘 다 실패 → `RuntimeError` ("--pocket-center를 지정하거나 fpocket을 설치하세요")

**재사용:** `lip/utils/pocket.py:extract_ligands()` (L190), `detect_pockets()` (L34)

### 1-4. `configs/default.yaml`

`pocket_center` 다음에 `auto_pocket: true` 추가.

---

## Feature 2: 상호작용 분석 통합

도킹 성공 시 protein-ligand 상호작용 분석 → `interaction_count`를 별도 스코어링 컴포넌트로 사용.
이미 `DockingConfig.analyze_interactions: bool = True`와 `interaction_weight: float = 0.3`이 config에 존재하나 미연결.

### 2-1. `lip/loop.py` — Manual 모드 (직접 API 호출)

`_score_constraints()` 메서드의 docking 블록 뒤에 추가:
```python
if self.config.docking.analyze_interactions:
    interaction_scores, interaction_passed, interaction_raw = (
        self._analyze_interactions_batch(docking_results)
    )
    results["interactions"] = ConstraintResult(...)
    weights["interactions"] = self.config.docking.interaction_weight
```

새 메서드 `_analyze_interactions_batch(docking_results)`:
- `_get_protein_mol()`로 protein_mol 1회 로드 후 캐싱
- 성공한 DockingResult마다 `analyze_pose(protein_pdb, pose_pdbqt, smiles, protein_mol)` 호출
- 정규화: `min(1.0, count / 10.0)`
- `DockingResult.interaction_count` 필드에 값 저장

새 메서드 `_get_protein_mol()`:
- `self._protein_mol_cache`에 1회 캐싱
- `Chem.MolFromPDBFile(receptor_pdb, removeHs=False)`

**재사용:** `lip/scoring/interactions.py:analyze_pose()` (L59) — `protein_mol` 파라미터 지원

### 2-2. `lip/scoring/docking.py` — Managed 모드 (ExternalProcess)

`vina_external_process_main()` (L295) 수정 — `--analyze-interactions` 시 실제 호출:

```python
# protein_mol 1회 로드
if args.analyze_interactions:
    protein_mol = Chem.MolFromPDBFile(args.receptor, removeHs=False)

# 각 SMILES 루프 내:
if args.analyze_interactions and result.success and result.pose_pdbqt:
    report = analyze_pose(..., protein_mol=protein_mol)
    interaction_counts.append(report.total_count)
else:
    interaction_counts.append(0)
```

### 2-3. `lip/component_builder.py` — REINVENT4 컴포넌트

`build_docking_component()` 수정:
- `analyze_interactions=True`면 args에 `" --analyze-interactions"` 추가

`build_interaction_component()` 새 메서드:
- docking과 **동일한 executable+args** (REINVENT4가 같은 subprocess 그룹핑 → 1회만 실행)
- `property_name="interaction_count"`
- transform: `{"type": "sigmoid", "low": 0.0, "high": 10.0, "k": 0.4}`
- weight: `config.docking.interaction_weight`

`build_all()` 수정:
- `build_interaction_component()` 결과도 components에 추가

---

## Feature 3: 합성경로 분석 (후처리)

최적화 완료 후 상위 N개 분자의 역합성 경로 분석. RL 루프 외부 후처리.

### 3-1. `lip/config.py` — SynthesisConfig 추가

```python
@dataclass
class SynthesisConfig:
    enabled: bool = True
    top_n: int = 10
    time_limit: int = 120
    iteration_limit: int = 100
    max_routes: int = 5
```

`LipConfig`에 `synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)` 추가.
`_from_dict()`에 synthesis 파싱, `merge_cli()` mapping에 synthesis 키 추가.

### 3-2. `lip/__main__.py` — CLI 지원

`_add_synthesis_args()` 추가:
- `--no-synthesis` (action=store_false, dest=synthesis_enabled)
- `--synthesis-top-n` (type=int)
- `--synthesis-time-limit` (type=int)

run 서브파서에 `_add_synthesis_args(run_parser)` 추가.

`synthesis` 서브커맨드 추가:
- `--smiles` 또는 `--input-csv`로 입력
- `analyze_batch()` → 결과 출력 + JSON 저장

### 3-3. `lip/runner.py` — 후처리

`run()`에서 `loop.run()` 완료 후:
```python
if config.synthesis.enabled and rounds:
    _run_synthesis_analysis(config, rounds, output_dir)
```

`_run_synthesis_analysis()`:
1. `is_available()` 체크 → 미설치 시 skip (경고만)
2. 전 라운드 분자 수집 → 점수순 → 중복 제거 → top_n개
3. `analyze_batch()` 호출
4. `output_dir/synthesis_analysis.json`에 저장
5. 요약 로그

**재사용:** `lip/scoring/synthesis.py:analyze_batch()` (L112), `is_available()` (L17), `lip/utils/io.py:save_json()` (L101)

### 3-4. `configs/default.yaml`

```yaml
synthesis:
  enabled: true
  top_n: 10
  time_limit: 120
  iteration_limit: 100
  max_routes: 5
```

---

## 수정 파일 요약

| 파일 | Feature | 변경 |
|------|---------|------|
| `lip/config.py` | F1, F3 | `auto_pocket`, `SynthesisConfig` 추가 |
| `lip/runner.py` | F1, F3 | `_auto_detect_pocket()`, `_run_synthesis_analysis()` 추가 |
| `lip/__main__.py` | F1, F3 | `--no-auto-pocket`, synthesis CLI 추가 |
| `lip/loop.py` | F2 | `_analyze_interactions_batch()`, `_get_protein_mol()` 추가 |
| `lip/scoring/docking.py` | F2 | `vina_external_process_main()`에서 `analyze_pose()` 실제 호출 |
| `lip/component_builder.py` | F2 | `build_interaction_component()` 추가, `build_all()` 수정 |
| `configs/default.yaml` | F1, F3 | `auto_pocket`, `synthesis` 섹션 추가 |

## 구현 순서

1. **Feature 1** (포켓 자동 탐지) — 파이프라인 진입점, 독립적
2. **Feature 2** (상호작용 분석) — 스코어링 루프 수정, 가장 복잡
3. **Feature 3** (합성 후처리) — 후처리, 독립적

## 검증

1. `python -c "from lip.config import LipConfig; c = LipConfig(); print(c.auto_pocket, c.synthesis.enabled)"`
2. `python -c "from lip.runner import _auto_detect_pocket"` — import 확인
3. `python -m lip synthesis --smiles "CCO"` — AiZynthFinder 미설치 안내
4. `configs/default.yaml` → `LipConfig.from_yaml()` → 새 필드 정상 파싱

## 구현 상태: ✅ Feature 1, 2, 3 모두 완료

---

## CHEM vs Lip 외부 도구 호출 방식 비교

### 도구별 코어 호출 방식 (동일)

| 도구 | 호출 방식 | CHEM | Lip |
|------|-----------|:----:|:---:|
| **Vina** | Python API (`from vina import Vina`) | ✅ | ✅ |
| **GNINA** | `subprocess.run()` 바이너리 | ✅ | ✅ |
| **fpocket** | `subprocess.run()` 바이너리 | ✅ | ✅ |
| **obabel** | `subprocess.run()` 바이너리 | ✅ | ✅ |
| **Pocket2Mol** | `subprocess.run()` + conda env python | ✅ | ✅ |
| **REINVENT4** | `subprocess.Popen()` + TOML 설정 | ✅ | ✅ |
| **AiZynthFinder** | Python API import | ✅ | ✅ |
| **DynamicBind** | `subprocess.run()` python script | ✅ | ✅ |
| **상호작용 분석** | RDKit API + obabel subprocess | ✅ | ✅ |
| **Shape similarity** | stdin/stdout JSON (ExternalProcess) | ✅ | ✅ |

> 코어 호출 방식은 동일. 차이는 그 위의 운영 레이어.

### CHEM에만 있는 운영 레이어 (Lip에서 제거됨)

#### 1. Vina 멀티프로세싱
- **CHEM**: `ProcessPoolExecutor` + 워커당 Vina 인스턴스 초기화, grid map 재사용
- **Lip**: 단일 스레드, 호출자에게 병렬화 위임
- **파일**: CHEM `engine/scoring/vina_docking.py`, Lip `lip/scoring/docking.py`

#### 2. Celery 비동기 태스크
- **CHEM**: 모든 무거운 작업이 `@celery_app.task`로 래핑, 웹 UI에서 비동기 실행
- **Lip**: 직접 동기 함수 호출 (`result = run(config)`)
- **파일**: CHEM `backend/worker/tasks.py`

#### 3. DB 연동 & 진행률 추적
- **CHEM**: SQLite에 분자별 점수/포즈 저장, Job 상태 실시간 추적, 프론트엔드 업데이트
- **Lip**: 파일 기반 출력 (CSV/JSON), `run_state.json` 체크포인트만
- **파일**: CHEM `backend/routers/results.py`

#### 4. 프로세스 트리 관리
- **CHEM**: `psutil` 기반 재귀적 프로세스 트리 종료, PID 파일 크로스-프로세스 취소
- **Lip**: 기본 `terminate()` → `wait(10)` → `kill()` 패턴
- **파일**: CHEM `engine/generator/reinvent_wrapper.py`

#### 5. GNINA CUDA 환경 관리
- **CHEM**: `LD_LIBRARY_PATH`에 conda lib 자동 주입 (`libcudnn.so.9` 등)
- **Lip**: 시스템 PATH에 의존 (환경 관리 없음)
- **파일**: CHEM `engine/scoring/gnina_docking.py`

### 구조적 차이 요약

```
CHEM:  도구 호출 ← ProcessPoolExecutor ← Celery Task ← DB 연동 ← 환경 관리
Lip:   도구 호출 ← 직접 함수 호출
```

### 향후 보강 가능 항목 (미정)

- [ ] Vina 멀티프로세싱 (가장 큰 성능 차이)
- [ ] 프로세스 트리 관리 강화 (psutil 기반)
- [ ] GNINA CUDA 환경 자동 관리
