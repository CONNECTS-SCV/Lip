# Lip 프로젝트 진행상황

## 개요

CHEM 엔진(`c:\Users\alsxo\Desktop\CHEM`)을 기반으로 독립 RL 기반 분자 생성/최적화 모델을 처음부터 새로 설계/구현한 프로젝트.

---

## Phase 1-6: 핵심 구현 (완료)

### Phase 1: 프로젝트 뼈대 + 설정
- `pyproject.toml` - 패키지 메타데이터
- `lip/config.py` - YAML + CLI 3단계 우선순위 설정 시스템 (CLI > YAML > 기본값)
- `configs/default.yaml` - 기본 설정 파일

### Phase 2: 유틸리티
- `lip/utils/chem.py` - SMILES 검증, 물성 계산, Morgan FP, Lipinski/PAINS 필터, REINVENT4 호환성 필터
- `lip/utils/conformer.py` - RDKit ETKDGv3 기반 3D 구조 생성
- `lip/utils/io.py` - CSV/SDF/JSON I/O (통합 `_write_csv()`)
- `lip/utils/math.py` - sigmoid, normalize_score, REINVENT4 transform 빌더
- `lip/utils/pocket.py` - fpocket 기반 포켓 감지 + 리간드 추출

### Phase 3: Constraint 시스템
- `lip/constraints/base.py` - Template method 패턴 (`score()` → `_score_mol()`)
- `lip/constraints/registry.py` - `@register` 데코레이터 자동 등록
- `lip/constraints/property_range.py` - MW, LogP, TPSA 범위
- `lip/constraints/qed.py` - QED 약물유사성
- `lip/constraints/sa.py` - 합성 접근성 (SA Score)
- `lip/constraints/similarity.py` - Tanimoto 유사도
- `lip/constraints/smarts.py` - SMARTS 패턴 매칭
- `lip/constraints/shape.py` - 3D shape similarity (CrippenO3A)
- `lip/constraints/dynamicbind.py` - DynamicBind 통합

### Phase 4: Scoring
- `lip/scoring/docking.py` - Vina/GNINA 도킹 + 레지스트리 패턴 + persistent file cache
- `lip/scoring/shape.py` - ExternalProcess shape scorer
- `lip/scoring/interactions.py` - 6종 상호작용 분석 (HBond, Hydrophobic, PiStacking, SaltBridge, HalogenBond, CationPi)
- `lip/scoring/synthesis.py` - AiZynthFinder 합성 가능성
- `lip/scoring/aggregator.py` - weighted_sum + Pareto ranking

### Phase 5: Generator + Loop
- `lip/generator/base.py` - BaseGenerator ABC
- `lip/generator/reinvent.py` - REINVENT4 subprocess 래퍼 (sampling + staged_learning)
- `lip/loop.py` - RL 최적화 루프 (managed + manual 모드)
- `lip/component_builder.py` - constraint → REINVENT4 ScoringComponent 변환 (레지스트리 기반)
- `lip/pocket2mol/generate.py` - Pocket2Mol 참조 분자 생성 + MaxMin 다양성 선택

### Phase 6: 진입점
- `lip/__main__.py` - CLI (run, score, pocket2mol 서브커맨드)
- `lip/runner.py` - 메인 오케스트레이터
- `lip/analysis/metrics.py` - RL bias 메트릭

---

## R1-R9: DRY / SRP / 확장성 리팩토링 (완료)

| ID | 내용 | 수정 파일 |
|----|------|-----------|
| R1 | BaseConstraint 템플릿 메서드 패턴 (`_score_mol()`) | `base.py` + 6개 constraint |
| R2 | Sigmoid/Transform 함수 공통화 → `utils/math.py` | `math.py`, `sa.py`, `loop.py`, `reinvent.py` |
| R3 | CSV save/append 통합 → `_write_csv()` | `utils/io.py` |
| R4 | ComponentBuilder 분리 (SRP) | `component_builder.py`, `loop.py` |
| R5 | Docking 레지스트리 패턴 | `scoring/docking.py` |
| R6 | Constraint threshold를 params로 이동 | `qed.py`, `shape.py`, `sa.py` |
| R7 | Fingerprint import 통일 → `get_morgan_fp()` | `chem.py`, `similarity.py`, `generate.py` |
| R8 | Interaction type 파라미터화 | `interactions.py` |
| R9 | constraint → component 레지스트리 | `component_builder.py` |

---

## G1-G5: 미구현 기능 갭 구현 (완료)

### G1: Manual Mode 전체 구현 [치명적 → 완료]

| 항목 | 파일 | 설명 |
|------|------|------|
| `sample()` | `generator/reinvent.py:198` | REINVENT4 sampling 모드 TOML 생성 + CSV 파싱 |
| `_generate_batch()` | `loop.py:348` | sample → is_valid → canonicalize → dedup |
| `_select_diverse()` | `loop.py:369` | Greedy diversity selection (Morgan FP Tanimoto) |

### G2: Managed Mode 완성 [치명적 → 완료]

| 항목 | 파일 | 설명 |
|------|------|------|
| Per-step 콜백 | `loop.py:143` | chunk 단위 RL CSV 파싱 → RoundResult 생성 + 콜백 |
| Inception buffer | `loop.py:251` | cross-chunk dedup + retention (top/random) |
| Bracket filter | `utils/chem.py:180` | `is_reinvent_compatible()` - REINVENT4 vocabulary 필터 |
| TOML retention | 변경 불필요 | CHEM 참조 확인: Python 레벨 관리가 정확 |

### G3: 상호작용 분석 확장 [높음 → 완료]

| 항목 | 파일 | 설명 |
|------|------|------|
| PiStacking | `scoring/interactions.py:183` | 방향족 고리 centroid 거리 ≤ 5.5A |
| CationPi | `scoring/interactions.py:199` | 양이온-방향족 양방향 (리간드→단백질, ARG/LYS/HIS→리간드) |
| 헬퍼 | `scoring/interactions.py:117` | `_get_ring_centroid`, `_centroid_dist`, `_get_aromatic_rings`, `_POS_RESIDUES` |

### G4: Pocket Detection 확장 [중간 → 완료]

| 항목 | 파일 | 설명 |
|------|------|------|
| Pocket 6필드 | `utils/pocket.py:20` | druggability, n_alpha_spheres, polar_sasa, apolar_sasa, residue_ids, residue_names |
| _info.txt 파서 | `utils/pocket.py:88` | Druggability Score, Alpha Spheres, SASA 추출 |
| 잔기 파서 | `utils/pocket.py:141` | `_parse_pocket_residues()` - pocket*_atm.pdb ATOM 줄 파싱 |

### G5: 기타 소규모 [낮음 → 완료]

| 항목 | 파일 | 설명 |
|------|------|------|
| Persistent receptor cache | `scoring/docking.py:99` | 3단계: in-memory → file (.cache/) → fresh conversion |
| REINVENT bracket filter | `utils/chem.py:180` | G2.1에서 처리 |
| Inception TOML retention | 변경 불필요 | CHEM 참조 확인 완료 |

---

## 프로젝트 구조

```
Lip/
├── pyproject.toml
├── configs/
│   └── default.yaml
├── lip/
│   ├── __init__.py
│   ├── __main__.py            # CLI: python -m lip
│   ├── config.py              # YAML + CLI 설정 시스템
│   ├── runner.py              # 메인 오케스트레이터
│   ├── loop.py                # RL 최적화 루프 (managed + manual)
│   ├── component_builder.py   # constraint → REINVENT4 component 변환
│   ├── generator/
│   │   ├── base.py            # BaseGenerator ABC
│   │   └── reinvent.py        # REINVENT4 subprocess 래퍼
│   ├── constraints/
│   │   ├── base.py            # BaseConstraint (템플릿 메서드)
│   │   ├── registry.py        # @register 자동 등록
│   │   ├── property_range.py  # MW, LogP, TPSA
│   │   ├── qed.py
│   │   ├── sa.py
│   │   ├── similarity.py      # Tanimoto
│   │   ├── smarts.py
│   │   ├── shape.py           # 3D shape
│   │   └── dynamicbind.py
│   ├── scoring/
│   │   ├── docking.py         # Vina + GNINA + registry + persistent cache
│   │   ├── shape.py           # ExternalProcess shape
│   │   ├── interactions.py    # 6종 상호작용 (HBond, Hydrophobic, PiStacking, SaltBridge, HalogenBond, CationPi)
│   │   ├── synthesis.py       # AiZynthFinder
│   │   └── aggregator.py      # weighted_sum + Pareto
│   ├── pocket2mol/
│   │   └── generate.py        # Pocket2Mol 참조 분자 생성
│   ├── utils/
│   │   ├── chem.py            # SMILES 검증, 물성, FP, REINVENT4 호환성
│   │   ├── conformer.py       # 3D 구조 생성
│   │   ├── io.py              # CSV/SDF/JSON I/O
│   │   ├── math.py            # sigmoid, normalize, transform 빌더
│   │   └── pocket.py          # fpocket 감지 (10필드) + 리간드 추출
│   └── analysis/
│       └── metrics.py         # RL bias 메트릭
```

---

## 핵심 설계 패턴

| 패턴 | 적용 위치 | 설명 |
|------|-----------|------|
| Template Method | `BaseConstraint.score()` → `_score_mol()` | SMILES 파싱/에러처리 공통화 |
| Registry/Factory | `DOCKING_REGISTRY`, `COMPONENT_FACTORIES`, `@register` | 확장 가능한 팩토리 |
| SRP 분리 | `ComponentBuilder`, `ScoreAggregator` | 루프 오케스트레이션과 변환/집계 분리 |
| 3단계 캐시 | `BaseDockingScorer.get_receptor_pdbqt()` | in-memory → file → conversion |
| Config 우선순위 | `LipConfig.merge_cli()` | CLI args > YAML > defaults |

---

## Lip이 CHEM보다 나은 점

| 기능 | 설명 |
|------|------|
| Pocket2Mol 통합 | CHEM에 없는 기능 |
| Config 시스템 | YAML + CLI 3단계 우선순위 |
| ComponentBuilder 분리 | 깔끔한 SRP |
| Resume/Checkpoint | RunState 기반 이어하기 |
| Math 유틸리티 | sigmoid/normalize 공통화 |
| Docking 레지스트리 | 확장 가능한 팩토리 패턴 |
| Constraint 템플릿 메서드 | BaseConstraint._score_mol() |
| Persistent receptor cache | MD5 기반 파일 캐시 |
| Interaction 파라미터화 | 커스텀 상호작용 타입 주입 |

---

## 의존성

```
필수: numpy, rdkit, meeko, vina, pyyaml
선택: aizynthfinder, psutil, pandas
외부 CLI: obabel, gnina, fpocket, REINVENT4
```

---

## CLI 사용법

```bash
# 기본 실행
python -m lip run --config config.yaml

# CLI 인자로 직접 지정
python -m lip run --receptor protein.pdb --pocket-center 10 20 30 --mode managed

# 이어하기
python -m lip run --resume results/run_001/

# 스코어링
python -m lip score --receptor protein.pdb --pocket-center 10 20 30 --smiles "CCO"

# Pocket2Mol
python -m lip pocket2mol --receptor protein.pdb --pocket-center 10 20 30
```
