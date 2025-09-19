# Time-Series-Library 모델 분류 학습 가이드

이 프로젝트는 Time-Series-Library의 다양한 모델들을 사용하여 시계열 분류 작업을 수행합니다.

## 🚀 지원 모델

### 1. TimesNet
- **특징**: FFT 기반 주기성 탐지 + 2D CNN
- **장점**: 시계열의 주기성을 자동으로 탐지하고 학습
- **적합한 경우**: 주기성이 있는 시계열 데이터
- **권장 설정**: `e_layers=4`, `d_model=64`

### 2. iTransformer
- **특징**: Inverted Transformer (변수별 어텐션)
- **장점**: 변수 간 관계를 효율적으로 학습
- **적합한 경우**: 다변량 시계열 데이터
- **권장 설정**: `e_layers=2`, `d_model=64`

### 3. PatchTST
- **특징**: Patch-based Time Series Transformer
- **장점**: Vision Transformer 스타일의 패치 기반 처리
- **적합한 경우**: 긴 시계열 데이터
- **권장 설정**: `e_layers=3`, `d_model=128`, `patch_len=16`

## 📦 설치 및 설정

### 1. 의존성 설치
```bash
pip install -r requirements_wandb.txt
```

### 2. Wandb 설정 (선택사항)
```bash
python setup_wandb.py
```

## 🎯 사용법

### 기본 사용법

#### TimesNet
```bash
# 기본 설정
python timesnet_classification.py

# 최적 설정 (1순위)
python timesnet_classification.py --e_layers 4

# 큰 모델
python timesnet_classification.py --e_layers 4 --d_model 96 --d_ff 192
```

#### iTransformer
```bash
# 기본 설정
python itransformer_classification.py

# 커스텀 설정
python itransformer_classification.py --e_layers 2 --d_model 64 --n_heads 8
```

#### PatchTST
```bash
# 기본 설정
python patchtst_classification.py

# 커스텀 설정
python patchtst_classification.py --e_layers 3 --d_model 128 --patch_len 16 --stride 8
```

### 정규화 방법 선택

```bash
# StandardScaler (FFT에 가장 적합, 권장)
python timesnet_classification.py --scaling standard

# 시계열별 정규화
python timesnet_classification.py --scaling sequence

# MinMaxScaler
python timesnet_classification.py --scaling minmax

# RobustScaler
python timesnet_classification.py --scaling robust

# 스케일링 없음
python timesnet_classification.py --scaling none
```

### 스케일링 범위 선택

```bash
# 전체 데이터 기준 정규화 (기본값)
python timesnet_classification.py --scaling standard --scaling_scope global

# 컬럼별 정규화
python timesnet_classification.py --scaling standard --scaling_scope column
```

### 스케줄러 선택

```bash
# 코사인 어닐링 (권장)
python timesnet_classification.py --scheduler cosine

# 플래토 스케줄러
python timesnet_classification.py --scheduler plateau

# 워밍업 + 코사인
python timesnet_classification.py --scheduler warmup_cosine

# 스텝 스케줄러
python timesnet_classification.py --scheduler step
```

### 모델 비교 실험

```bash
# 모든 모델 비교
python compare_models.py

# 도움말 보기
python show_model_help.py
```

## ⚙️ 주요 옵션들

### 공통 옵션
| 옵션 | 설명 | 기본값 | 선택지 |
|------|------|--------|--------|
| `--scaling` | 정규화 방법 | standard | standard, minmax, robust, sequence, none |
| `--scaling_scope` | 스케일링 범위 | global | global(전체), column(컬럼별) |
| `--scheduler` | 스케줄러 | cosine | cosine, step, plateau, exponential, warmup_cosine |
| `--e_layers` | 레이어 수 | 모델별 다름 | 1-4 |
| `--d_model` | 모델 차원 | 모델별 다름 | 32, 64, 96, 128 |
| `--d_ff` | Feed-forward 차원 | 모델별 다름 | 64, 128, 192, 256 |
| `--learning_rate` | 학습률 | 0.001 | 0.0001-0.01 |
| `--batch_size` | 배치 크기 | 모델별 다름 | 16, 32, 64, 128 |
| `--epochs` | 학습 에포크 수 | 50 | 10-100 |
| `--use_wandb` | Wandb 로깅 | True | True/False |

### 모델별 특화 옵션

#### TimesNet
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--top_k` | FFT에서 선택할 상위 k개 주기 | 5 |
| `--num_kernels` | Inception block의 커널 수 | 6 |

#### PatchTST
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--patch_len` | 패치 길이 | 16 |
| `--stride` | 패치 스트라이드 | 8 |

## 📊 성능 비교

### 하이퍼파라미터 실험 결과 (TimesNet 기준)
| 순위 | 정규화 | e_layers | d_model | d_ff | val_macro_f1 |
|------|--------|----------|---------|------|--------------|
| 1 | standard | 4 | 64 | 128 | 최고 |
| 2 | standard | 3 | 64 | 128 | 두 번째 |
| 3 | standard | 2 | 96 | 192 | 세 번째 |

### 모델별 특징 비교
| 모델 | 파라미터 수 | 학습 시간 | 메모리 사용량 | 적합한 데이터 |
|------|-------------|-----------|---------------|---------------|
| TimesNet | 중간 | 중간 | 중간 | 주기성 있는 시계열 |
| iTransformer | 적음 | 빠름 | 적음 | 다변량 시계열 |
| PatchTST | 많음 | 느림 | 많음 | 긴 시계열 |

## 🔧 고급 사용법

### 1. 커스텀 실험 설정
```bash
# 큰 모델 + 워밍업 코사인 스케줄러
python timesnet_classification.py --e_layers 4 --d_model 96 --scheduler warmup_cosine --epochs 40

# 빠른 실험 (작은 모델)
python itransformer_classification.py --e_layers 1 --d_model 32 --epochs 20

# PatchTST 최적화
python patchtst_classification.py --e_layers 4 --d_model 128 --patch_len 16 --stride 8
```

### 2. Wandb 없이 실행
```bash
python timesnet_classification.py --no_wandb
```

### 3. 배치 크기 조정
```bash
# GPU 메모리가 부족한 경우
python timesnet_classification.py --batch_size 32

# GPU 메모리가 충분한 경우
python timesnet_classification.py --batch_size 128
```

## 📈 결과 분석

### 출력 파일들
- `{model}_submission.csv`: 테스트 예측 결과
- `model_comparison_results.csv`: 모델 비교 실험 결과
- Wandb 대시보드: 실시간 학습 모니터링

### 평가 메트릭
- **Accuracy**: 전체 정확도
- **Macro F1**: 클래스별 F1 점수의 평균 (불균형 데이터에 적합)
- **Micro F1**: 전체 F1 점수
- **Weighted F1**: 클래스별 가중 평균 F1 점수

## 🚨 주의사항

1. **데이터 형식**: `train.csv`와 `test.csv` 파일이 필요합니다.
2. **GPU 메모리**: 큰 모델은 충분한 GPU 메모리가 필요합니다.
3. **정규화**: FFT 기반 모델(TimesNet)에는 `standard` 정규화를 권장합니다.
4. **배치 크기**: GPU 메모리에 따라 조정하세요.

## 🆘 문제 해결

### 일반적인 오류들
1. **CUDA 메모리 부족**: `--batch_size`를 줄이세요.
2. **NumPy 버전 오류**: `pip install numpy==1.24.3`으로 다운그레이드하세요.
3. **Wandb 로그인 오류**: `python setup_wandb.py`를 실행하세요.

### 성능 최적화 팁
1. **정규화 방법**: TimesNet은 `standard`, PatchTST는 `sequence`를 시도해보세요.
2. **스케줄러**: `cosine` 또는 `warmup_cosine`을 사용하세요.
3. **Early Stopping**: `--patience` 값을 조정하여 과적합을 방지하세요.

## 📚 참고 자료

- [TimesNet 논문](https://arxiv.org/abs/2210.02186)
- [iTransformer 논문](https://arxiv.org/abs/2310.06625)
- [PatchTST 논문](https://arxiv.org/pdf/2211.14730.pdf)
- [Time-Series-Library GitHub](https://github.com/thuml/Time-Series-Library)
