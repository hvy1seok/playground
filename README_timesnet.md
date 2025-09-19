# TimesNet 분류 모델 사용 가이드

이 프로젝트는 Time-Series-Library의 TimesNet 모델을 사용하여 CSV 데이터를 분류하는 예제입니다.

## 파일 구조

```
├── timesnet_classification.py    # 메인 학습 스크립트
├── test_timesnet.py             # 테스트 스크립트
├── README_timesnet.md           # 이 파일
├── datasests/                   # 데이터 폴더
│   ├── train.csv               # 학습 데이터
│   ├── test.csv                # 테스트 데이터
│   └── sample_submission.csv   # 제출 양식
└── Time-Series-Library/        # TimesNet 라이브러리
```

## 데이터셋 정보

- **학습 데이터**: 21,686개 샘플, 52개 특성, 21개 클래스 (0~20)
- **테스트 데이터**: 15,000개 샘플, 52개 특성
- **문제 유형**: 다중 클래스 분류

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements_wandb.txt
```

또는 개별 설치:
```bash
pip install torch pandas numpy scikit-learn wandb matplotlib seaborn
```

### 2. Wandb 설정 (선택사항)

Wandb 로깅을 사용하려면 먼저 설정해야 합니다:

```bash
# 방법 1: 자동 설정 스크립트 사용
python setup_wandb.py

# 방법 2: 수동 설정
wandb login
```

### 3. 테스트 실행

먼저 모든 구성 요소가 올바르게 작동하는지 테스트합니다:

```bash
python test_timesnet.py
```

### 4. 모델 학습

테스트가 성공하면 모델을 학습합니다:

#### 기본 학습 (권장 설정)
```bash
python timesnet_classification.py
```

#### 정규화 방법 선택
```bash
# StandardScaler (FFT에 가장 적합)
python timesnet_classification.py --scaling standard

# 시계열별 정규화
python timesnet_classification.py --scaling sequence

# MinMaxScaler
python timesnet_classification.py --scaling minmax

# RobustScaler (기존 방식)
python timesnet_classification.py --scaling robust

# 스케일링 없음 (원본 데이터 그대로 사용)
python timesnet_classification.py --scaling none
```

#### 스케줄러 선택
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

#### 하이퍼파라미터 조정
```bash
# 큰 모델로 학습
python timesnet_classification.py --e_layers 3 --d_model 96 --d_ff 192 --top_k 7 --num_kernels 8

# 빠른 학습 (작은 모델)
python timesnet_classification.py --e_layers 1 --d_model 32 --d_ff 64 --epochs 20

# 학습률 조정
python timesnet_classification.py --learning_rate 0.0005 --batch_size 32

# Wandb 비활성화
python timesnet_classification.py --no_wandb
```

#### 실험 실행 스크립트
```bash
# 미리 정의된 실험들 실행
python run_experiments.py
```

## 모델 특징

### TimesNet 아키텍처
- **FFT 기반 주기성 탐지**: Fast Fourier Transform을 사용하여 시계열의 주기성을 자동으로 탐지
- **2D Convolution**: 1D 시계열을 2D로 변환하여 복잡한 패턴 학습
- **Inception Block**: 다양한 크기의 커널을 사용하여 다양한 시간 스케일의 패턴 포착

### 데이터 변환
- 52개 특성을 52개 시점의 1차원 시계열로 변환
- TimesNet이 FFT를 통해 특성들 간의 주기성과 패턴을 학습할 수 있도록 함

### 하이퍼파라미터
- **모델 차원**: 64
- **Feed-forward 차원**: 128
- **TimesBlock 레이어**: 2개
- **Top-k 주기**: 5개
- **학습률**: 0.001
- **배치 크기**: 64
- **에포크**: 50 (Early Stopping 적용)

## Wandb 로깅 기능

### 로깅되는 정보
- **학습 메트릭**: 각 에포크별 학습/검증 손실, 정확도, Macro F1 스코어, 학습률
- **상세 평가**: Macro/Micro/Weighted F1 스코어, 클래스별 F1 스코어
- **혼동 행렬**: 검증 데이터에 대한 분류 성능 시각화
- **예측 결과 테이블**: 테스트 데이터의 예측 결과와 확률
- **클래스 분포**: 예측된 클래스들의 분포
- **모델 설정**: 하이퍼파라미터 및 모델 구성

### Wandb에서 확인할 수 있는 것들
1. **실시간 학습 곡선**: 손실, 정확도, Macro F1 스코어 변화 추이
2. **상세 성능 메트릭**: Macro/Micro/Weighted F1 스코어 비교
3. **클래스별 성능**: 각 클래스의 F1 스코어 분석
4. **혼동 행렬**: 분류 성능 시각화
5. **모델 성능 비교**: 다른 실험과의 성능 비교
6. **예측 결과 분석**: 테스트 데이터의 예측 결과 상세 분석
7. **아티팩트**: CSV 파일 자동 업로드 및 버전 관리

## 출력 파일

학습 완료 후 다음 파일들이 생성됩니다:

- `timesnet_submission.csv`: 테스트 데이터에 대한 예측 결과
- Wandb 아티팩트: 예측 결과 CSV가 자동으로 업로드됨

## 성능 모니터링

학습 중 다음 정보가 출력됩니다:
- 각 에포크별 학습/검증 손실
- 검증 정확도 및 Macro F1 스코어
- 상세 평가 결과 (Macro/Micro/Weighted F1)
- 클래스별 F1 스코어 (상위 10개)
- 학습 시간
- Early Stopping 상태 (Macro F1 기준)

## 문제 해결

### 일반적인 문제들

1. **Import 오류**: Time-Series-Library 경로가 올바른지 확인
2. **CUDA 오류**: GPU가 없으면 자동으로 CPU 사용
3. **메모리 부족**: 배치 크기를 줄이거나 모델 크기를 줄임

### 성능 개선 팁

1. **하이퍼파라미터 튜닝**:
   - `d_model`, `d_ff` 크기 조정
   - `e_layers` 수 조정
   - `top_k` 값 조정

2. **데이터 전처리**:
   - 다른 정규화 방법 시도 (StandardScaler, MinMaxScaler)
   - 특성 선택 또는 차원 축소

3. **앙상블**:
   - 여러 모델의 예측 결과 결합
   - 다른 시계열 변환 방법 조합

## 참고 자료

- [TimesNet 논문](https://openreview.net/pdf?id=ju_Uqw384Oq)
- [Time-Series-Library GitHub](https://github.com/thuml/Time-Series-Library)
