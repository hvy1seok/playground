# Ultimate Ensemble Experiment 사용 가이드

## 개요
iTransformer + TabTransformer 통합 앙상블 시스템으로, 다양한 메타 스태킹 모델과 데이터 증강 옵션을 제공합니다.

## 필수 라이브러리 설치

```bash
# 기본 라이브러리
pip install pandas numpy scikit-learn torch matplotlib seaborn

# 선택적 라이브러리 (메타 모델용)
pip install lightgbm  # LightGBM 메타 모델 사용 시
pip install xgboost   # XGBoost 메타 모델 사용 시
```

## 사용법

### 1. 기본 실행 (Logistic Regression + 데이터 증강)
```bash
python ultimate_ensemble_experiment.py
```

### 2. LightGBM 메타 모델 + 데이터 증강
```bash
python ultimate_ensemble_experiment.py --meta_model lightgbm
```

### 3. XGBoost 메타 모델 + 데이터 증강
```bash
python ultimate_ensemble_experiment.py --meta_model xgboost
```

### 4. Logistic Regression + 데이터 증강 없음
```bash
python ultimate_ensemble_experiment.py --no_augmentation
```

### 5. LightGBM + 데이터 증강 없음
```bash
python ultimate_ensemble_experiment.py --meta_model lightgbm --no_augmentation
```

### 6. XGBoost + 데이터 증강 없음
```bash
python ultimate_ensemble_experiment.py --meta_model xgboost --no_augmentation
```

## 명령줄 옵션

### `--meta_model` (메타 스태킹 모델 선택)
- **선택지**: `logistic`, `lightgbm`, `xgboost`
- **기본값**: `logistic`
- **설명**: OOF 확률을 학습하는 메타 모델 선택

**각 모델 특징:**

#### Logistic Regression
- 빠르고 안정적
- 추가 라이브러리 불필요
- 선형 관계 학습
- Macro-F1 최적화를 위한 `class_weight='balanced'` 적용

#### LightGBM
- 비선형 관계 학습 가능
- 트리 기반 앙상블
- 클래스 가중치 자동 학습
- 200 rounds, early stopping 50

#### XGBoost
- 강력한 비선형 학습
- 트리 기반 앙상블
- 히스토그램 기반 빠른 학습
- 200 rounds, early stopping 50

### `--use_augmentation` / `--no_augmentation` (데이터 증강)
- **기본값**: 증강 사용 (`--use_augmentation`)
- **설명**: 문제 클래스(0, 9, 15)에 대한 데이터 증강 활성화/비활성화

**데이터 증강 기법:**
- **TimeWarp**: 시간축 왜곡 (σ=0.2)
- **Jitter**: 가우시안 노이즈 추가 (σ=0.03)
- **TSMixup**: 시계열 믹스업 (α=0.2)
- **Augmentation Ratio**: 1.5x

## 출력 파일

### 1. `ultimate_ensemble_submission.csv`
제출용 파일
```csv
ID,target
0,5
1,12
...
```

### 2. `ultimate_ensemble_detailed.csv`
모든 클래스 확률 포함
```csv
ID,target,prob_0,prob_1,...,prob_20
0,5,0.01,0.02,...,0.03
...
```

### 3. `ultimate_ensemble_oof.csv`
OOF 예측 결과 (각 모델별)
```csv
ID,true_label,pred_label_itrans,pred_label_tabtrans,pred_label_meta,prob_itrans_0,...
0,5,5,5,5,0.01,...
...
```

### 4. `ultimate_ensemble_meta_info.json`
메타 모델 성능 정보
```json
{
  "method": "LIGHTGBM Meta-Stacking",
  "use_augmentation": true,
  "oof_f1_itransformer": 0.6234,
  "oof_f1_tabtransformer": 0.6145,
  "oof_f1_simple_weighted": 0.6198,
  "oof_f1_meta_stacking": 0.6312,
  "improvement": 0.0114,
  "meta_features": {
    "base_probs": 42,
    "entropy": 2,
    "top1_confidence": 2,
    "margin": 2,
    "agreement": 1,
    "total": 47
  }
}
```

### 5. `oof_confusion_matrix_itransformer.png`
iTransformer OOF Confusion Matrix

### 6. `oof_confusion_matrix_tabtransformer.png`
TabTransformer OOF Confusion Matrix

## 실행 흐름

```
1. 데이터 로드
   ↓
2. 5-Fold Cross Validation
   각 Fold마다:
     ├─ Seed 앙상블 (819, 42, 24)
     │   ├─ 데이터 증강 (선택적)
     │   ├─ iTransformer 학습
     │   └─ TabTransformer 학습
     ↓
3. OOF 평가
   ├─ iTransformer OOF F1
   ├─ TabTransformer OOF F1
   └─ Confusion Matrix 생성
   ↓
4. 메타 스태킹
   ├─ 메타 특징 생성 (47개)
   ├─ 메타 모델 학습 (Logistic/LightGBM/XGBoost)
   └─ OOF 기반 모델 선택
   ↓
5. OVR & Pairwise 보정
   ├─ OVR 전문가 (0, 9, 15)
   └─ Pairwise 전문가 (0↔15, 0↔9, 15↔9)
   ↓
6. 최종 예측 및 저장
```

## 메타 특징 (47개)

1. **기본 확률 (42개)**
   - iTransformer: 21개 클래스 확률
   - TabTransformer: 21개 클래스 확률

2. **엔트로피 (2개)**
   - 각 모델의 예측 불확실성

3. **Top-1 Confidence (2개)**
   - 각 모델의 최대 확률

4. **Margin (2개)**
   - Top-1 - Top-2 차이

5. **Agreement (1개)**
   - 두 모델의 예측 일치 여부

## 성능 비교 팁

다양한 조합을 실행하여 최적 설정을 찾으세요:

```bash
# 1. Logistic + 증강 O
python ultimate_ensemble_experiment.py

# 2. Logistic + 증강 X
python ultimate_ensemble_experiment.py --no_augmentation

# 3. LightGBM + 증강 O
python ultimate_ensemble_experiment.py --meta_model lightgbm

# 4. LightGBM + 증강 X
python ultimate_ensemble_experiment.py --meta_model lightgbm --no_augmentation

# 5. XGBoost + 증강 O
python ultimate_ensemble_experiment.py --meta_model xgboost

# 6. XGBoost + 증강 X
python ultimate_ensemble_experiment.py --meta_model xgboost --no_augmentation
```

각 실행 후 `ultimate_ensemble_meta_info.json`을 비교하여 최고 성능 설정을 선택하세요!

## 예상 실행 시간

- **CPU**: 약 3-4시간 (Seed 3개 × Fold 5개 × 모델 2개)
- **GPU (CUDA)**: 약 1-2시간

## 문제 해결

### LightGBM/XGBoost 설치 오류
```bash
# 윈도우
pip install lightgbm --prefer-binary
pip install xgboost --prefer-binary

# 리눅스/맥
pip install lightgbm
pip install xgboost
```

### CUDA 메모리 부족
- Batch size를 줄이세요 (코드 내 `batch_size=64` → `32`)

### 실행 중 중단
- OOF 결과가 저장되므로 완료된 fold는 재사용 가능
- 필요시 코드 수정하여 특정 fold부터 재시작

## 최적화 기법 요약

✅ **Cosine Attention** (양쪽 Transformer)
✅ **Multi-Seed Ensemble** (819, 42, 24)
✅ **5-Fold CV with OOF**
✅ **RobustScaler** (column-wise)
✅ **Focal Loss** (γ=2.0)
✅ **Pairwise Margin Loss** (0↔15, 0↔9, 15↔9)
✅ **Data Augmentation** (선택적)
✅ **Cosine Annealing Scheduler**
✅ **OOF F1-based Weighting**
✅ **Meta Stacking** (3가지 선택 가능)
✅ **OVR Specialist** (0, 9, 15)
✅ **Pairwise Routing** (Top-2 tie-break)

## 라이선스

MIT License

