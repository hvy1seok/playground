# Ultimate Ensemble Implementation - 모델 구성

## 🎯 최종 구현 요약

### **모델 아키텍처**

#### **1. iTransformer (공식 Time-Series-Library 구현)**
```python
from models.iTransformer import Model as iTransformerOfficial

# 특징:
✅ 공식 검증된 구현
✅ Cosine Attention 지원 (use_cosine_attention=True)
✅ 입력: [B, seq_len=52, enc_in=1]
✅ Feature-wise attention (각 변수를 독립적으로 처리)
✅ 시계열 특화 구조
```

**설정:**
```python
itrans_config = iTransformerConfig(
    input_dim=52,
    num_classes=21,
    d_model=128,        # 모델 차원
    n_heads=4,          # 어텐션 헤드
    e_layers=4,         # 인코더 레이어
    dropout=0.3,
    use_cosine_attention=True  # Cosine Attention 활성화
)
```

#### **2. TabTransformer (Multi-head Cosine Attention)**
```python
class TabTransformer(nn.Module):
    # 특징:
    ✅ Multi-head Cosine Attention (4 heads)
    ✅ 입력: [B, input_dim=52]
    ✅ FFN with 4x expansion + GELU
    ✅ 표준 Transformer 블록 구조
```

**구조:**
```python
Embedding → [
    Multi-head Cosine Attention
    → LayerNorm
    → FFN (4x expansion, GELU)
    → LayerNorm
] × 4 layers → Classifier
```

---

## 📊 두 모델 비교

| 항목 | iTransformer (공식) | TabTransformer (커스텀) |
|------|---------------------|------------------------|
| **구현** | Time-Series-Library | 직접 구현 |
| **입력 형태** | `[B, 52, 1]` | `[B, 52]` |
| **Attention** | Cosine (공식) | Multi-head Cosine (4 heads) |
| **특화** | 시계열 Feature-wise | 표준 Transformer |
| **검증** | 논문 검증됨 | 실험적 |
| **복잡도** | 높음 | 중간 |

---

## 🔧 실행 방법

### **기본 실행**
```bash
# Logistic Regression 메타 모델 + 데이터 증강
python ultimate_ensemble_experiment.py

# LightGBM 메타 모델 + 데이터 증강
python ultimate_ensemble_experiment.py --meta_model lightgbm

# XGBoost 메타 모델 + 증강 없음
python ultimate_ensemble_experiment.py --meta_model xgboost --no_augmentation
```

---

## 📈 학습 파이프라인

### **전체 흐름**

```
1. 데이터 로드 & 전처리 (RobustScaler)
   ↓
2. 5-Fold Cross Validation
   각 Fold마다:
     ├─ Seed 앙상블 (819, 42, 24)
     │   각 Seed마다:
     │     ├─ 데이터 증강 (선택적)
     │     ├─ iTransformer 학습 (공식 라이브러리)
     │     │   └─ Cosine Attention + Focal Loss + Pairwise Loss
     │     └─ TabTransformer 학습 (Multi-head)
     │         └─ Cosine Attention + Focal Loss + Pairwise Loss
     ↓
3. Seed 앙상블 (logits 평균)
   ↓
4. Fold 앙상블 (logits 평균)
   ↓
5. Cross-Model 앙상블 (OOF F1 기반 가중치)
   ↓
6. 확률 보정 (Isotonic Regression)
   ↓
7. 임계값 최적화 (클래스별)
   ↓
8. Logit Adjustment (희소 클래스 부스팅)
   ↓
9. 메타 스태킹 (Logistic/LightGBM/XGBoost)
   - 47개 메타 특징
   - class_weight='balanced' for Macro-F1
   ↓
10. OVR & Pairwise 보정 (0, 9, 15)
    ↓
최종 예측
```

---

## 🎯 핵심 개선 사항

### **1. iTransformer → 공식 라이브러리**
- ✅ 간소화 버전에서 **공식 검증 버전**으로 업그레이드
- ✅ Cosine Attention 유지
- ✅ 더 나은 성능 기대

### **2. TabTransformer → Multi-head Cosine**
- ✅ 단순 Cosine → **Multi-head Cosine Attention** (4 heads)
- ✅ 더 복잡한 FFN (4x expansion + GELU)
- ✅ 표준 Transformer 블록 구조

### **3. 손실 함수**
```python
Total Loss = CE Loss 
           + 0.5 × Focal Loss (γ=2.0)
           + 0.1 × Pairwise Margin Loss (0↔15, 0↔9, 15↔9)
```

---

## 💪 예상 성능

```
iTransformer (공식, Cosine Attention)     : 0.68-0.72
TabTransformer (Multi-head Cosine)        : 0.65-0.70

Seed 앙상블 (3 seeds)                      : +0.01-0.02
Fold 앙상블 (5 folds)                      : +0.01-0.02
Cross-Model 앙상블                         : +0.01-0.03
메타 스태킹 (LightGBM/XGBoost)             : +0.02-0.04
OVR & Pairwise 보정                        : +0.01-0.02

최종 예상: 0.72-0.78 Macro-F1 🎯
```

---

## 📁 출력 파일

```
ultimate_ensemble_submission.csv          - 최종 제출 파일
ultimate_ensemble_detailed.csv            - 모든 클래스 확률
ultimate_ensemble_oof.csv                 - OOF 예측 (3개 모델)
ultimate_ensemble_meta_info.json          - 메타 모델 정보
oof_confusion_matrix_itransformer.png     - iTransformer 혼동 행렬
oof_confusion_matrix_tabtransformer.png   - TabTransformer 혼동 행렬
```

---

## 🚀 실행 시간

```
CPU: 약 4-5시간
  - iTransformer: 더 느림 (공식 구현)
  - TabTransformer: 빠름
  - 3 seeds × 5 folds × 2 models = 30번 학습

GPU (CUDA): 약 1.5-2시간
  - iTransformer: 빠름
  - TabTransformer: 매우 빠름
```

---

## ⚡ 핵심 차이점 요약

### **이전 버전 (간소화)**
```python
# 간소화된 iTransformer
class iTransformer(nn.Module):
    # 직접 구현
    # 빠르지만 검증 부족
```

### **현재 버전 (공식 + 개선)**
```python
# 공식 iTransformer + Multi-head TabTransformer
from models.iTransformer import Model as iTransformerOfficial

# 공식 검증 + 개선된 TabTransformer
# 느리지만 더 나은 성능
# Cosine Attention 양쪽 모두 지원
```

---

## 🎓 추천 전략

### **실험용**
```bash
# 빠른 테스트 (no augmentation)
python ultimate_ensemble_experiment.py --no_augmentation

# 빠른 조합 테스트
python ultimate_ensemble_experiment.py --meta_model lightgbm
```

### **최종 제출용**
```bash
# 모든 최적화 활성화
python ultimate_ensemble_experiment.py --meta_model xgboost

# 또는 LightGBM
python ultimate_ensemble_experiment.py --meta_model lightgbm
```

---

## 🔍 디버깅 팁

### **라이브러리 확인**
```bash
# Time-Series-Library 설치 확인
ls Time-Series-Library/models/iTransformer.py

# 필요한 패키지 확인
pip install torch scikit-learn lightgbm xgboost pandas numpy matplotlib seaborn
```

### **Fallback 모드**
```python
# iTransformer 라이브러리를 찾을 수 없으면 자동으로 간소화 버전 사용
if not ITRANSFORMER_AVAILABLE:
    print("⚠️ 간소화 버전 사용")
```

---

## ✅ 완료된 기능

- [x] iTransformer 공식 라이브러리 통합
- [x] Cosine Attention 지원
- [x] Multi-head Cosine Attention (TabTransformer)
- [x] Focal Loss + Pairwise Margin Loss
- [x] 3 Seeds × 5 Folds 앙상블
- [x] 메타 스태킹 (Logistic/LightGBM/XGBoost)
- [x] OVR & Pairwise 보정
- [x] 확률 보정 & 임계값 최적화
- [x] 데이터 증강 (선택적)
- [x] Early stopping (patience=20)
- [x] Epoch별 진행상황 출력

최고의 성능을 기대합니다! 🎉

