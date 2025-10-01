# 체크포인트 기능 가이드

## 🎯 체크포인트 시스템 개요

학습 중 중단되어도 **완료된 Fold부터 재개**할 수 있습니다.

---

## 📋 주요 기능

### **1. 자동 체크포인트 저장**
각 Fold 완료 후 자동으로 저장:
```
ultimate_checkpoints/
├── checkpoint_fold_1.pkl  ✅ Fold 1 완료
├── checkpoint_fold_2.pkl  ✅ Fold 2 완료
├── checkpoint_fold_3.pkl  ✅ Fold 3 완료
├── checkpoint_fold_4.pkl  ✅ Fold 4 완료
└── checkpoint_fold_5.pkl  ✅ Fold 5 완료
```

### **2. 저장되는 데이터**
각 체크포인트에 포함:
- ✅ OOF 확률 (iTransformer, TabTransformer)
- ✅ Test logits (iTransformer, TabTransformer)
- ✅ Fold 번호 및 타임스탬프

---

## 🚀 사용 방법

### **시나리오 1: 처음부터 시작**
```bash
python ultimate_ensemble_experiment.py --meta_model xgboost
```

### **시나리오 2: 중단 후 재개**
```bash
# 1차 실행 (Fold 1, 2 완료 후 중단)
python ultimate_ensemble_experiment.py --meta_model xgboost
# Ctrl+C로 중단

# 2차 실행 (Fold 3부터 자동 재개)
python ultimate_ensemble_experiment.py --meta_model xgboost --resume
```

**재개 시 동작:**
```
⏩ Fold 1 체크포인트 발견! 로드 중...
  ✅ Checkpoint 로드: ultimate_checkpoints/checkpoint_fold_1.pkl
  iTransformer OOF F1: 0.8456
  TabTransformer OOF F1: 0.8634

⏩ Fold 2 체크포인트 발견! 로드 중...
  ✅ Checkpoint 로드: ultimate_checkpoints/checkpoint_fold_2.pkl
  ...

🔄 Fold 3 학습 시작...  ← 여기부터 새로 학습
```

### **시나리오 3: 처음부터 다시 시작**
```bash
# 기존 체크포인트 삭제 후 시작
python ultimate_ensemble_experiment.py --meta_model xgboost --clear_checkpoints
```

---

## 📊 체크포인트 동작 흐름

```
FOLD 1
  ├─ Seed 819 학습
  ├─ Seed 42 학습
  ├─ Seed 24 학습
  └─ ✅ checkpoint_fold_1.pkl 저장
  
FOLD 2
  ├─ Seed 819 학습
  ├─ Seed 42 학습
  ├─ Seed 24 학습
  └─ ✅ checkpoint_fold_2.pkl 저장

(중단 발생)

재실행 (--resume)
  ⏩ FOLD 1 로드 (학습 생략)
  ⏩ FOLD 2 로드 (학습 생략)
  🔄 FOLD 3 학습 시작
  🔄 FOLD 4 학습 시작
  🔄 FOLD 5 학습 시작
```

---

## 💡 명령어 예시

### **1. 새로 시작**
```bash
# LightGBM + 증강 + 처음부터
python ultimate_ensemble_experiment.py --meta_model lightgbm

# XGBoost + 증강 없음 + 처음부터
python ultimate_ensemble_experiment.py --meta_model xgboost --no_augmentation
```

### **2. 재개**
```bash
# 중단된 지점부터 재개
python ultimate_ensemble_experiment.py --meta_model xgboost --resume

# 증강 없이 재개
python ultimate_ensemble_experiment.py --meta_model xgboost --no_augmentation --resume
```

### **3. 초기화**
```bash
# 체크포인트 삭제 후 새로 시작
python ultimate_ensemble_experiment.py --meta_model lightgbm --clear_checkpoints

# 또는 수동 삭제
rm -rf ultimate_checkpoints/  # Linux/Mac
rmdir /s ultimate_checkpoints  # Windows
```

---

## 🔍 오류 해결 완료!

### **이전 오류 (수정됨)**
```python
# 문제: 같은 데이터를 훈련/검증으로 사용
ovr_models = train_ovr_classifiers(X_full, y, X_full, y, device)
#                                  ↑ 훈련      ↑ 검증 (같음!)
# 결과: Val F1 = 0.0000
```

### **수정 후**
```python
# OVR 학습 시 train_test_split으로 분리
X_ovr_train, X_ovr_val, y_ovr_train, y_ovr_val = train_test_split(
    X_scaled_full, y_binary, test_size=0.2, stratify=y_binary, random_state=123
)
# 결과: Val F1 > 0.5 (정상 학습)
```

---

## 📈 학습 결과 보존

### **완료된 학습 (날아가지 않음!)**
```
✅ 5-Fold × 3 Seeds 학습 완료
   - iTransformer (공식): OOF F1 = 0.8614
   - TabTransformer (Multi-head): OOF F1 = 0.8712
   - 메타 스태킹 (XGBoost): OOF F1 = 0.9707
```

**이제 두 가지 옵션:**

#### **옵션 1: 체크포인트 없이 재실행 (1.5-2시간)**
```bash
python ultimate_ensemble_experiment.py --meta_model xgboost --no_augmentation
```
- 5개 Fold 모두 다시 학습
- OVR & Pairwise 보정 포함
- 최종 결과 저장

#### **옵션 2: 빠른 테스트용 스크립트 (5분)**
이미 메타 스태킹이 0.9707이므로, OVR 보정만 추가하는 간단한 스크립트 작성 가능

---

## 🎯 추천 실행 방법

### **처음 실행**
```bash
# 체크포인트 기능 활성화 (중단 대비)
python ultimate_ensemble_experiment.py --meta_model xgboost
```

### **중단 후**
```bash
# 자동으로 완료된 Fold는 건너뛰고 재개
python ultimate_ensemble_experiment.py --meta_model xgboost --resume
```

### **다시 처음부터**
```bash
# 체크포인트 삭제 후 새로 시작
python ultimate_ensemble_experiment.py --meta_model xgboost --clear_checkpoints
```

---

## 📊 예상 시간

| 단계 | 시간 (GPU) | 체크포인트 |
|------|-----------|-----------|
| Fold 1 | ~20분 | ✅ 저장 |
| Fold 2 | ~20분 | ✅ 저장 |
| Fold 3 | ~20분 | ✅ 저장 |
| Fold 4 | ~20분 | ✅ 저장 |
| Fold 5 | ~20분 | ✅ 저장 |
| 메타 스태킹 | ~5분 | - |
| OVR 보정 | ~10분 | - |
| **총** | **~2시간** | - |

**재개 시:**
- Fold 1-3 완료 → Fold 4-5만 학습 (~40분)

---

## ✅ 수정 완료 사항

1. ✅ **체크포인트 저장/로드** 기능 추가
2. ✅ **OVR 훈련/검증 분리** 수정
3. ✅ **Pairwise 훈련/검증 분리** 수정
4. ✅ **best_state None 오류** 수정
5. ✅ **iTransformer 공식 라이브러리** 통합
6. ✅ **--resume, --clear_checkpoints** 옵션 추가

이제 안전하게 실행할 수 있습니다! 🚀

