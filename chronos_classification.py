#!/usr/bin/env python3
"""
chronos-forecasting을 사용한 시계열 분류
"""

import pandas as pd
import numpy as np
from chronos import ChronosPipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
import random

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(123)

def convert_to_timeseries(X, sequence_length=52):
    """특성들을 시계열로 변환"""
    n_samples, n_features = X.shape
    # 각 샘플을 시계열로 변환
    X_ts = X.reshape(n_samples, sequence_length, 1)
    return X_ts

def extract_chronos_features(X_ts, pipeline, feature_type='prediction'):
    """chronos를 사용한 특징 추출"""
    features = []
    
    for i in range(len(X_ts)):
        ts = X_ts[i].flatten()  # (52,)
        ts_tensor = torch.tensor(ts, dtype=torch.float32)  # numpy array를 torch tensor로 변환
        
        try:
            if feature_type == 'prediction':
                # 예측 기반 특징
                pred = pipeline.predict(ts_tensor, prediction_length=5)
                if isinstance(pred, torch.Tensor):
                    pred = pred.cpu().numpy()
                features.append(pred.flatten())
                
            elif feature_type == 'embedding':
                # 임베딩 기반 특징
                embedding = pipeline.encode(ts_tensor)
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().numpy()
                features.append(embedding.flatten())
                
            elif feature_type == 'statistical':
                # 통계적 특징 + chronos 예측
                pred = pipeline.predict(ts_tensor, prediction_length=3)
                if isinstance(pred, torch.Tensor):
                    pred = pred.cpu().numpy()
                
                stats = [
                    np.mean(ts), np.std(ts), np.min(ts), np.max(ts),
                    np.median(ts), np.percentile(ts, 25), np.percentile(ts, 75)
                ]
                combined = np.concatenate([pred.flatten(), stats])
                features.append(combined)
                
        except Exception as e:
            print(f"특징 추출 오류 (샘플 {i}): {e}")
            # 오류 발생 시 더미 특징 생성
            if feature_type == 'prediction':
                dummy_features = np.random.random(5)  # prediction_length=5
            elif feature_type == 'embedding':
                dummy_features = np.random.random(128)  # 임베딩 크기 추정
            else:  # statistical
                dummy_features = np.concatenate([np.random.random(3), np.random.random(7)])
            features.append(dummy_features)
    
    return np.array(features)

def chronos_classification():
    """chronos를 사용한 시계열 분류"""
    
    print("chronos-forecasting을 사용한 시계열 분류 시작")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("\n1. 데이터 로드")
    print("-" * 30)
    
    train_df = pd.read_csv("./datasests/train.csv")
    test_df = pd.read_csv("./datasests/test.csv")
    
    X = train_df.drop(columns=["ID", "target"]).values
    y = train_df["target"].values
    X_test = test_df.drop(columns=["ID"]).values
    test_ids = test_df["ID"].values
    
    print(f"훈련 데이터: {X.shape}")
    print(f"테스트 데이터: {X_test.shape}")
    print(f"클래스 수: {len(np.unique(y))}")
    
    # 2. 전처리 (tabtrasformer_classification.py와 동일)
    print("\n2. 데이터 전처리")
    print("-" * 30)
    
    # RobustScaler 적용 (훈련 데이터만으로 fit)
    scaler = RobustScaler()
    X = scaler.fit_transform(X)  # 훈련 데이터로만 fit
    X_test = scaler.transform(X_test)  # 테스트 데이터는 transform만
    
    print("RobustScaler 적용 완료 (훈련 데이터만으로 fit)")
    
    # 3. 시계열로 변환
    print("\n3. 시계열 변환")
    print("-" * 30)
    
    X_ts = convert_to_timeseries(X)
    X_test_ts = convert_to_timeseries(X_test)
    
    print(f"시계열 변환 후 형태: {X_ts.shape}")
    
    # 4. chronos 모델 로드
    print("\n4. chronos 모델 로드")
    print("-" * 30)
    
    try:
        pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-tiny")
        print("✅ chronos 모델 로드 완료")
    except Exception as e:
        print(f"❌ chronos 모델 로드 실패: {e}")
        return None, None, None
    
    # 5. 다양한 특징 추출 방법 실험
    print("\n5. 특징 추출 방법 실험")
    print("-" * 30)
    
    feature_methods = ['prediction', 'statistical']
    best_method = None
    best_score = 0
    
    for method in feature_methods:
        print(f"\n{method} 방법으로 특징 추출 중...")
        
        # 특징 추출
        X_features = extract_chronos_features(X_ts, pipeline, method)
        X_test_features = extract_chronos_features(X_test_ts, pipeline, method)
        
        print(f"추출된 특징 형태: {X_features.shape}")
        
        # 5-Fold CV로 성능 평가
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_features, y), 1):
            X_train_fold = X_features[train_idx]
            X_val_fold = X_features[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            
            # 분류 모델 학습
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            classifier.fit(X_train_fold, y_train_fold)
            
            # 검증
            y_pred_fold = classifier.predict(X_val_fold)
            f1 = f1_score(y_val_fold, y_pred_fold, average='macro')
            fold_scores.append(f1)
        
        avg_score = np.mean(fold_scores)
        print(f"{method} 방법 평균 F1: {avg_score:.4f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_method = method
            best_X_features = X_features
            best_X_test_features = X_test_features
    
    print(f"\n✅ 최적 방법: {best_method} (F1: {best_score:.4f})")
    
    # 6. 최종 모델 학습
    print("\n6. 최종 모델 학습")
    print("-" * 30)
    
    # 다양한 분류 모델 실험
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
    }
    
    best_classifier = None
    best_classifier_score = 0
    best_classifier_name = ""
    
    for name, classifier in classifiers.items():
        # 5-Fold CV
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        fold_scores = []
        
        for train_idx, val_idx in kf.split(best_X_features, y):
            X_train_fold = best_X_features[train_idx]
            X_val_fold = best_X_features[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            
            classifier.fit(X_train_fold, y_train_fold)
            y_pred_fold = classifier.predict(X_val_fold)
            f1 = f1_score(y_val_fold, y_pred_fold, average='macro')
            fold_scores.append(f1)
        
        avg_score = np.mean(fold_scores)
        print(f"{name}: {avg_score:.4f}")
        
        if avg_score > best_classifier_score:
            best_classifier_score = avg_score
            best_classifier = classifier
            best_classifier_name = name
    
    print(f"\n✅ 최적 분류기: {best_classifier_name} (F1: {best_classifier_score:.4f})")
    
    # 7. 최종 예측
    print("\n7. 최종 예측")
    print("-" * 30)
    
    # 전체 데이터로 재학습
    best_classifier.fit(best_X_features, y)
    y_pred = best_classifier.predict(best_X_test_features)
    
    # 확률 예측 (가능한 경우)
    if hasattr(best_classifier, 'predict_proba'):
        y_probs = best_classifier.predict_proba(best_X_test_features)
    else:
        # 확률이 없는 경우 더미 확률 생성
        y_probs = np.zeros((len(y_pred), 21))
        for i, pred in enumerate(y_pred):
            y_probs[i, pred] = 1.0
    
    # 8. 결과 저장 (tabtrasformer_classification.py와 동일한 형식)
    print("\n8. 결과 저장")
    print("-" * 30)
    
    # 제출용 파일 (ID, target만)
    submission = pd.DataFrame({"ID": test_ids, "target": y_pred})
    submission.to_csv("chronos_classification_submission.csv", index=False)
    
    # 상세 파일 (ID, target, 모든 클래스 확률)
    detailed = pd.DataFrame({
        "ID": test_ids,
        "target": y_pred,
        **{f"prob_{i}": y_probs[:, i] for i in range(21)}
    })
    detailed.to_csv("chronos_classification_detailed.csv", index=False)
    
    print(f"✅ chronos 분류 완료!")
    print(f"제출 파일: chronos_classification_submission.csv")
    print(f"상세 결과: chronos_classification_detailed.csv")
    print(f"예측 분포: {np.bincount(y_pred, minlength=21)}")
    
    # 9. 성능 요약
    print("\n9. 성능 요약")
    print("-" * 30)
    print(f"최적 특징 추출 방법: {best_method}")
    print(f"최적 분류기: {best_classifier_name}")
    print(f"검증 F1 점수: {best_classifier_score:.4f}")
    print(f"예측 분포: {np.bincount(y_pred, minlength=21)}")
    
    return y_pred, y_probs, best_X_features, best_X_test_features

if __name__ == "__main__":
    chronos_classification()
