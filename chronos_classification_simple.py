#!/usr/bin/env python3
"""
chronos-forecasting을 사용한 시계열 분류 (간단 버전)
chronos 설치 없이도 실행 가능한 대안 제공
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
import torch
import random

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(123)

def extract_timeseries_features(X_ts, feature_type='statistical'):
    """시계열 특징 추출 (chronos 없이)"""
    features = []
    
    for i in range(len(X_ts)):
        ts = X_ts[i].flatten()  # (52,)
        
        if feature_type == 'statistical':
            # 통계적 특징
            stats = [
                np.mean(ts), np.std(ts), np.min(ts), np.max(ts),
                np.median(ts), np.percentile(ts, 25), np.percentile(ts, 75),
                np.var(ts), np.ptp(ts),  # 분산, 범위
                np.mean(np.diff(ts)), np.std(np.diff(ts)),  # 차분 통계
                np.corrcoef(ts[:-1], ts[1:])[0, 1] if len(ts) > 1 else 0,  # 자기상관
            ]
            features.append(stats)
            
        elif feature_type == 'fourier':
            # FFT 기반 특징
            fft = np.fft.fft(ts)
            fft_magnitude = np.abs(fft)
            fft_phase = np.angle(fft)
            
            # 주요 주파수 성분
            top_freqs = np.argsort(fft_magnitude)[-10:]  # 상위 10개 주파수
            features.append(np.concatenate([
                fft_magnitude[top_freqs],
                fft_phase[top_freqs]
            ]))
            
        elif feature_type == 'wavelet':
            # 웨이블릿 변환 (간단한 버전)
            # Haar 웨이블릿 근사
            def haar_transform(signal):
                if len(signal) <= 1:
                    return signal
                # 간단한 Haar 웨이블릿
                n = len(signal)
                if n % 2 == 1:
                    signal = np.append(signal, signal[-1])
                n = len(signal)
                low = (signal[::2] + signal[1::2]) / np.sqrt(2)
                high = (signal[::2] - signal[1::2]) / np.sqrt(2)
                return np.concatenate([low, high])
            
            wavelet_coeffs = haar_transform(ts)
            features.append(wavelet_coeffs[:20])  # 처음 20개 계수만 사용
    
    return np.array(features)

def convert_to_timeseries(X, sequence_length=52):
    """특성들을 시계열로 변환"""
    n_samples, n_features = X.shape
    # 각 샘플을 시계열로 변환
    X_ts = X.reshape(n_samples, sequence_length, 1)
    return X_ts

def chronos_classification_simple():
    """시계열 분류 (chronos 없이)"""
    
    print("시계열 분류 시작 (chronos 대안)")
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
    
    # 2. 전처리
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
    
    # 4. 다양한 특징 추출 방법 실험
    print("\n4. 특징 추출 방법 실험")
    print("-" * 30)
    
    feature_methods = ['statistical', 'fourier', 'wavelet']
    best_method = None
    best_score = 0
    
    for method in feature_methods:
        print(f"\n{method} 방법으로 특징 추출 중...")
        
        # 특징 추출
        X_features = extract_timeseries_features(X_ts, method)
        X_test_features = extract_timeseries_features(X_test_ts, method)
        
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
    
    # 5. 최종 모델 학습
    print("\n5. 최종 모델 학습")
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
    
    # 6. 최종 예측
    print("\n6. 최종 예측")
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
    
    # 7. 결과 저장
    print("\n7. 결과 저장")
    print("-" * 30)
    
    # 제출용 파일 (ID, target만)
    submission = pd.DataFrame({"ID": test_ids, "target": y_pred})
    submission.to_csv("timeseries_classification_submission.csv", index=False)
    
    # 상세 파일 (ID, target, 모든 클래스 확률)
    detailed = pd.DataFrame({
        "ID": test_ids,
        "target": y_pred,
        **{f"prob_{i}": y_probs[:, i] for i in range(21)}
    })
    detailed.to_csv("timeseries_classification_detailed.csv", index=False)
    
    print(f"✅ 시계열 분류 완료!")
    print(f"제출 파일: timeseries_classification_submission.csv")
    print(f"상세 결과: timeseries_classification_detailed.csv")
    print(f"예측 분포: {np.bincount(y_pred, minlength=21)}")
    
    # 8. 성능 요약
    print("\n8. 성능 요약")
    print("-" * 30)
    print(f"최적 특징 추출 방법: {best_method}")
    print(f"최적 분류기: {best_classifier_name}")
    print(f"검증 F1 점수: {best_classifier_score:.4f}")
    print(f"예측 분포: {np.bincount(y_pred, minlength=21)}")
    
    return y_pred, y_probs, best_X_features, best_X_test_features

if __name__ == "__main__":
    chronos_classification_simple()
