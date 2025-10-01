#!/usr/bin/env python3
"""
AutoGluon을 사용한 시계열 분류 - 간단하고 효과적인 버전
TimeSeries 대신 Tabular + 시계열 특징 추출 사용
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import random
import os
import warnings
warnings.filterwarnings('ignore')

# AutoGluon 설치 확인 및 임포트
try:
    from autogluon.tabular import TabularPredictor
    print("✅ AutoGluon Tabular 임포트 성공")
except ImportError:
    print("❌ AutoGluon이 설치되지 않았습니다.")
    print("설치 명령어: pip install autogluon")
    exit(1)

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)

set_seed(123)

def prepare_data():
    """데이터 준비"""
    print("데이터 준비")
    print("-" * 30)
    
    # 데이터 로드
    train_df = pd.read_csv("./datasests/train.csv")
    test_df = pd.read_csv("./datasests/test.csv")
    
    print(f"훈련 데이터: {train_df.shape}")
    print(f"테스트 데이터: {test_df.shape}")
    
    # 특성과 타겟 분리
    feature_columns = [col for col in train_df.columns if col not in ['ID', 'target']]
    X = train_df[feature_columns]
    y = train_df['target']
    X_test = test_df[feature_columns]
    test_ids = test_df['ID']
    
    print(f"특성 수: {len(feature_columns)}")
    print(f"클래스 수: {len(np.unique(y))}")
    
    return X, y, X_test, test_ids, feature_columns

def extract_timeseries_features(X, feature_columns):
    """시계열 특징 추출"""
    print("시계열 특징 추출")
    print("-" * 30)
    
    # 시계열 통계 특징 추출
    ts_features = []
    feature_names = []
    
    for idx, (_, row) in enumerate(X.iterrows()):
        ts_values = row.values
        
        # 기본 통계 특징
        features = [
            np.mean(ts_values),           # 평균
            np.std(ts_values),            # 표준편차
            np.min(ts_values),            # 최솟값
            np.max(ts_values),            # 최댓값
            np.median(ts_values),         # 중앙값
            np.percentile(ts_values, 25), # 25% 분위수
            np.percentile(ts_values, 75), # 75% 분위수
            np.var(ts_values),            # 분산
            np.ptp(ts_values),            # 범위
            np.mean(np.diff(ts_values)),  # 차분 평균
            np.std(np.diff(ts_values)),   # 차분 표준편차
        ]
        
        # 자기상관 특징
        if len(ts_values) > 1:
            autocorr = np.corrcoef(ts_values[:-1], ts_values[1:])[0, 1]
            features.append(autocorr)
        else:
            features.append(0)
        
        # 추가 시계열 특징
        features.extend([
            np.sum(ts_values > np.mean(ts_values)),  # 평균보다 큰 값의 개수
            np.sum(ts_values < np.mean(ts_values)),  # 평균보다 작은 값의 개수
            len(np.where(np.diff(ts_values) > 0)[0]),  # 증가 구간 개수
            len(np.where(np.diff(ts_values) < 0)[0]),  # 감소 구간 개수
            np.max(np.abs(np.diff(ts_values))),  # 최대 변화량
            np.mean(np.abs(np.diff(ts_values))),  # 평균 변화량
        ])
        
        ts_features.append(features)
    
    # 특징 이름 생성
    feature_names = [
        'ts_mean', 'ts_std', 'ts_min', 'ts_max', 'ts_median',
        'ts_q25', 'ts_q75', 'ts_var', 'ts_range', 'ts_diff_mean',
        'ts_diff_std', 'ts_autocorr', 'ts_above_mean', 'ts_below_mean',
        'ts_increasing', 'ts_decreasing', 'ts_max_change', 'ts_avg_change'
    ]
    
    ts_features = np.array(ts_features)
    print(f"시계열 특징 형태: {ts_features.shape}")
    print(f"추출된 특징: {len(feature_names)}개")
    
    return ts_features, feature_names

def run_autogluon_classification():
    """AutoGluon 분류 실행"""
    print("AutoGluon 분류 시작")
    print("=" * 60)
    
    # 1. 데이터 준비
    X, y, X_test, test_ids, feature_columns = prepare_data()
    
    # 2. 시계열 특징 추출
    ts_features, ts_feature_names = extract_timeseries_features(X, feature_columns)
    ts_features_test, _ = extract_timeseries_features(X_test, feature_columns)
    
    # 3. 원본 특성과 시계열 특징 결합
    print("\n3. 특성 결합")
    print("-" * 30)
    
    # 스케일링
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    # 원본 특성 + 시계열 특징
    combined_features = np.hstack([X_scaled, ts_features])
    combined_features_test = np.hstack([X_test_scaled, ts_features_test])
    
    # DataFrame으로 변환
    all_feature_names = feature_columns + ts_feature_names
    combined_df = pd.DataFrame(combined_features, columns=all_feature_names)
    combined_df_test = pd.DataFrame(combined_features_test, columns=all_feature_names)
    
    combined_df['target'] = y
    
    print(f"결합된 특성 수: {len(all_feature_names)}")
    print(f"원본 특성: {len(feature_columns)}개")
    print(f"시계열 특징: {len(ts_feature_names)}개")
    
    # 4. AutoGluon 모델 학습
    print("\n4. AutoGluon 모델 학습")
    print("-" * 30)
    
    predictor = TabularPredictor(
        label='target',
        problem_type='multiclass',
        eval_metric='f1_macro',
        path='autogluon_timeseries_classification'
    )
    
    print("모델 학습 시작...")
    print("사용할 모델들:")
    print("- LightGBM")
    print("- XGBoost")
    print("- CatBoost")
    print("- Random Forest")
    print("- Extra Trees")
    print("- Neural Network")
    print("- KNN")
    
    try:
        predictor.fit(
            combined_df,
            time_limit=60,  # 60분
            presets='best_quality'  # 최고 품질
        )
        print("✅ 모델 학습 완료")
        
        # 모델 성능 확인
        print("\n모델 성능 확인:")
        leaderboard = predictor.leaderboard(silent=True)
        if len(leaderboard) > 0:
            print(leaderboard[['model', 'score_val']].head(10))
        
    except Exception as e:
        print(f"모델 학습 실패: {e}")
        print("기본 설정으로 재시도...")
        predictor.fit(combined_df, time_limit=30, presets='medium_quality')
    
    # 5. 예측
    print("\n5. 예측")
    print("-" * 30)
    
    y_pred = predictor.predict(combined_df_test)
    y_probs = predictor.predict_proba(combined_df_test)
    
    print(f"예측 완료: {len(y_pred)}개")
    print(f"예측 분포: {np.bincount(y_pred, minlength=21)}")
    
    # 6. 결과 저장
    print("\n6. 결과 저장")
    print("-" * 30)
    
    # 제출 파일
    submission = pd.DataFrame({
        "ID": test_ids,
        "target": y_pred
    })
    submission.to_csv("autogluon_timeseries_classification_submission.csv", index=False)
    
    # 상세 파일
    detailed = pd.DataFrame({
        "ID": test_ids,
        "target": y_pred,
        **{f"prob_{i}": y_probs.iloc[:, i] for i in range(21)}
    })
    detailed.to_csv("autogluon_timeseries_classification_detailed.csv", index=False)
    
    print("✅ 분류 완료!")
    print("제출 파일: autogluon_timeseries_classification_submission.csv")
    print("상세 결과: autogluon_timeseries_classification_detailed.csv")
    
    return y_pred, y_probs

def run_basic_autogluon():
    """기본 AutoGluon 분류 (시계열 특징 없이)"""
    print("\n기본 AutoGluon 분류")
    print("=" * 60)
    
    # 1. 데이터 준비
    X, y, X_test, test_ids, feature_columns = prepare_data()
    
    # 2. 스케일링
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_columns, index=X.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_columns, index=X_test.index)
    
    # 3. 데이터 준비
    train_data = X_scaled.copy()
    train_data['target'] = y
    
    # 4. AutoGluon 모델 학습
    predictor = TabularPredictor(
        label='target',
        problem_type='multiclass',
        eval_metric='f1_macro',
        path='autogluon_basic_classification'
    )
    
    print("기본 모델 학습 시작...")
    predictor.fit(train_data, time_limit=30, presets='best_quality')
    
    # 5. 예측
    y_pred = predictor.predict(X_test_scaled)
    y_probs = predictor.predict_proba(X_test_scaled)
    
    # 6. 결과 저장
    submission = pd.DataFrame({
        "ID": test_ids,
        "target": y_pred
    })
    submission.to_csv("autogluon_basic_classification_submission.csv", index=False)
    
    detailed = pd.DataFrame({
        "ID": test_ids,
        "target": y_pred,
        **{f"prob_{i}": y_probs.iloc[:, i] for i in range(21)}
    })
    detailed.to_csv("autogluon_basic_classification_detailed.csv", index=False)
    
    print("✅ 기본 분류 완료!")
    print("제출 파일: autogluon_basic_classification_submission.csv")
    print("상세 결과: autogluon_basic_classification_detailed.csv")
    
    return y_pred, y_probs

if __name__ == "__main__":
    print("AutoGluon 시계열 분류 옵션:")
    print("1. 시계열 특징 + AutoGluon (권장)")
    print("2. 기본 AutoGluon (시계열 특징 없이)")
    print("3. 둘 다 실행")
    
    choice = input("선택하세요 (1/2/3): ").strip()
    
    if choice == "1":
        run_autogluon_classification()
    elif choice == "2":
        run_basic_autogluon()
    elif choice == "3":
        print("\n=== 시계열 특징 + AutoGluon ===")
        run_autogluon_classification()
        print("\n=== 기본 AutoGluon ===")
        run_basic_autogluon()
    else:
        print("시계열 특징 + AutoGluon을 실행합니다...")
        run_autogluon_classification()
