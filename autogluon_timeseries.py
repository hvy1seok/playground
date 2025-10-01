#!/usr/bin/env python3
"""
AutoGluon TimeSeries를 사용한 시계열 분류
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import random
import os

# AutoGluon TimeSeries 설치 확인 및 임포트
try:
    from autogluon.timeseries import TimeSeriesPredictor
    from autogluon.tabular import TabularPredictor
    print("✅ AutoGluon TimeSeries 임포트 성공")
except ImportError:
    print("❌ AutoGluon TimeSeries가 설치되지 않았습니다.")
    print("설치 명령어: pip install autogluon[timeseries]")
    exit(1)

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)

set_seed(123)

def prepare_timeseries_data():
    """시계열 데이터 준비"""
    print("시계열 데이터 준비")
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
    
    # 스케일링
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_columns, index=X.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_columns, index=X_test.index)
    
    print(f"특성 수: {len(feature_columns)}")
    print(f"클래스 수: {len(np.unique(y))}")
    
    return X_scaled, y, X_test_scaled, test_ids, feature_columns

def convert_to_timeseries_format(X, y, feature_columns):
    """AutoGluon TimeSeries 형식으로 변환"""
    print("시계열 형식으로 변환")
    print("-" * 30)
    
    # 각 샘플을 시계열로 변환
    timeseries_data = []
    
    for idx, (_, row) in enumerate(X.iterrows()):
        # 시계열 데이터 생성 (52개 특성을 시간 순서로)
        ts_data = {
            'item_id': f'series_{idx}',  # 시계열 ID
            'timestamp': list(range(52)),  # 시간 인덱스
            'target': row.values.tolist(),  # 시계열 값들
            'class': y.iloc[idx]  # 클래스 라벨
        }
        timeseries_data.append(ts_data)
    
    # DataFrame으로 변환
    ts_df = pd.DataFrame(timeseries_data)
    
    print(f"시계열 데이터 형태: {ts_df.shape}")
    print(f"시계열 길이: 52")
    print(f"시계열 개수: {len(ts_df)}")
    
    return ts_df

def autogluon_timeseries_classification():
    """AutoGluon TimeSeries를 사용한 분류"""
    
    print("AutoGluon TimeSeries 분류 시작")
    print("=" * 60)
    
    # 1. 데이터 준비
    X, y, X_test, test_ids, feature_columns = prepare_timeseries_data()
    
    # 2. 시계열 형식으로 변환
    train_ts = convert_to_timeseries_format(X, y, feature_columns)
    
    # 3. TimeSeries Predictor 설정
    print("\n3. TimeSeries Predictor 설정")
    print("-" * 30)
    
    # 시계열 예측기 생성
    predictor = TimeSeriesPredictor(
        target="target",
        prediction_length=1,  # 1스텝 예측
        path="autogluon_timeseries_models"
    )
    
    print("✅ TimeSeries Predictor 생성 완료")
    
    # 4. 시계열 모델 학습
    print("\n4. 시계열 모델 학습")
    print("-" * 30)
    
    print("시계열 모델 학습 시작...")
    print("사용 가능한 모델들:")
    print("- ARIMA")
    print("- ETS (Exponential Smoothing)")
    print("- Prophet")
    print("- DeepAR")
    print("- Transformer")
    
    try:
        predictor.fit(
            train_ts,
            time_limit=20,  # 20분
            presets="medium_quality"
        )
        print("✅ 시계열 모델 학습 완료")
    except Exception as e:
        print(f"시계열 학습 실패: {e}")
        print("Tabular 모델로 대체합니다...")
        return autogluon_tabular_fallback(X, y, X_test, test_ids)
    
    # 5. 테스트 데이터 예측
    print("\n5. 테스트 데이터 예측")
    print("-" * 30)
    
    # 테스트 데이터를 시계열 형식으로 변환
    test_ts = convert_to_timeseries_format(X_test, pd.Series([0] * len(X_test)), feature_columns)
    
    try:
        # 시계열 예측
        predictions = predictor.predict(test_ts)
        print(f"시계열 예측 완료: {len(predictions)}개")
        
        # 예측 결과를 클래스로 변환 (간단한 방법)
        y_pred = []
        for pred in predictions['mean']:
            # 예측값을 클래스로 변환 (0-20 범위로 정규화)
            class_pred = int(np.clip(pred * 20, 0, 20))
            y_pred.append(class_pred)
        
        y_pred = np.array(y_pred)
        
    except Exception as e:
        print(f"시계열 예측 실패: {e}")
        print("Tabular 모델로 대체합니다...")
        return autogluon_tabular_fallback(X, y, X_test, test_ids)
    
    # 6. 결과 저장
    print("\n6. 결과 저장")
    print("-" * 30)
    
    # 확률 생성 (간단한 방법)
    y_probs = np.zeros((len(y_pred), 21))
    for i, pred in enumerate(y_pred):
        y_probs[i, pred] = 1.0
    
    # 제출 파일
    submission = pd.DataFrame({
        "ID": test_ids,
        "target": y_pred
    })
    submission.to_csv("autogluon_timeseries_submission.csv", index=False)
    
    # 상세 파일
    detailed = pd.DataFrame({
        "ID": test_ids,
        "target": y_pred,
        **{f"prob_{i}": y_probs[:, i] for i in range(21)}
    })
    detailed.to_csv("autogluon_timeseries_detailed.csv", index=False)
    
    print("✅ TimeSeries 분류 완료!")
    print("제출 파일: autogluon_timeseries_submission.csv")
    print("상세 결과: autogluon_timeseries_detailed.csv")
    print(f"예측 분포: {np.bincount(y_pred, minlength=21)}")
    
    return y_pred, y_probs

def autogluon_tabular_fallback(X, y, X_test, test_ids):
    """Tabular 모델로 대체"""
    print("\nTabular 모델로 대체 실행")
    print("-" * 30)
    
    # Tabular 데이터 준비
    train_data = X.copy()
    train_data['target'] = y
    
    # Tabular Predictor
    predictor = TabularPredictor(
        label='target',
        problem_type='multiclass',
        eval_metric='f1_macro',
        path='autogluon_tabular_fallback'
    )
    
    # 학습
    predictor.fit(train_data, time_limit=10, presets='medium_quality')
    
    # 예측
    y_pred = predictor.predict(X_test)
    y_probs = predictor.predict_proba(X_test)
    
    # 결과 저장
    submission = pd.DataFrame({
        "ID": test_ids,
        "target": y_pred
    })
    submission.to_csv("autogluon_tabular_fallback_submission.csv", index=False)
    
    detailed = pd.DataFrame({
        "ID": test_ids,
        "target": y_pred,
        **{f"prob_{i}": y_probs.iloc[:, i] for i in range(21)}
    })
    detailed.to_csv("autogluon_tabular_fallback_detailed.csv", index=False)
    
    print("✅ Tabular 대체 모델 완료!")
    return y_pred, y_probs

def autogluon_hybrid_approach():
    """하이브리드 접근법: TimeSeries + Tabular"""
    print("\n하이브리드 접근법: TimeSeries + Tabular")
    print("=" * 60)
    
    # 1. 데이터 준비
    X, y, X_test, test_ids, feature_columns = prepare_timeseries_data()
    
    # 2. TimeSeries 특징 추출
    print("\n2. TimeSeries 특징 추출")
    print("-" * 30)
    
    # 시계열 통계 특징 추출
    ts_features = []
    for idx, (_, row) in enumerate(X.iterrows()):
        ts_values = row.values
        
        # 시계열 통계 특징
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
            np.corrcoef(ts_values[:-1], ts_values[1:])[0, 1] if len(ts_values) > 1 else 0,  # 자기상관
        ]
        ts_features.append(features)
    
    ts_features = np.array(ts_features)
    ts_features_test = []
    
    for idx, (_, row) in enumerate(X_test.iterrows()):
        ts_values = row.values
        features = [
            np.mean(ts_values), np.std(ts_values), np.min(ts_values), np.max(ts_values),
            np.median(ts_values), np.percentile(ts_values, 25), np.percentile(ts_values, 75),
            np.var(ts_values), np.ptp(ts_values), np.mean(np.diff(ts_values)),
            np.std(np.diff(ts_values)), np.corrcoef(ts_values[:-1], ts_values[1:])[0, 1] if len(ts_values) > 1 else 0,
        ]
        ts_features_test.append(features)
    
    ts_features_test = np.array(ts_features_test)
    
    print(f"시계열 특징 형태: {ts_features.shape}")
    
    # 3. 원본 특성과 시계열 특징 결합
    print("\n3. 특성 결합")
    print("-" * 30)
    
    # 원본 특성 + 시계열 특징
    combined_features = np.hstack([X.values, ts_features])
    combined_features_test = np.hstack([X_test.values, ts_features_test])
    
    # DataFrame으로 변환
    feature_names = feature_columns + [f'ts_feature_{i}' for i in range(ts_features.shape[1])]
    combined_df = pd.DataFrame(combined_features, columns=feature_names)
    combined_df_test = pd.DataFrame(combined_features_test, columns=feature_names)
    
    combined_df['target'] = y
    
    print(f"결합된 특성 수: {len(feature_names)}")
    
    # 4. Tabular 모델로 학습
    print("\n4. 하이브리드 모델 학습")
    print("-" * 30)
    
    predictor = TabularPredictor(
        label='target',
        problem_type='multiclass',
        eval_metric='f1_macro',
        path='autogluon_hybrid_models'
    )
    
    predictor.fit(combined_df, time_limit=15, presets='medium_quality')
    
    # 5. 예측
    y_pred = predictor.predict(combined_df_test)
    y_probs = predictor.predict_proba(combined_df_test)
    
    # 6. 결과 저장
    submission = pd.DataFrame({
        "ID": test_ids,
        "target": y_pred
    })
    submission.to_csv("autogluon_hybrid_submission.csv", index=False)
    
    detailed = pd.DataFrame({
        "ID": test_ids,
        "target": y_pred,
        **{f"prob_{i}": y_probs.iloc[:, i] for i in range(21)}
    })
    detailed.to_csv("autogluon_hybrid_detailed.csv", index=False)
    
    print("✅ 하이브리드 모델 완료!")
    print("제출 파일: autogluon_hybrid_submission.csv")
    print("상세 결과: autogluon_hybrid_detailed.csv")
    print(f"예측 분포: {np.bincount(y_pred, minlength=21)}")
    
    return y_pred, y_probs

if __name__ == "__main__":
    print("AutoGluon TimeSeries 분류 옵션:")
    print("1. 순수 TimeSeries 모델")
    print("2. 하이브리드 접근법 (권장)")
    print("3. 둘 다 실행")
    
    choice = input("선택하세요 (1/2/3): ").strip()
    
    if choice == "1":
        autogluon_timeseries_classification()
    elif choice == "2":
        autogluon_hybrid_approach()
    elif choice == "3":
        print("\n=== 순수 TimeSeries 모델 ===")
        autogluon_timeseries_classification()
        print("\n=== 하이브리드 모델 ===")
        autogluon_hybrid_approach()
    else:
        print("하이브리드 접근법을 실행합니다...")
        autogluon_hybrid_approach()
