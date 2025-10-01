#!/usr/bin/env python3
"""
AutoGluon TimeSeries 고급 모델들을 사용한 시계열 분류
- Chronos, PatchTST, DLinear, SimpleFeedForward, TemporalFusionTransformer
- TiDE, WaveNet, DirectTabular, PerStepTabular, RecursiveTabular
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

def run_advanced_timeseries_models():
    """고급 TimeSeries 모델들 실행"""
    print("AutoGluon TimeSeries 고급 모델들 실행")
    print("=" * 60)
    
    # 1. 데이터 준비
    X, y, X_test, test_ids, feature_columns = prepare_timeseries_data()
    
    # 2. 시계열 형식으로 변환
    train_ts = convert_to_timeseries_format(X, y, feature_columns)
    
    # 3. 고급 모델들 설정
    print("\n3. 고급 TimeSeries 모델들 설정")
    print("-" * 30)
    
    # 사용할 모델들
    models = {
        'ChronosModel': {
            'model': 'ChronosModel',
            'description': 'Chronos pretrained models for zero-shot forecasting'
        },
        'PatchTSTModel': {
            'model': 'PatchTSTModel',
            'description': 'Transformer-based forecaster with patching'
        },
        'DLinearModel': {
            'model': 'DLinearModel',
            'description': 'Simple feedforward with trend subtraction'
        },
        'SimpleFeedForwardModel': {
            'model': 'SimpleFeedForwardModel',
            'description': 'Simple feedforward neural network'
        },
        'TemporalFusionTransformerModel': {
            'model': 'TemporalFusionTransformerModel',
            'description': 'LSTM + Transformer for quantile prediction'
        },
        'TiDEModel': {
            'model': 'TiDEModel',
            'description': 'Time series dense encoder model'
        },
        'WaveNetModel': {
            'model': 'WaveNetModel',
            'description': 'WaveNet with quantized targets'
        },
        'DirectTabularModel': {
            'model': 'DirectTabularModel',
            'description': 'Direct tabular regression for all future values'
        },
        'PerStepTabularModel': {
            'model': 'PerStepTabularModel',
            'description': 'Separate tabular model for each time step'
        },
        'RecursiveTabularModel': {
            'model': 'RecursiveTabularModel',
            'description': 'Recursive tabular prediction one by one'
        }
    }
    
    print("사용할 모델들:")
    for name, info in models.items():
        print(f"- {name}: {info['description']}")
    
    # 4. TimeSeries Predictor 설정
    print("\n4. TimeSeries Predictor 설정")
    print("-" * 30)
    
    # 시계열 예측기 생성
    predictor = TimeSeriesPredictor(
        target="target",
        prediction_length=1,  # 1스텝 예측
        path="autogluon_advanced_timeseries_models"
    )
    
    print("✅ TimeSeries Predictor 생성 완료")
    
    # 5. 고급 모델들 학습
    print("\n5. 고급 모델들 학습")
    print("-" * 30)
    
    print("고급 모델들 학습 시작...")
    print("예상 학습 시간: 30-60분")
    
    try:
        # 모든 모델을 포함한 학습
        predictor.fit(
            train_ts,
            time_limit=60,  # 60분
            presets="best_quality",  # 최고 품질
            hyperparameters={
                'ChronosModel': {
                    'model_path': 'amazon/chronos-t5-tiny',  # 작은 모델로 시작
                    'max_length': 52
                },
                'PatchTSTModel': {
                    'patch_len': 8,
                    'stride': 4,
                    'd_model': 64,
                    'n_heads': 4,
                    'e_layers': 2,
                    'd_ff': 128
                },
                'DLinearModel': {
                    'individual': True,
                    'moving_avg': 25
                },
                'SimpleFeedForwardModel': {
                    'hidden_dimensions': [64, 32],
                    'context_length': 52
                },
                'TemporalFusionTransformerModel': {
                    'hidden_size': 64,
                    'lstm_layers': 2,
                    'attention_head_size': 4,
                    'dropout': 0.1
                },
                'TiDEModel': {
                    'hidden_dim': 64,
                    'num_layers': 2,
                    'temporal_width': 4
                },
                'WaveNetModel': {
                    'n_filters': 32,
                    'n_layers': 4,
                    'kernel_size': 3
                },
                'DirectTabularModel': {
                    'tabular_hyperparameters': {
                        'GBM': {'num_boost_round': 100},
                        'RF': {'n_estimators': 100}
                    }
                },
                'PerStepTabularModel': {
                    'tabular_hyperparameters': {
                        'GBM': {'num_boost_round': 50},
                        'RF': {'n_estimators': 50}
                    }
                },
                'RecursiveTabularModel': {
                    'tabular_hyperparameters': {
                        'GBM': {'num_boost_round': 50},
                        'RF': {'n_estimators': 50}
                    }
                }
            }
        )
        print("✅ 고급 모델들 학습 완료")
        
        # 모델 성능 확인
        print("\n모델 성능 확인:")
        leaderboard = predictor.leaderboard(silent=True)
        if len(leaderboard) > 0:
            print(leaderboard[['model', 'score_val']].head(10))
        
    except Exception as e:
        print(f"고급 모델 학습 실패: {e}")
        print("기본 모델로 대체합니다...")
        return run_basic_timeseries_models(X, y, X_test, test_ids)
    
    # 6. 테스트 데이터 예측
    print("\n6. 테스트 데이터 예측")
    print("-" * 30)
    
    # 테스트 데이터를 시계열 형식으로 변환
    test_ts = convert_to_timeseries_format(X_test, pd.Series([0] * len(X_test)), feature_columns)
    
    try:
        # 시계열 예측
        predictions = predictor.predict(test_ts)
        print(f"시계열 예측 완료: {len(predictions)}개")
        
        # 예측 결과를 클래스로 변환 (고급 방법)
        y_pred = convert_predictions_to_classes(predictions, y)
        
    except Exception as e:
        print(f"시계열 예측 실패: {e}")
        print("Tabular 모델로 대체합니다...")
        return run_tabular_fallback(X, y, X_test, test_ids)
    
    # 7. 결과 저장
    print("\n7. 결과 저장")
    print("-" * 30)
    
    # 확률 생성 (고급 방법)
    y_probs = generate_advanced_probabilities(predictions, y_pred)
    
    # 제출 파일
    submission = pd.DataFrame({
        "ID": test_ids,
        "target": y_pred
    })
    submission.to_csv("autogluon_advanced_timeseries_submission.csv", index=False)
    
    # 상세 파일
    detailed = pd.DataFrame({
        "ID": test_ids,
        "target": y_pred,
        **{f"prob_{i}": y_probs[:, i] for i in range(21)}
    })
    detailed.to_csv("autogluon_advanced_timeseries_detailed.csv", index=False)
    
    print("✅ 고급 TimeSeries 분류 완료!")
    print("제출 파일: autogluon_advanced_timeseries_submission.csv")
    print("상세 결과: autogluon_advanced_timeseries_detailed.csv")
    print(f"예측 분포: {np.bincount(y_pred, minlength=21)}")
    
    return y_pred, y_probs

def convert_predictions_to_classes(predictions, y_train):
    """예측 결과를 클래스로 변환 (고급 방법)"""
    print("예측 결과를 클래스로 변환")
    
    # 훈련 데이터의 클래스 분포 확인
    class_counts = np.bincount(y_train)
    class_probs = class_counts / len(y_train)
    
    y_pred = []
    for pred in predictions['mean']:
        # 예측값을 클래스 확률로 변환
        pred_value = float(pred)
        
        # 클래스별 확률 계산 (고급 방법)
        class_scores = []
        for class_idx in range(21):
            # 예측값과 클래스 인덱스의 거리 기반 점수
            distance = abs(pred_value - class_idx)
            score = np.exp(-distance / 5.0)  # 거리 기반 점수
            
            # 클래스 빈도 가중치 적용
            weighted_score = score * class_probs[class_idx]
            class_scores.append(weighted_score)
        
        # 가장 높은 점수의 클래스 선택
        predicted_class = np.argmax(class_scores)
        y_pred.append(predicted_class)
    
    return np.array(y_pred)

def generate_advanced_probabilities(predictions, y_pred):
    """고급 확률 생성"""
    print("고급 확률 생성")
    
    y_probs = np.zeros((len(y_pred), 21))
    
    for i, pred in enumerate(predictions['mean']):
        pred_value = float(pred)
        
        # 각 클래스에 대한 확률 계산
        for class_idx in range(21):
            # 예측값과 클래스 인덱스의 거리 기반 확률
            distance = abs(pred_value - class_idx)
            prob = np.exp(-distance / 3.0)  # 거리 기반 확률
            y_probs[i, class_idx] = prob
        
        # 확률 정규화
        y_probs[i] = y_probs[i] / np.sum(y_probs[i])
    
    return y_probs

def run_basic_timeseries_models(X, y, X_test, test_ids):
    """기본 TimeSeries 모델들 실행"""
    print("\n기본 TimeSeries 모델들 실행")
    print("-" * 30)
    
    # 시계열 형식으로 변환
    train_ts = convert_to_timeseries_format(X, y, X.columns.tolist())
    
    # 기본 모델들로 학습
    predictor = TimeSeriesPredictor(
        target="target",
        prediction_length=1,
        path="autogluon_basic_timeseries_models"
    )
    
    try:
        predictor.fit(train_ts, time_limit=30, presets="medium_quality")
        
        # 테스트 예측
        test_ts = convert_to_timeseries_format(X_test, pd.Series([0] * len(X_test)), X.columns.tolist())
        predictions = predictor.predict(test_ts)
        
        y_pred = convert_predictions_to_classes(predictions, y)
        y_probs = generate_advanced_probabilities(predictions, y_pred)
        
        # 결과 저장
        submission = pd.DataFrame({
            "ID": test_ids,
            "target": y_pred
        })
        submission.to_csv("autogluon_basic_timeseries_submission.csv", index=False)
        
        detailed = pd.DataFrame({
            "ID": test_ids,
            "target": y_pred,
            **{f"prob_{i}": y_probs[:, i] for i in range(21)}
        })
        detailed.to_csv("autogluon_basic_timeseries_detailed.csv", index=False)
        
        print("✅ 기본 TimeSeries 모델 완료!")
        return y_pred, y_probs
        
    except Exception as e:
        print(f"기본 모델도 실패: {e}")
        return run_tabular_fallback(X, y, X_test, test_ids)

def run_tabular_fallback(X, y, X_test, test_ids):
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
    predictor.fit(train_data, time_limit=20, presets='best_quality')
    
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

def run_hybrid_approach():
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
    
    predictor.fit(combined_df, time_limit=30, presets='best_quality')
    
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
    print("AutoGluon TimeSeries 고급 분류 옵션:")
    print("1. 고급 TimeSeries 모델들 (Chronos, PatchTST, DLinear, etc.)")
    print("2. 하이브리드 접근법 (TimeSeries + Tabular)")
    print("3. 둘 다 실행")
    
    choice = input("선택하세요 (1/2/3): ").strip()
    
    if choice == "1":
        run_advanced_timeseries_models()
    elif choice == "2":
        run_hybrid_approach()
    elif choice == "3":
        print("\n=== 고급 TimeSeries 모델들 ===")
        run_advanced_timeseries_models()
        print("\n=== 하이브리드 모델 ===")
        run_hybrid_approach()
    else:
        print("고급 TimeSeries 모델들을 실행합니다...")
        run_advanced_timeseries_models()
