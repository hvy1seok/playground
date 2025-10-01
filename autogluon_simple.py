#!/usr/bin/env python3
"""
AutoGluon 간단 버전 - 최소한의 설정으로 실행
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import random
import os

# AutoGluon 설치 확인 및 임포트
try:
    from autogluon.tabular import TabularPredictor
    print("✅ AutoGluon 임포트 성공")
except ImportError:
    print("❌ AutoGluon이 설치되지 않았습니다.")
    print("설치 명령어: pip install autogluon")
    exit(1)

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)

set_seed(123)

def autogluon_simple():
    """AutoGluon 간단 버전"""
    
    print("AutoGluon 간단 분류 시작")
    print("=" * 50)
    
    # 1. 데이터 로드
    print("\n1. 데이터 로드")
    print("-" * 30)
    
    train_df = pd.read_csv("./datasests/train.csv")
    test_df = pd.read_csv("./datasests/test.csv")
    
    print(f"훈련 데이터: {train_df.shape}")
    print(f"테스트 데이터: {test_df.shape}")
    
    # 2. 전처리
    feature_columns = [col for col in train_df.columns if col not in ['ID', 'target']]
    X = train_df[feature_columns]
    y = train_df['target']
    X_test = test_df[feature_columns]
    test_ids = test_df['ID']
    
    # 스케일링
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_columns, index=X.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_columns, index=X_test.index)
    
    # 타겟 추가
    train_data = X_scaled.copy()
    train_data['target'] = y
    
    print(f"특성 수: {len(feature_columns)}")
    print(f"클래스 수: {len(np.unique(y))}")
    
    # 3. AutoGluon 설정
    print("\n2. AutoGluon 설정")
    print("-" * 30)
    
    predictor = TabularPredictor(
        label='target',
        problem_type='multiclass',
        eval_metric='f1_macro',
        path='autogluon_simple_models'
    )
    
    # 4. 학습 (간단한 설정)
    print("\n3. 모델 학습")
    print("-" * 30)
    
    print("학습 시작...")
    predictor.fit(
        train_data,
        time_limit=10,  # 10분으로 단축
        presets='medium_quality'  # 중간 품질로 안정성 확보
    )
    
    print("✅ 학습 완료")
    
    # 5. 성능 확인
    print("\n4. 성능 확인")
    print("-" * 30)
    
    try:
        leaderboard = predictor.leaderboard(silent=True)
        if len(leaderboard) > 0:
            print("모델 성능:")
            print(leaderboard[['model', 'score_val']].head(5))
            
            best_score = leaderboard.iloc[0]['score_val']
            print(f"최고 성능: {best_score:.4f}")
        else:
            print("성능 정보를 가져올 수 없습니다.")
            best_score = 0.0
    except Exception as e:
        print(f"성능 확인 중 오류: {e}")
        best_score = 0.0
    
    # 6. 예측
    print("\n5. 예측 수행")
    print("-" * 30)
    
    y_pred = predictor.predict(X_test_scaled)
    y_probs = predictor.predict_proba(X_test_scaled)
    
    print(f"예측 완료: {len(y_pred)}개 샘플")
    print(f"예측 분포: {np.bincount(y_pred, minlength=21)}")
    
    # 7. 결과 저장
    print("\n6. 결과 저장")
    print("-" * 30)
    
    # 제출 파일
    submission = pd.DataFrame({
        "ID": test_ids,
        "target": y_pred
    })
    submission.to_csv("autogluon_simple_submission.csv", index=False)
    
    # 상세 파일
    detailed = pd.DataFrame({
        "ID": test_ids,
        "target": y_pred,
        **{f"prob_{i}": y_probs.iloc[:, i] for i in range(21)}
    })
    detailed.to_csv("autogluon_simple_detailed.csv", index=False)
    
    print("✅ 완료!")
    print("제출 파일: autogluon_simple_submission.csv")
    print("상세 결과: autogluon_simple_detailed.csv")
    
    return y_pred, y_probs

if __name__ == "__main__":
    autogluon_simple()
