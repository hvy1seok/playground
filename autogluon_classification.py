#!/usr/bin/env python3
"""
AutoGluon을 사용한 자동 머신러닝 분류
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import random
import os

# AutoGluon 설치 확인 및 임포트
try:
    from autogluon.tabular import TabularPredictor
    from autogluon.tabular.models import TabularNeuralNetTorchModel
    print("✅ AutoGluon 임포트 성공")
except ImportError:
    print("❌ AutoGluon이 설치되지 않았습니다.")
    print("설치 명령어: pip install autogluon")
    exit(1)

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)

set_seed(123)

def load_and_preprocess_data():
    """데이터 로드 및 전처리"""
    print("데이터 로드 및 전처리")
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
    print(f"클래스 분포: {np.bincount(y, minlength=21)}")
    
    # 스케일링 (AutoGluon은 자동으로 처리하지만 일관성을 위해)
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_columns, index=X.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_columns, index=X_test.index)
    
    # 타겟 추가
    train_data = X_scaled.copy()
    train_data['target'] = y
    
    return train_data, X_test_scaled, test_ids, feature_columns

def autogluon_classification():
    """AutoGluon을 사용한 분류"""
    
    print("AutoGluon 자동 머신러닝 분류 시작")
    print("=" * 60)
    
    # 1. 데이터 준비
    train_data, X_test, test_ids, feature_columns = load_and_preprocess_data()
    
    # 2. AutoGluon 설정
    print("\n2. AutoGluon 설정")
    print("-" * 30)
    
    # 예측기 생성
    predictor = TabularPredictor(
        label='target',
        problem_type='multiclass',
        eval_metric='f1_macro',  # Macro F1 점수 사용
        path='autogluon_models'  # 모델 저장 경로
    )
    
    print("✅ AutoGluon 예측기 생성 완료")
    
    # 3. 모델 학습
    print("\n3. 모델 학습")
    print("-" * 30)
    
    # 시간 제한 설정 (분 단위)
    time_limit = 30  # 30분
    
    print(f"학습 시작 (시간 제한: {time_limit}분)")
    print("AutoGluon이 자동으로 최적의 모델을 찾습니다...")
    
    # 학습 실행
    try:
        predictor.fit(
            train_data,
            time_limit=time_limit,
            presets='best_quality',  # 최고 품질 프리셋
            verbosity=2  # 상세 로그 출력
        )
    except Exception as e:
        print(f"best_quality 프리셋 실패, medium_quality로 재시도: {e}")
        predictor.fit(
            train_data,
            time_limit=time_limit,
            presets='medium_quality',  # 중간 품질 프리셋
            verbosity=2
        )
    
    print("✅ 모델 학습 완료")
    
    # 4. 모델 성능 평가
    print("\n4. 모델 성능 평가")
    print("-" * 30)
    
    # 리더보드 출력
    print("모델 성능 순위:")
    leaderboard = predictor.leaderboard(silent=True)
    print(leaderboard[['model', 'score_val', 'fit_time', 'pred_time_val']].head(10))
    
    # 최고 성능 모델 정보
    best_model = leaderboard.iloc[0]
    print(f"\n최고 성능 모델: {best_model['model']}")
    print(f"검증 F1 점수: {best_model['score_val']:.4f}")
    
    # 5. 테스트 데이터 예측
    print("\n5. 테스트 데이터 예측")
    print("-" * 30)
    
    # 예측 수행
    y_pred = predictor.predict(X_test)
    y_probs = predictor.predict_proba(X_test)
    
    print(f"예측 완료: {len(y_pred)}개 샘플")
    print(f"예측 분포: {np.bincount(y_pred, minlength=21)}")
    
    # 6. 결과 저장
    print("\n6. 결과 저장")
    print("-" * 30)
    
    # 제출용 파일 (ID, target만)
    submission = pd.DataFrame({
        "ID": test_ids,
        "target": y_pred
    })
    submission.to_csv("autogluon_classification_submission.csv", index=False)
    
    # 상세 파일 (ID, target, 모든 클래스 확률)
    detailed = pd.DataFrame({
        "ID": test_ids,
        "target": y_pred,
        **{f"prob_{i}": y_probs.iloc[:, i] for i in range(21)}
    })
    detailed.to_csv("autogluon_classification_detailed.csv", index=False)
    
    print(f"✅ AutoGluon 분류 완료!")
    print(f"제출 파일: autogluon_classification_submission.csv")
    print(f"상세 결과: autogluon_classification_detailed.csv")
    
    # 7. 모델 정보 저장
    print("\n7. 모델 정보 저장")
    print("-" * 30)
    
    # 사용된 모델들 정보
    model_info = {
        'best_model': best_model['model'],
        'best_score': best_model['score_val'],
        'total_models': len(leaderboard),
        'feature_columns': feature_columns,
        'prediction_distribution': np.bincount(y_pred, minlength=21).tolist()
    }
    
    # 모델 정보를 텍스트 파일로 저장
    with open("autogluon_model_info.txt", "w", encoding="utf-8") as f:
        f.write("AutoGluon 모델 정보\n")
        f.write("=" * 50 + "\n")
        f.write(f"최고 성능 모델: {model_info['best_model']}\n")
        f.write(f"검증 F1 점수: {model_info['best_score']:.4f}\n")
        f.write(f"총 모델 수: {model_info['total_models']}\n")
        f.write(f"특성 수: {len(feature_columns)}\n")
        f.write(f"예측 분포: {model_info['prediction_distribution']}\n")
        f.write("\n모델 성능 순위:\n")
        f.write(leaderboard[['model', 'score_val']].to_string(index=False))
    
    print(f"모델 정보: autogluon_model_info.txt")
    
    return y_pred, y_probs, predictor

def autogluon_cv_evaluation():
    """AutoGluon CV 평가 (선택사항)"""
    print("\n8. Cross Validation 평가 (선택사항)")
    print("-" * 30)
    
    try:
        # 데이터 준비
        train_data, _, _, _ = load_and_preprocess_data()
        
        # CV 평가
        predictor_cv = TabularPredictor(
            label='target',
            problem_type='multiclass',
            eval_metric='f1_macro',
            path='autogluon_cv_models'
        )
        
        print("Cross Validation 실행 중...")
        predictor_cv.fit(
            train_data,
            time_limit=15,  # CV는 시간을 더 줄여서
            presets='medium_quality'  # 안정적인 프리셋 사용
        )
        
        # CV 결과 - 리더보드에서 직접 확인
        cv_leaderboard = predictor_cv.leaderboard(silent=True)
        if len(cv_leaderboard) > 0:
            best_cv_score = cv_leaderboard.iloc[0]['score_val']
            print(f"CV 최고 F1: {best_cv_score:.4f}")
            
            # 여러 모델의 평균 성능 계산
            if len(cv_leaderboard) > 1:
                avg_score = cv_leaderboard['score_val'].mean()
                std_score = cv_leaderboard['score_val'].std()
                print(f"CV 평균 F1: {avg_score:.4f}")
                print(f"CV 표준편차: {std_score:.4f}")
            
            cv_results = {
                'best_score': best_cv_score,
                'avg_score': avg_score if len(cv_leaderboard) > 1 else best_cv_score,
                'std_score': std_score if len(cv_leaderboard) > 1 else 0.0
            }
        else:
            print("CV 결과를 가져올 수 없습니다.")
            cv_results = None
        
        return cv_results
        
    except Exception as e:
        print(f"CV 평가 중 오류: {e}")
        return None

if __name__ == "__main__":
    # AutoGluon 분류 실행
    y_pred, y_probs, predictor = autogluon_classification()
    
    # CV 평가 (선택사항)
    cv_results = autogluon_cv_evaluation()
    
    print("\n" + "=" * 60)
    print("AutoGluon 분류 완료!")
    print("생성된 파일들:")
    print("  - autogluon_classification_submission.csv (제출용)")
    print("  - autogluon_classification_detailed.csv (앙상블용)")
    print("  - autogluon_model_info.txt (모델 정보)")
    print("  - autogluon_models/ (학습된 모델들)")
    print("=" * 60)
