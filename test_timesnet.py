#!/usr/bin/env python3
"""
TimesNet 분류 모델 테스트 스크립트
간단한 테스트를 위해 작은 데이터셋으로 모델을 테스트합니다.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Time-Series-Library 경로 추가
sys.path.append('./Time-Series-Library')

def test_data_loading():
    """데이터 로딩 테스트"""
    print("데이터 로딩 테스트...")
    
    try:
        train_df = pd.read_csv("datasests/train.csv")
        test_df = pd.read_csv("datasests/test.csv")
        
        print(f"✓ Train 데이터: {train_df.shape}")
        print(f"✓ Test 데이터: {test_df.shape}")
        print(f"✓ 클래스 개수: {len(np.unique(train_df['target']))}")
        
        return True
    except Exception as e:
        print(f"✗ 데이터 로딩 실패: {e}")
        return False

def test_timesnet_import():
    """TimesNet 모델 import 테스트"""
    print("\nTimesNet 모델 import 테스트...")
    
    try:
        from models.TimesNet import Model as TimesNet
        from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
        print("✓ TimesNet 모델 import 성공")
        return True
    except Exception as e:
        print(f"✗ TimesNet 모델 import 실패: {e}")
        return False

def test_data_preprocessing():
    """데이터 전처리 테스트"""
    print("\n데이터 전처리 테스트...")
    
    try:
        # 작은 샘플 데이터로 테스트
        train_df = pd.read_csv("datasests/train.csv")
        sample_df = train_df.sample(n=1000, random_state=42)  # 1000개 샘플만 사용
        
        X = sample_df.drop(columns=["ID", "target"]).values
        y = sample_df["target"].values
        
        # 정규화
        scaler = RobustScaler()
        X = scaler.fit_transform(X)
        
        # 학습/검증 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=123
        )
        
        print(f"✓ 전처리 완료 - Train: {X_train.shape}, Val: {X_val.shape}")
        return True
    except Exception as e:
        print(f"✗ 데이터 전처리 실패: {e}")
        return False

def test_timesnet_model():
    """TimesNet 모델 생성 테스트"""
    print("\nTimesNet 모델 생성 테스트...")
    
    try:
        from models.TimesNet import Model as TimesNet
        
        # 간단한 설정
        class SimpleConfig:
            def __init__(self):
                self.task_name = 'classification'
                self.seq_len = 52
                self.label_len = 0
                self.pred_len = 0
                self.enc_in = 1
                self.num_class = 21
                self.e_layers = 1  # 테스트용으로 1개 레이어만
                self.d_model = 16  # 작은 모델
                self.d_ff = 32
                self.top_k = 2
                self.num_kernels = 2
                self.dropout = 0.1
                self.embed = 'timeF'
                self.freq = 'h'
        
        config = SimpleConfig()
        model = TimesNet(config)
        
        # 테스트 입력
        batch_size = 4
        test_input = torch.randn(batch_size, 52, 1)
        test_mask = torch.ones(batch_size, 52)
        
        # Forward pass 테스트
        with torch.no_grad():
            output = model.classification(test_input, test_mask)
        
        print(f"✓ 모델 생성 성공 - 입력: {test_input.shape}, 출력: {output.shape}")
        print(f"✓ 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"✗ TimesNet 모델 생성 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("TimesNet 분류 모델 테스트 시작")
    print("=" * 50)
    
    tests = [
        test_data_loading,
        test_timesnet_import,
        test_data_preprocessing,
        test_timesnet_model
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"테스트 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("✓ 모든 테스트 통과! TimesNet 모델을 사용할 수 있습니다.")
        print("\n실행 방법:")
        print("python timesnet_classification.py")
    else:
        print("✗ 일부 테스트 실패. 문제를 해결한 후 다시 시도하세요.")

if __name__ == "__main__":
    main()
