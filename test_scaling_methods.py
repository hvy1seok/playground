#!/usr/bin/env python3
"""
TimesNet에서 다양한 정규화 방법 테스트
FFT에 가장 적합한 정규화 방법을 찾기 위한 실험
"""

import sys
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import time

# Time-Series-Library 경로 추가
sys.path.append('./Time-Series-Library')

from models.TimesNet import Model as TimesNet
from utils.tools import EarlyStopping

def create_simple_config():
    """간단한 테스트용 설정"""
    class SimpleConfig:
        def __init__(self):
            self.task_name = 'classification'
            self.seq_len = 52
            self.label_len = 0
            self.pred_len = 0
            self.enc_in = 1
            self.num_class = 21
            self.e_layers = 1  # 테스트용으로 1개 레이어만
            self.d_model = 32  # 작은 모델
            self.d_ff = 64
            self.top_k = 3
            self.num_kernels = 4
            self.learning_rate = 0.001
            self.batch_size = 64
            self.train_epochs = 5  # 테스트용으로 5 에포크만
            self.patience = 3
            self.dropout = 0.1
            self.embed = 'timeF'
            self.freq = 'h'
            self.use_gpu = torch.cuda.is_available()
            self.device = torch.device('cuda' if self.use_gpu else 'cpu')
            self.lradj = 'type1'
    return SimpleConfig()

def test_scaling_method(scaling_method, X_train, X_val, y_train, y_val, device):
    """특정 정규화 방법으로 모델 테스트"""
    print(f"\n=== {scaling_method.upper()} 정규화 테스트 ===")
    
    # 정규화 적용
    if scaling_method == 'standard':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
    elif scaling_method == 'robust':
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
    elif scaling_method == 'sequence':
        # 시계열별 정규화
        def normalize_sequence(seq):
            seq_mean = np.mean(seq, axis=1, keepdims=True)
            seq_std = np.std(seq, axis=1, keepdims=True)
            return (seq - seq_mean) / (seq_std + 1e-8)
        X_train_scaled = normalize_sequence(X_train)
        X_val_scaled = normalize_sequence(X_val)
    elif scaling_method == 'none':
        # 정규화 없음
        X_train_scaled = X_train
        X_val_scaled = X_val
    
    # 시계열 변환
    X_train_ts = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_val_ts = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)
    
    # FFT 분석
    print("FFT 분석:")
    sample_fft = np.fft.rfft(X_train_ts[0, :, 0])
    freq_amplitudes = np.abs(sample_fft)
    print(f"  주파수 성분 수: {len(freq_amplitudes)}")
    print(f"  최대 진폭: {np.max(freq_amplitudes):.4f}")
    print(f"  평균 진폭: {np.mean(freq_amplitudes):.4f}")
    print(f"  진폭 표준편차: {np.std(freq_amplitudes):.4f}")
    
    # 모델 학습 (간단한 버전)
    config = create_simple_config()
    model = TimesNet(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    # DataLoader 생성
    from torch.utils.data import DataLoader, TensorDataset
    train_mask = torch.ones(X_train_ts.shape[0], X_train_ts.shape[1])
    val_mask = torch.ones(X_val_ts.shape[0], X_val_ts.shape[1])
    
    train_dataset = TensorDataset(
        torch.tensor(X_train_ts, dtype=torch.float32),
        train_mask,
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_ts, dtype=torch.float32),
        val_mask,
        torch.tensor(y_val, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 학습
    start_time = time.time()
    best_f1 = 0
    
    for epoch in range(config.train_epochs):
        # 학습
        model.train()
        for batch_x, batch_mask, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model.classification(batch_x, batch_mask)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # 검증
        model.eval()
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for batch_x, batch_mask, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_mask = batch_mask.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model.classification(batch_x, batch_mask)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_trues.extend(batch_y.cpu().numpy())
        
        f1 = f1_score(all_trues, all_preds, average='macro')
        if f1 > best_f1:
            best_f1 = f1
        
        print(f"  Epoch {epoch+1}: F1 = {f1:.4f}")
    
    training_time = time.time() - start_time
    print(f"  최종 F1: {best_f1:.4f}")
    print(f"  학습 시간: {training_time:.2f}초")
    
    return best_f1, training_time

def main():
    """메인 함수"""
    print("TimesNet 정규화 방법 비교 실험")
    print("=" * 50)
    
    # 데이터 로드
    train_df = pd.read_csv("datasests/train.csv")
    X = train_df.drop(columns=["ID", "target"]).values
    y = train_df["target"].values
    
    # 작은 샘플로 테스트 (속도를 위해)
    sample_size = 2000
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]
    
    # 학습/검증 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X_sample, y_sample, test_size=0.2, stratify=y_sample, random_state=123
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 다양한 정규화 방법 테스트
    scaling_methods = ['none', 'standard', 'minmax', 'robust', 'sequence']
    results = {}
    
    for method in scaling_methods:
        try:
            f1, time_taken = test_scaling_method(method, X_train, X_val, y_train, y_val, device)
            results[method] = {'f1': f1, 'time': time_taken}
        except Exception as e:
            print(f"  오류 발생: {e}")
            results[method] = {'f1': 0, 'time': 0}
    
    # 결과 출력
    print("\n" + "=" * 50)
    print("정규화 방법별 성능 비교")
    print("=" * 50)
    print(f"{'Method':<12} {'F1 Score':<10} {'Time(s)':<10}")
    print("-" * 50)
    
    for method, result in results.items():
        print(f"{method:<12} {result['f1']:<10.4f} {result['time']:<10.2f}")
    
    # 최고 성능 방법 찾기
    best_method = max(results.keys(), key=lambda x: results[x]['f1'])
    print(f"\n최고 성능: {best_method} (F1: {results[best_method]['f1']:.4f})")
    
    print("\n권장사항:")
    print("- FFT에 가장 적합한 정규화 방법을 사용하세요")
    print("- StandardScaler가 일반적으로 FFT에 가장 적합합니다")
    print("- 시계열별 정규화도 좋은 선택지입니다")

if __name__ == "__main__":
    main()
