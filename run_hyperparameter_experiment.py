#!/usr/bin/env python3
"""
TimesNet 하이퍼파라미터 실험 스크립트
모든 정규화 방법과 하이퍼파라미터 조합을 실험하여 최적 설정을 찾습니다.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import time
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
import warnings

# Time-Series-Library 경로 추가
sys.path.append('./Time-Series-Library')

from models.TimesNet import Model as TimesNet
from utils.tools import EarlyStopping

warnings.filterwarnings('ignore')

class HyperparameterExperiment:
    """하이퍼파라미터 실험 클래스"""
    
    def __init__(self):
        self.results = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
    def load_data(self, sample_size=4000):
        """데이터 로드 (빠른 실험을 위해 샘플링)"""
        train_df = pd.read_csv("datasests/train.csv")
        X = train_df.drop(columns=["ID", "target"]).values
        y = train_df["target"].values
        
        # 샘플링
        if sample_size < len(X):
            indices = np.random.choice(len(X), sample_size, replace=False)
            X = X[indices]
            y = y[indices]
        
        # 학습/검증 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=123
        )
        
        return X_train, X_val, y_train, y_val
    
    def apply_scaling(self, X_train, X_val, scaling_method):
        """정규화 적용"""
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
            def normalize_sequence(seq):
                seq_mean = np.mean(seq, axis=1, keepdims=True)
                seq_std = np.std(seq, axis=1, keepdims=True)
                return (seq - seq_mean) / (seq_std + 1e-8)
            X_train_scaled = normalize_sequence(X_train)
            X_val_scaled = normalize_sequence(X_val)
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        return X_train_scaled, X_val_scaled
    
    def create_model_config(self, e_layers, d_model, d_ff, top_k, num_kernels, 
                           learning_rate, batch_size, train_epochs=10):
        """모델 설정 생성"""
        class ModelConfig:
            def __init__(self):
                self.task_name = 'classification'
                self.seq_len = 52
                self.label_len = 0
                self.pred_len = 0
                self.enc_in = 1
                self.num_class = 21
                self.e_layers = e_layers
                self.d_model = d_model
                self.d_ff = d_ff
                self.top_k = top_k
                self.num_kernels = num_kernels
                self.learning_rate = learning_rate
                self.batch_size = batch_size
                self.train_epochs = train_epochs
                self.patience = 5
                self.dropout = 0.1
                self.embed = 'timeF'
                self.freq = 'h'
                self.use_gpu = torch.cuda.is_available()
                self.device = self.device
                self.lradj = 'type1'
                self.use_wandb = False
        
        return ModelConfig()
    
    def train_model(self, X_train, X_val, y_train, y_val, config):
        """모델 학습"""
        try:
            # 시계열 변환
            X_train_ts = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val_ts = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            
            # 모델 생성
            model = TimesNet(config).to(self.device)
            optimizer = torch.optim.RAdam(model.parameters(), lr=config.learning_rate)
            criterion = torch.nn.CrossEntropyLoss()
            
            # DataLoader 생성
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
            best_f1 = 0
            for epoch in range(config.train_epochs):
                # 학습
                model.train()
                for batch_x, batch_mask, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_mask = batch_mask.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
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
                        batch_x = batch_x.to(self.device)
                        batch_mask = batch_mask.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = model.classification(batch_x, batch_mask)
                        preds = torch.argmax(outputs, dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_trues.extend(batch_y.cpu().numpy())
                
                f1 = f1_score(all_trues, all_preds, average='macro')
                if f1 > best_f1:
                    best_f1 = f1
            
            return best_f1, 'success'
            
        except Exception as e:
            print(f"오류 발생: {e}")
            return 0.0, 'error'
    
    def run_single_experiment(self, scaling_method, e_layers, d_model, d_ff, top_k, 
                            num_kernels, learning_rate, batch_size):
        """단일 실험 실행"""
        start_time = time.time()
        
        # 데이터 로드
        X_train, X_val, y_train, y_val = self.load_data()
        
        # 정규화 적용
        X_train_scaled, X_val_scaled = self.apply_scaling(X_train, X_val, scaling_method)
        
        # 모델 설정 생성
        config = self.create_model_config(e_layers, d_model, d_ff, top_k, num_kernels, 
                                        learning_rate, batch_size)
        
        # 모델 학습
        val_macro_f1, status = self.train_model(X_train_scaled, X_val_scaled, y_train, y_val, config)
        
        training_time = time.time() - start_time
        
        # 결과 저장
        result = {
            'scaling_method': scaling_method,
            'e_layers': e_layers,
            'd_model': d_model,
            'd_ff': d_ff,
            'top_k': top_k,
            'num_kernels': num_kernels,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'val_macro_f1': val_macro_f1,
            'training_time': training_time,
            'status': status
        }
        
        self.results.append(result)
        
        return result
    
    def run_all_experiments(self):
        """모든 실험 실행"""
        # 실험 파라미터 정의
        scaling_methods = ['standard', 'minmax', 'robust', 'sequence']
        
        # 하이퍼파라미터 조합
        param_combinations = [
            # 기본 설정들
            (2, 32, 64, 3, 4, 0.001, 64),
            (2, 64, 128, 5, 6, 0.001, 64),
            (3, 64, 128, 5, 6, 0.001, 64),
            (2, 128, 256, 5, 6, 0.001, 64),
            
            # 학습률 변화
            (2, 64, 128, 5, 6, 0.0005, 64),
            (2, 64, 128, 5, 6, 0.002, 64),
            (2, 64, 128, 5, 6, 0.005, 64),
            
            # 배치 크기 변화
            (2, 64, 128, 5, 6, 0.001, 32),
            (2, 64, 128, 5, 6, 0.001, 128),
            (2, 64, 128, 5, 6, 0.001, 256),
            
            # 모델 크기 변화
            (1, 32, 64, 3, 4, 0.001, 64),
            (4, 64, 128, 5, 6, 0.001, 64),
            (2, 32, 64, 3, 4, 0.001, 32),
            (2, 96, 192, 7, 8, 0.001, 64),
        ]
        
        total_experiments = len(scaling_methods) * len(param_combinations)
        experiment_count = 0
        
        print(f"총 실험 수: {total_experiments}")
        print("=" * 60)
        
        for scaling_method in scaling_methods:
            print(f"\n정규화 방법: {scaling_method}")
            print("-" * 40)
            
            for e_layers, d_model, d_ff, top_k, num_kernels, learning_rate, batch_size in param_combinations:
                experiment_count += 1
                
                print(f"[{experiment_count}/{total_experiments}] ", end="")
                print(f"e_layers={e_layers}, d_model={d_model}, d_ff={d_ff}, ", end="")
                print(f"top_k={top_k}, num_kernels={num_kernels}, lr={learning_rate}, batch_size={batch_size}")
                
                result = self.run_single_experiment(
                    scaling_method, e_layers, d_model, d_ff, top_k, 
                    num_kernels, learning_rate, batch_size
                )
                
                print(f"  → Val Macro F1: {result['val_macro_f1']:.4f}, "
                      f"시간: {result['training_time']:.1f}초, "
                      f"상태: {result['status']}")
        
        return self.results
    
    def analyze_results(self):
        """결과 분석"""
        if not self.results:
            print("실험 결과가 없습니다.")
            return
        
        # DataFrame 생성
        df = pd.DataFrame(self.results)
        
        # 결과 저장
        os.makedirs('results', exist_ok=True)
        df.to_csv('results/hyperparameter_experiment_results.csv', index=False)
        
        # 최고 성능 결과
        best_result = df.loc[df['val_macro_f1'].idxmax()]
        
        print("\n" + "=" * 60)
        print("실험 완료!")
        print("=" * 60)
        
        print(f"\n최고 성능 결과:")
        print(f"  Val Macro F1: {best_result['val_macro_f1']:.4f}")
        print(f"  정규화 방법: {best_result['scaling_method']}")
        print(f"  e_layers: {best_result['e_layers']}")
        print(f"  d_model: {best_result['d_model']}")
        print(f"  d_ff: {best_result['d_ff']}")
        print(f"  top_k: {best_result['top_k']}")
        print(f"  num_kernels: {best_result['num_kernels']}")
        print(f"  learning_rate: {best_result['learning_rate']}")
        print(f"  batch_size: {best_result['batch_size']}")
        print(f"  학습 시간: {best_result['training_time']:.1f}초")
        
        # 상위 5개 결과
        print(f"\n상위 5개 결과:")
        top5 = df.nlargest(5, 'val_macro_f1')
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            print(f"  {i}. {row['scaling_method']} - F1: {row['val_macro_f1']:.4f} "
                  f"(e_layers={row['e_layers']}, d_model={row['d_model']}, "
                  f"lr={row['learning_rate']}, batch_size={row['batch_size']})")
        
        # 정규화 방법별 평균 성능
        print(f"\n정규화 방법별 평균 성능:")
        scaling_avg = df.groupby('scaling_method')['val_macro_f1'].agg(['mean', 'std', 'count'])
        for method, stats in scaling_avg.iterrows():
            print(f"  {method}: 평균 F1 = {stats['mean']:.4f} ± {stats['std']:.4f} "
                  f"(실험 수: {stats['count']})")
        
        print(f"\n결과가 'results/hyperparameter_experiment_results.csv'에 저장되었습니다.")
        
        return best_result

def main():
    """메인 함수"""
    print("TimesNet 하이퍼파라미터 실험 시작")
    print("=" * 60)
    
    # 실험 실행
    experiment = HyperparameterExperiment()
    results = experiment.run_all_experiments()
    
    # 결과 분석
    best_result = experiment.analyze_results()
    
    print(f"\n최적 파라미터로 최종 모델을 학습하려면:")
    print(f"python timesnet_classification.py")

if __name__ == "__main__":
    main()
