#!/usr/bin/env python3
"""
TimesNet 스케줄러 비교 실험
다양한 학습률 스케줄러의 성능을 비교합니다.
"""

import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
import warnings

# Time-Series-Library 경로 추가
sys.path.append('./Time-Series-Library')

from models.TimesNet import Model as TimesNet
from utils.tools import EarlyStopping

warnings.filterwarnings('ignore')

def test_scheduler(scheduler_type, epochs=20):
    """특정 스케줄러로 모델 테스트"""
    print(f"\n=== {scheduler_type.upper()} 스케줄러 테스트 ===")
    
    # 데이터 로드
    train_df = pd.read_csv("datasests/train.csv")
    X = train_df.drop(columns=["ID", "target"]).values
    y = train_df["target"].values
    
    # 작은 샘플로 빠른 테스트
    sample_size = 2000
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]
    
    # 학습/검증 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X_sample, y_sample, test_size=0.2, stratify=y_sample, random_state=123
    )
    
    # 정규화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 시계열 변환
    X_train_ts = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_val_ts = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)
    
    # 모델 설정
    class TestConfig:
        def __init__(self, scheduler_type):
            self.task_name = 'classification'
            self.seq_len = 52
            self.label_len = 0
            self.pred_len = 0
            self.enc_in = 1
            self.num_class = 21
            self.e_layers = 1  # 빠른 테스트를 위해 1개 레이어
            self.d_model = 32
            self.d_ff = 64
            self.top_k = 3
            self.num_kernels = 4
            self.learning_rate = 0.001
            self.batch_size = 64
            self.train_epochs = epochs
            self.patience = 10
            self.dropout = 0.1
            self.embed = 'timeF'
            self.freq = 'h'
            self.use_gpu = torch.cuda.is_available()
            self.device = torch.device('cuda' if self.use_gpu else 'cpu')
            self.lradj = scheduler_type
    
    config = TestConfig(scheduler_type)
    model = TimesNet(config).to(config.device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 스케줄러 생성
    scheduler = None
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.5
        )
    elif scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.95
        )
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=False
        )
    elif scheduler_type == 'warmup_cosine':
        def lr_lambda(epoch):
            if epoch < 3:
                return epoch / 3
            else:
                return 0.5 * (1 + np.cos(np.pi * (epoch - 3) / (epochs - 3)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
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
    learning_rates = []
    val_f1_scores = []
    best_f1 = 0
    
    for epoch in range(epochs):
        # 학습
        model.train()
        for batch_x, batch_mask, batch_y in train_loader:
            batch_x = batch_x.to(config.device)
            batch_mask = batch_mask.to(config.device)
            batch_y = batch_y.to(config.device)
            
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
                batch_x = batch_x.to(config.device)
                batch_mask = batch_mask.to(config.device)
                batch_y = batch_y.to(config.device)
                
                outputs = model.classification(batch_x, batch_mask)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_trues.extend(batch_y.cpu().numpy())
        
        f1 = f1_score(all_trues, all_preds, average='macro')
        if f1 > best_f1:
            best_f1 = f1
        
        # 현재 학습률 기록
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        val_f1_scores.append(f1)
        
        # 스케줄러 업데이트
        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(f1)
            else:
                scheduler.step()
        
        print(f"  Epoch {epoch+1:2d}: LR={current_lr:.6f}, F1={f1:.4f}")
    
    return learning_rates, val_f1_scores, best_f1

def plot_scheduler_comparison(results):
    """스케줄러 비교 시각화"""
    plt.figure(figsize=(15, 10))
    
    # 학습률 변화 그래프
    plt.subplot(2, 2, 1)
    for scheduler_type, (lrs, f1s, best_f1) in results.items():
        plt.plot(lrs, label=f'{scheduler_type} (best F1: {best_f1:.4f})')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    # F1 스코어 변화 그래프
    plt.subplot(2, 2, 2)
    for scheduler_type, (lrs, f1s, best_f1) in results.items():
        plt.plot(f1s, label=f'{scheduler_type} (best: {best_f1:.4f})')
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    # 최고 성능 비교
    plt.subplot(2, 2, 3)
    scheduler_names = list(results.keys())
    best_f1s = [results[name][2] for name in scheduler_names]
    bars = plt.bar(scheduler_names, best_f1s)
    plt.title('Best F1 Score Comparison')
    plt.ylabel('Best F1 Score')
    plt.xticks(rotation=45)
    
    # 막대 위에 값 표시
    for bar, f1 in zip(bars, best_f1s):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{f1:.4f}', ha='center', va='bottom')
    
    # 학습률 분포 (마지막 5 에포크 평균)
    plt.subplot(2, 2, 4)
    final_lrs = [np.mean(lrs[-5:]) for lrs, _, _ in results.values()]
    bars = plt.bar(scheduler_names, final_lrs)
    plt.title('Final Learning Rate (Last 5 Epochs Avg)')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/scheduler_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 함수"""
    print("TimesNet 스케줄러 비교 실험")
    print("=" * 50)
    
    # 테스트할 스케줄러들
    schedulers = ['none', 'cosine', 'step', 'exponential', 'plateau', 'warmup_cosine']
    
    results = {}
    
    for scheduler_type in schedulers:
        try:
            lrs, f1s, best_f1 = test_scheduler(scheduler_type, epochs=15)
            results[scheduler_type] = (lrs, f1s, best_f1)
        except Exception as e:
            print(f"  오류 발생: {e}")
            results[scheduler_type] = ([], [], 0.0)
    
    # 결과 분석
    print("\n" + "=" * 50)
    print("스케줄러 성능 비교")
    print("=" * 50)
    
    print(f"{'Scheduler':<15} {'Best F1':<10} {'Final LR':<12} {'LR Range':<15}")
    print("-" * 60)
    
    for scheduler_type, (lrs, f1s, best_f1) in results.items():
        if lrs:
            final_lr = np.mean(lrs[-3:])  # 마지막 3 에포크 평균
            lr_range = f"{min(lrs):.2e} - {max(lrs):.2e}"
        else:
            final_lr = 0
            lr_range = "N/A"
        
        print(f"{scheduler_type:<15} {best_f1:<10.4f} {final_lr:<12.2e} {lr_range:<15}")
    
    # 최고 성능 스케줄러 찾기
    best_scheduler = max(results.keys(), key=lambda x: results[x][2])
    print(f"\n최고 성능 스케줄러: {best_scheduler} (F1: {results[best_scheduler][2]:.4f})")
    
    # 시각화
    try:
        plot_scheduler_comparison(results)
    except Exception as e:
        print(f"시각화 오류: {e}")
    
    print(f"\n권장사항:")
    print(f"- {best_scheduler} 스케줄러를 사용하세요")
    print(f"- timesnet_classification.py에서 lradj='{best_scheduler}'로 설정하세요")

if __name__ == "__main__":
    main()
