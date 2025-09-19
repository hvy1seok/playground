#!/usr/bin/env python3
"""
iTransformer 분류 모델 학습 스크립트
Time-Series-Library의 iTransformer 모델을 사용한 분류 학습
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

# Time-Series-Library 경로 추가
sys.path.append('./Time-Series-Library')

from models.iTransformer import Model as iTransformer
from utils.tools import EarlyStopping, adjust_learning_rate

class iTransformerConfig:
    """iTransformer 설정 클래스"""
    def __init__(self):
        # 기본 설정
        self.task_name = 'classification'
        self.seq_len = 52  # 시계열 길이 (데이터에 따라 조정)
        self.pred_len = 0  # 분류에서는 사용하지 않음
        self.enc_in = 1    # 입력 차원 (데이터에 따라 조정)
        self.num_class = 21  # 클래스 수 (데이터에 따라 조정)
        
        # 모델 하이퍼파라미터
        self.e_layers = 2      # 인코더 레이어 수
        self.d_model = 64      # 모델 차원
        self.d_ff = 128        # Feed-forward 차원
        self.n_heads = 8       # 어텐션 헤드 수
        self.factor = 1        # 어텐션 팩터
        self.dropout = 0.1     # 드롭아웃 비율
        self.activation = 'gelu'  # 활성화 함수
        
        # 임베딩 설정
        self.embed = 'timeF'   # 임베딩 타입
        self.freq = 'h'        # 주파수
        
        # 학습 설정
        self.learning_rate = 0.001
        self.batch_size = 64
        self.train_epochs = 50
        self.patience = 10
        
        # 스케줄러 설정
        self.lradj = 'cosine'  # learning rate adjustment 타입
        
        # Wandb 설정
        self.use_wandb = True
        self.wandb_project = 'itransformer-classification'
        self.wandb_entity = None

class iTransformerTrainer:
    """iTransformer 트레이너 클래스"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.early_stopping = None
        self.wandb_run = None
        
        if self.config.use_wandb:
            self.init_wandb()
    
    def init_wandb(self):
        """Wandb 초기화"""
        run_name = f"itransformer_{int(time.time())}"
        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=run_name,
            config=vars(self.config)
        )
        print(f"Wandb 초기화 완료: {self.wandb_run.url}")
    
    def log_metrics(self, metrics, step=None):
        """Wandb에 메트릭 로깅"""
        if self.wandb_run:
            wandb.log(metrics, step=step)
    
    def log_confusion_matrix(self, y_true, y_pred, class_names=None):
        """혼동 행렬 로깅"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if self.wandb_run:
            wandb.log({"confusion_matrix": wandb.Image(plt)})
        plt.close()
    
    def log_predictions_table(self, test_ids, predictions, probabilities=None):
        """예측 결과 테이블 로깅"""
        if self.wandb_run:
            data = {'ID': test_ids, 'Prediction': predictions}
            if probabilities is not None:
                for i in range(probabilities.shape[1]):
                    data[f'Class_{i}_Prob'] = probabilities[:, i]
            
            df = pd.DataFrame(data)
            table = wandb.Table(dataframe=df)
            wandb.log({"predictions_table": table})
    
    def upload_csv(self, csv_path, artifact_name="submission"):
        """CSV 파일을 Wandb에 업로드"""
        if self.wandb_run:
            artifact = wandb.Artifact(artifact_name, type="dataset")
            artifact.add_file(csv_path)
            self.wandb_run.log_artifact(artifact)
            print(f"CSV 파일이 Wandb에 업로드되었습니다: {artifact_name}")
    
    def create_time_series_data(self, X):
        """데이터를 시계열 형태로 변환"""
        # X: [N, features] -> [N, seq_len, enc_in]
        X_ts = X.reshape(X.shape[0], X.shape[1], 1)  # [N, 52, 1]
        return X_ts
    
    def prepare_data(self, X_train, y_train, X_val, y_val, X_test):
        """데이터 준비 및 DataLoader 생성"""
        # 시계열 데이터로 변환
        X_train_ts = self.create_time_series_data(X_train)
        X_val_ts = self.create_time_series_data(X_val)
        X_test_ts = self.create_time_series_data(X_test)
        
        print(f"시계열 변환 후 형태: {X_train_ts.shape}")
        
        # 설정 업데이트
        self.config.seq_len = X_train_ts.shape[1]
        self.config.enc_in = X_train_ts.shape[2]
        self.config.num_class = len(np.unique(y_train))
        
        print(f"업데이트된 설정 - seq_len: {self.config.seq_len}, enc_in: {self.config.enc_in}")
        
        # DataLoader 생성
        train_dataset = TensorDataset(
            torch.tensor(X_train_ts, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_ts, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test_ts, dtype=torch.float32)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def build_model(self):
        """모델 빌드"""
        self.model = iTransformer(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)
        self.scheduler = self._create_scheduler()
        
        # 모델 파라미터 수 계산
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"모델 파라미터 수: {total_params:,}")
        print(f"학습 가능한 파라미터 수: {trainable_params:,}")
    
    def _create_scheduler(self):
        """학습률 스케줄러 생성"""
        if self.config.lradj == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.train_epochs, eta_min=1e-6
            )
        elif self.config.lradj == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config.train_epochs//3, gamma=0.5
            )
        elif self.config.lradj == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
            )
        elif self.config.lradj == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.95
            )
        elif self.config.lradj == 'warmup_cosine':
            def lr_lambda(epoch):
                if epoch < 5:  # 워밍업
                    return epoch / 5
                else:
                    return 0.5 * (1 + np.cos(np.pi * (epoch - 5) / (self.config.train_epochs - 5)))
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            return None
    
    def train_epoch(self, train_loader):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # iTransformer는 x_mark_enc를 None으로 전달
            outputs = self.model(batch_x, None, None, None)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """검증"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_trues = []
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x, None, None, None)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_trues.extend(batch_y.cpu().numpy())
        
        accuracy = accuracy_score(all_trues, all_preds)
        macro_f1 = f1_score(all_trues, all_preds, average='macro')
        
        return total_loss / len(val_loader), accuracy, macro_f1
    
    def train(self, train_loader, val_loader):
        """전체 학습 과정"""
        print("학습 시작...")
        start_time = time.time()
        
        for epoch in range(self.config.train_epochs):
            epoch_start = time.time()
            
            # 학습
            train_loss = self.train_epoch(train_loader)
            
            # 검증
            val_loss, val_acc, val_macro_f1 = self.validate(val_loader)
            
            # 스케줄러 업데이트
            if self.scheduler is not None:
                if self.config.lradj == 'plateau':
                    self.scheduler.step(val_macro_f1)
                else:
                    self.scheduler.step()
            else:
                # 기존 adjust_learning_rate 사용
                adjust_learning_rate(self.optimizer, epoch + 1, self.config)
            
            epoch_time = time.time() - epoch_start
            
            # 로깅
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_macro_f1': val_macro_f1,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time
            }
            
            self.log_metrics(metrics)
            
            print(f"Epoch {epoch+1}/{self.config.train_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro F1: {val_macro_f1:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            print("-" * 50)
            
            # Early stopping
            self.early_stopping(-val_macro_f1, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
        
        total_time = time.time() - start_time
        print(f"학습 완료! 총 시간: {total_time:.2f}s")
        
        return total_time
    
    def evaluate_detailed(self, val_loader):
        """상세 평가"""
        self.model.eval()
        all_preds = []
        all_trues = []
        all_probs = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x, None, None, None)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_trues.extend(batch_y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # 메트릭 계산
        accuracy = accuracy_score(all_trues, all_preds)
        macro_f1 = f1_score(all_trues, all_preds, average='macro')
        micro_f1 = f1_score(all_trues, all_preds, average='micro')
        weighted_f1 = f1_score(all_trues, all_preds, average='weighted')
        f1_per_class = f1_score(all_trues, all_preds, average=None)
        
        print(f"\n=== 상세 평가 결과 ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Micro F1: {micro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        
        # 클래스별 F1 점수
        print(f"\n클래스별 F1 점수:")
        for i, f1 in enumerate(f1_per_class):
            print(f"  클래스 {i}: {f1:.4f}")
        
        # 분류 리포트
        print(f"\n분류 리포트:")
        print(classification_report(all_trues, all_preds))
        
        # 혼동 행렬
        self.log_confusion_matrix(all_trues, all_preds)
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'weighted_f1': weighted_f1,
            'f1_per_class': f1_per_class,
            'predictions': all_preds,
            'probabilities': np.array(all_probs)
        }
    
    def predict(self, test_loader):
        """테스트 데이터 예측"""
        self.model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch_x in test_loader:
                batch_x = batch_x[0].to(self.device)
                
                outputs = self.model(batch_x, None, None, None)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs)

def fft_friendly_scaling(X_train, X_val, X_test, method='standard', scope='global'):
    """FFT에 적합한 정규화 방법들"""
    if method == 'standard':
        from sklearn.preprocessing import StandardScaler
        if scope == 'global':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            print("StandardScaler 적용 (전체 데이터 기준, FFT에 적합)")
        else:  # column
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            print("StandardScaler 적용 (컬럼별, FFT에 적합)")
        
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        if scope == 'global':
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            print("MinMaxScaler 적용 (전체 데이터 기준, 0-1 정규화)")
        else:  # column
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            print("MinMaxScaler 적용 (컬럼별, 0-1 정규화)")
        
    elif method == 'sequence':
        # 시계열별 정규화 (각 샘플의 시계열을 개별 정규화)
        def normalize_sequence(seq):
            seq_mean = np.mean(seq, axis=1, keepdims=True)
            seq_std = np.std(seq, axis=1, keepdims=True)
            return (seq - seq_mean) / (seq_std + 1e-8)
        
        X_train_scaled = normalize_sequence(X_train)
        X_val_scaled = normalize_sequence(X_val)
        X_test_scaled = normalize_sequence(X_test)
        scaler = None  # 시계열별 정규화는 scaler 객체가 없음
        print("시계열별 정규화 적용 (각 샘플 독립 정규화)")
        
    elif method == 'robust':
        # 기존 RobustScaler (비교용)
        from sklearn.preprocessing import RobustScaler
        if scope == 'global':
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            print("RobustScaler 적용 (전체 데이터 기준, 기존 방식)")
        else:  # column
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            print("RobustScaler 적용 (컬럼별, 기존 방식)")
        
    elif method == 'none':
        # 스케일링 없음 (원본 데이터 그대로 사용)
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()
        scaler = None
        print("스케일링 없음 (원본 데이터 사용)")
        
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def load_and_preprocess_data(scaling_method='standard', scaling_scope='global'):
    """데이터 로드 및 전처리"""
    print("데이터 로딩 중...")
    
    # 데이터 로드
    train_df = pd.read_csv("./datasests/train.csv")
    test_df = pd.read_csv("./datasests/test.csv")
    
    print(f"Train 데이터: {train_df.shape}")
    print(f"Test 데이터: {test_df.shape}")
    
    # 특성과 타겟 분리
    X = train_df.drop(columns=["ID", "target"]).values
    y = train_df["target"].values
    X_test = test_df.drop(columns=["ID"]).values
    test_ids = test_df["ID"].values
    
    print(f"클래스 개수: {len(np.unique(y))}")
    print(f"클래스 분포: {np.bincount(y)}")
    
    # 학습/검증 분할 (stratify=y로 클래스 비율 유지)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=123
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # 정규화 적용
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = fft_friendly_scaling(
        X_train, X_val, X_test, method=scaling_method, scope=scaling_scope
    )
    
    print(f"Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}")
    print(f"정규화 방법: {scaling_method}, 범위: {scaling_scope}")
    
    return X_train_scaled, X_val_scaled, y_train, y_val, X_test_scaled, test_ids, scaler

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='iTransformer 분류 모델 학습')
    
    # 정규화 방법 선택
    parser.add_argument('--scaling', type=str, default='standard',
                       choices=['standard', 'minmax', 'robust', 'sequence', 'none'],
                       help='정규화 방법 선택 (default: standard)')
    
    # 스케일링 범위 선택
    parser.add_argument('--scaling_scope', type=str, default='global',
                       choices=['global', 'column'],
                       help='스케일링 범위 선택: global(전체), column(컬럼별) (default: global)')
    
    # 스케줄러 선택
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'exponential', 'warmup_cosine', 'type1', 'type2', 'type3'],
                       help='학습률 스케줄러 선택 (default: cosine)')
    
    # 모델 하이퍼파라미터
    parser.add_argument('--e_layers', type=int, default=2,
                       help='인코더 레이어 수 (default: 2)')
    parser.add_argument('--d_model', type=int, default=64,
                       help='모델 차원 (default: 64)')
    parser.add_argument('--d_ff', type=int, default=128,
                       help='Feed-forward 차원 (default: 128)')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='어텐션 헤드 수 (default: 8)')
    parser.add_argument('--factor', type=int, default=1,
                       help='어텐션 팩터 (default: 1)')
    
    # 학습 설정
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='학습률 (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='배치 크기 (default: 64)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='학습 에포크 수 (default: 50)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    
    # Wandb 설정
    parser.add_argument('--use_wandb', action='store_true', default=True,
                       help='Wandb 로깅 사용 (default: True)')
    parser.add_argument('--no_wandb', dest='use_wandb', action='store_false',
                       help='Wandb 로깅 비활성화')
    parser.add_argument('--wandb_project', type=str, default='itransformer-classification',
                       help='Wandb 프로젝트 이름 (default: itransformer-classification)')
    
    args = parser.parse_args()
    
    print("iTransformer 분류 모델 학습 시작")
    print("=" * 50)
    print(f"정규화 방법: {args.scaling}")
    print(f"스케일링 범위: {args.scaling_scope}")
    print(f"스케줄러: {args.scheduler}")
    print(f"모델 설정: e_layers={args.e_layers}, d_model={args.d_model}, d_ff={args.d_ff}")
    print(f"학습 설정: lr={args.learning_rate}, batch_size={args.batch_size}, epochs={args.epochs}")
    print(f"Wandb 사용: {args.use_wandb}")
    print("=" * 50)
    
    # 데이터 로드 및 전처리
    X_train, X_val, y_train, y_val, X_test, test_ids, scaler = load_and_preprocess_data(args.scaling, args.scaling_scope)
    
    # 설정 및 트레이너 초기화
    config = iTransformerConfig()
    
    # 명령행 인자로 설정 업데이트
    config.lradj = args.scheduler
    config.e_layers = args.e_layers
    config.d_model = args.d_model
    config.d_ff = args.d_ff
    config.n_heads = args.n_heads
    config.factor = args.factor
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    config.train_epochs = args.epochs
    config.patience = args.patience
    config.use_wandb = args.use_wandb
    config.wandb_project = args.wandb_project
    
    trainer = iTransformerTrainer(config)
    
    # 데이터 준비 (시계열 변환 후 설정 업데이트)
    train_loader, val_loader, test_loader = trainer.prepare_data(
        X_train, y_train, X_val, y_val, X_test
    )
    
    # 모델 빌드
    trainer.build_model()
    
    # 학습
    training_time = trainer.train(train_loader, val_loader)
    
    # 상세 평가
    eval_results = trainer.evaluate_detailed(val_loader)
    
    # 테스트 예측
    test_predictions, test_probabilities = trainer.predict(test_loader)
    
    # 결과 저장
    submission_df = pd.DataFrame({
        'ID': test_ids,
        'target': test_predictions
    })
    submission_df.to_csv('itransformer_submission.csv', index=False)
    print("예측 결과가 'itransformer_submission.csv'에 저장되었습니다.")
    
    # Wandb에 CSV 업로드
    if config.use_wandb:
        trainer.upload_csv('itransformer_submission.csv', 'itransformer_submission')
        trainer.log_predictions_table(test_ids, test_predictions, test_probabilities)
        
        # 최종 메트릭 로깅
        final_metrics = {
            'final_val_accuracy': eval_results['accuracy'],
            'final_val_macro_f1': eval_results['macro_f1'],
            'final_val_micro_f1': eval_results['micro_f1'],
            'final_val_weighted_f1': eval_results['weighted_f1'],
            'training_time': training_time
        }
        trainer.log_metrics(final_metrics)
    
    print(f"\n최종 성능:")
    print(f"Validation Accuracy: {eval_results['accuracy']:.4f}")
    print(f"Validation Macro F1: {eval_results['macro_f1']:.4f}")
    print(f"학습 시간: {training_time:.2f}초")
    
    if config.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
