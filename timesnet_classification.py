import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import argparse
import time
import warnings
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

# Time-Series-Library 경로 추가
sys.path.append('./Time-Series-Library')

from models.TimesNet import Model as TimesNet
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy

warnings.filterwarnings('ignore')

class TimesNetConfig:
    """TimesNet 모델 설정 클래스"""
    def __init__(self):
        # 기본 설정
        self.task_name = 'classification'
        self.seq_len = 52  # 시계열 길이 (52개 시점)
        self.label_len = 0
        self.pred_len = 0
        self.enc_in = 1  # 입력 특성 개수 (1차원 시계열)
        self.num_class = 21  # 클래스 개수 (0~20)
        
        # 모델 하이퍼파라미터
        self.e_layers = 2  # TimesBlock 레이어 수
        self.d_model = 64  # 모델 차원 (더 큰 모델로 성능 향상)
        self.d_ff = 128  # Feed-forward 차원
        self.top_k = 5  # FFT에서 선택할 상위 k개 주기 (더 많은 주기성 탐지)
        self.num_kernels = 6  # Inception block의 커널 수
        
        # 학습 설정
        self.learning_rate = 0.001
        self.batch_size = 64
        self.train_epochs = 50
        self.patience = 10
        self.dropout = 0.1
        
        # 기타 설정
        self.embed = 'timeF'  # 임베딩 타입
        self.freq = 'h'  # 주파수
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # Learning rate adjustment 설정
        self.lradj = 'cosine'  # learning rate adjustment 타입 (cosine, type1, type2, type3, step, plateau)
        
        # Wandb 설정
        self.use_wandb = True  # False로 설정하면 Wandb 로깅 비활성화
        self.wandb_project = 'timesnet-classification'
        self.wandb_entity = None  # 사용자의 wandb entity (선택사항)

class TimesNetTrainer:
    """TimesNet 모델 학습 클래스"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.early_stopping = None
        self.wandb_run = None
        
        # Wandb 초기화
        if self.config.use_wandb:
            self.init_wandb()
    
    def init_wandb(self):
        """Wandb 초기화"""
        wandb_config = {
            'model': 'TimesNet',
            'task': 'classification',
            'seq_len': self.config.seq_len,
            'enc_in': self.config.enc_in,
            'num_class': self.config.num_class,
            'e_layers': self.config.e_layers,
            'd_model': self.config.d_model,
            'd_ff': self.config.d_ff,
            'top_k': self.config.top_k,
            'num_kernels': self.config.num_kernels,
            'learning_rate': self.config.learning_rate,
            'batch_size': self.config.batch_size,
            'train_epochs': self.config.train_epochs,
            'patience': self.config.patience,
            'dropout': self.config.dropout,
            'device': str(self.config.device)
        }
        
        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            config=wandb_config,
            name=f"timesnet_{int(time.time())}"
        )
        
        print(f"Wandb 초기화 완료: {self.wandb_run.url}")
    
    def log_metrics(self, metrics, step=None):
        """Wandb에 메트릭 로깅"""
        if self.config.use_wandb and self.wandb_run:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
    
    def evaluate_detailed(self, val_loader):
        """상세한 평가 (F1 스코어, 분류 리포트 등)"""
        self.model.eval()
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for batch_x, batch_mask, batch_y in val_loader:
                batch_x = batch_x.to(self.config.device)
                batch_mask = batch_mask.to(self.config.device)
                batch_y = batch_y.to(self.config.device)
                
                outputs = self.model.classification(batch_x, batch_mask)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_trues.extend(batch_y.cpu().numpy())
        
        # 다양한 F1 스코어 계산
        macro_f1 = f1_score(all_trues, all_preds, average='macro')
        micro_f1 = f1_score(all_trues, all_preds, average='micro')
        weighted_f1 = f1_score(all_trues, all_preds, average='weighted')
        
        # 클래스별 F1 스코어
        f1_per_class = f1_score(all_trues, all_preds, average=None)
        
        # 정확도
        accuracy = accuracy_score(all_trues, all_preds)
        
        # 분류 리포트
        class_report = classification_report(all_trues, all_preds, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'weighted_f1': weighted_f1,
            'f1_per_class': f1_per_class,
            'classification_report': class_report,
            'predictions': all_preds,
            'true_labels': all_trues
        }
    
    def log_confusion_matrix(self, y_true, y_pred, class_names=None):
        """혼동 행렬을 Wandb에 로깅"""
        if self.config.use_wandb and self.wandb_run:
            cm = confusion_matrix(y_true, y_pred)
            
            # 혼동 행렬 시각화
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            # Wandb에 이미지로 로깅
            wandb.log({"confusion_matrix": wandb.Image(plt)})
            plt.close()
    
    def log_predictions_table(self, test_ids, predictions, probabilities=None):
        """예측 결과를 Wandb 테이블로 로깅"""
        if self.config.use_wandb and self.wandb_run:
            data = []
            for i, (test_id, pred) in enumerate(zip(test_ids, predictions)):
                row = {"ID": test_id, "prediction": int(pred)}
                if probabilities is not None:
                    for j, prob in enumerate(probabilities[i]):
                        row[f"prob_class_{j}"] = float(prob)
                data.append(row)
            
            table = wandb.Table(dataframe=pd.DataFrame(data))
            wandb.log({"predictions_table": table})
    
    def upload_csv(self, csv_path, artifact_name="submission"):
        """CSV 파일을 Wandb 아티팩트로 업로드"""
        if self.config.use_wandb and self.wandb_run:
            artifact = wandb.Artifact(artifact_name, type="dataset")
            artifact.add_file(csv_path)
            self.wandb_run.log_artifact(artifact)
            print(f"CSV 파일이 Wandb에 업로드되었습니다: {artifact_name}")
        
    def build_model(self):
        """모델 빌드"""
        self.model = TimesNet(self.config).to(self.config.device)
        self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=self.config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)
        
        # 스케줄러 설정
        self.scheduler = self._create_scheduler()
        
        print(f"모델 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"스케줄러: {self.config.lradj}")
    
    def _create_scheduler(self):
        """스케줄러 생성"""
        if self.config.lradj == 'cosine':
            # 코사인 어닐링 (가장 권장)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.train_epochs, eta_min=1e-6
            )
        elif self.config.lradj == 'step':
            # 스텝 스케줄러 (매 10 에포크마다 0.1배)
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
        elif self.config.lradj == 'plateau':
            # 플래토 스케줄러 (검증 손실이 개선되지 않으면 감소)
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=3, verbose=True
            )
        elif self.config.lradj == 'exponential':
            # 지수 감소 스케줄러
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.95
            )
        elif self.config.lradj == 'warmup_cosine':
            # 워밍업 + 코사인 어닐링
            def lr_lambda(epoch):
                if epoch < 5:  # 워밍업
                    return epoch / 5
                else:  # 코사인 어닐링
                    return 0.5 * (1 + np.cos(np.pi * (epoch - 5) / (self.config.train_epochs - 5)))
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            # 기본값: 스케줄러 없음
            return None
        
    def create_time_series_data(self, X):
        """특성 벡터를 시계열 데이터로 변환"""
        # 방법 1: 단순 reshape (52개 특성 -> 52개 시점)
        X_ts = X.reshape(X.shape[0], X.shape[1], 1)  # [N, 52, 1]
        
        # 방법 2: 특성들을 정렬하여 더 의미있는 시계열 생성
        # 각 샘플의 특성들을 값의 크기 순으로 정렬
        X_sorted = np.sort(X, axis=1)  # [N, 52]
        X_ts_sorted = X_sorted.reshape(X_sorted.shape[0], X_sorted.shape[1], 1)  # [N, 52, 1]
        
        # 방법 3: 특성들의 차분을 계산하여 변화율 시계열 생성
        X_diff = np.diff(X, axis=1)  # [N, 51]
        X_diff_padded = np.pad(X_diff, ((0, 0), (0, 1)), mode='constant', constant_values=0)  # [N, 52]
        X_ts_diff = X_diff_padded.reshape(X_diff_padded.shape[0], X_diff_padded.shape[1], 1)  # [N, 52, 1]
        
        # 원본 특성 시계열 사용 (가장 직관적)
        return X_ts
    
    def prepare_data(self, X_train, y_train, X_val, y_val, X_test=None):
        """데이터 준비 및 DataLoader 생성"""
        
        # 특성을 시계열로 변환
        X_train_ts = self.create_time_series_data(X_train)
        X_val_ts = self.create_time_series_data(X_val)
        
        # 설정 업데이트 (실제 시계열 길이와 특성 수에 맞춤)
        self.config.seq_len = X_train_ts.shape[1]
        self.config.enc_in = X_train_ts.shape[2]
        
        print(f"시계열 변환 후 형태: {X_train_ts.shape}")
        print(f"업데이트된 설정 - seq_len: {self.config.seq_len}, enc_in: {self.config.enc_in}")
        
        # 마스크 생성 (모든 시점이 유효하다고 가정)
        train_mask = torch.ones(X_train_ts.shape[0], X_train_ts.shape[1])
        val_mask = torch.ones(X_val_ts.shape[0], X_val_ts.shape[1])
        
        # DataLoader 생성
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
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        if X_test is not None:
            X_test_ts = self.create_time_series_data(X_test)
            test_mask = torch.ones(X_test_ts.shape[0], X_test_ts.shape[1])
            test_dataset = TensorDataset(torch.tensor(X_test_ts, dtype=torch.float32), test_mask)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
            return train_loader, val_loader, test_loader
        
        return train_loader, val_loader, None
    
    def train_epoch(self, train_loader):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_mask, batch_y in train_loader:
            batch_x = batch_x.to(self.config.device)
            batch_mask = batch_mask.to(self.config.device)
            batch_y = batch_y.to(self.config.device)
            
            self.optimizer.zero_grad()
            
            # TimesNet forward pass
            outputs = self.model.classification(batch_x, batch_mask)
            loss = self.criterion(outputs, batch_y)
            
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
        
        with torch.no_grad():
            for batch_x, batch_mask, batch_y in val_loader:
                batch_x = batch_x.to(self.config.device)
                batch_mask = batch_mask.to(self.config.device)
                batch_y = batch_y.to(self.config.device)
                
                outputs = self.model.classification(batch_x, batch_mask)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_trues.extend(batch_y.cpu().numpy())
        
        accuracy = accuracy_score(all_trues, all_preds)
        macro_f1 = f1_score(all_trues, all_preds, average='macro')
        return total_loss / len(val_loader), accuracy, macro_f1
    
    def train(self, train_loader, val_loader):
        """모델 학습"""
        print("학습 시작...")
        
        for epoch in range(self.config.train_epochs):
            start_time = time.time()
            
            # 학습
            train_loss = self.train_epoch(train_loader)
            
            # 검증
            val_loss, val_acc, val_macro_f1 = self.validate(val_loader)
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{self.config.train_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro F1: {val_macro_f1:.4f}')
            print(f'  Time: {epoch_time:.2f}s')
            print('-' * 50)
            
            # Wandb 로깅
            self.log_metrics({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_macro_f1': val_macro_f1,
                'epoch_time': epoch_time,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
            
            # Early Stopping (Macro F1 스코어 기준)
            # F1 스코어가 높을수록 좋으므로 -val_macro_f1을 사용
            self.early_stopping(-val_macro_f1, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping (Macro F1 기준)")
                break
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if self.config.lradj == 'plateau':
                    # 플래토 스케줄러는 검증 손실을 기준으로 함
                    self.scheduler.step(val_macro_f1)
                else:
                    # 다른 스케줄러들은 에포크 기준
                    self.scheduler.step()
            else:
                # 기존 방식 (type1, type2, type3, cosine)
                adjust_learning_rate(self.optimizer, epoch + 1, self.config)
    
    def predict(self, test_loader, return_probabilities=False):
        """예측"""
        self.model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch_x, batch_mask in test_loader:
                batch_x = batch_x.to(self.config.device)
                batch_mask = batch_mask.to(self.config.device)
                
                outputs = self.model.classification(batch_x, batch_mask)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                if return_probabilities:
                    all_probs.extend(probs.cpu().numpy())
        
        if return_probabilities:
            return np.array(all_preds), np.array(all_probs)
        return np.array(all_preds)

def fft_friendly_scaling(X_train, X_val, X_test, method='standard'):
    """FFT에 적합한 정규화 방법들"""
    
    if method == 'standard':
        # StandardScaler 사용 (FFT에 가장 적합)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        print("StandardScaler 적용 (FFT에 적합)")
        
    elif method == 'minmax':
        # MinMaxScaler 사용
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        print("MinMaxScaler 적용 (0-1 정규화)")
        
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
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        print("RobustScaler 적용 (기존 방식)")
        
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def load_and_preprocess_data(scaling_method='standard'):
    """데이터 로드 및 전처리"""
    print("데이터 로딩 중...")
    
    # 데이터 로드
    train_df = pd.read_csv("datasests/train.csv")
    test_df = pd.read_csv("datasests/test.csv")
    
    print(f"Train 데이터: {train_df.shape}")
    print(f"Test 데이터: {test_df.shape}")
    
    # 특성과 타겟 분리
    X = train_df.drop(columns=["ID", "target"]).values
    y = train_df["target"].values
    X_test = test_df.drop(columns=["ID"]).values
    test_ids = test_df["ID"].values
    
    print(f"클래스 개수: {len(np.unique(y))}")
    print(f"클래스 분포: {np.bincount(y)}")
    
    # 학습/검증 분할 (정규화 전에 분할)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=123
    )
    
    # FFT에 적합한 정규화 적용
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = fft_friendly_scaling(
        X_train, X_val, X_test, method=scaling_method
    )
    
    print(f"Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}")
    print(f"정규화 방법: {scaling_method}")
    
    return X_train_scaled, X_val_scaled, y_train, y_val, X_test_scaled, test_ids, scaler

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='TimesNet 분류 모델 학습')
    
    # 정규화 방법 선택
    parser.add_argument('--scaling', type=str, default='standard',
                       choices=['standard', 'minmax', 'robust', 'sequence'],
                       help='정규화 방법 선택 (default: standard)')
    
    # 스케줄러 선택
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'exponential', 'warmup_cosine', 'type1', 'type2', 'type3'],
                       help='학습률 스케줄러 선택 (default: cosine)')
    
    # 모델 하이퍼파라미터
    parser.add_argument('--e_layers', type=int, default=2,
                       help='TimesBlock 레이어 수 (default: 2)')
    parser.add_argument('--d_model', type=int, default=64,
                       help='모델 차원 (default: 64)')
    parser.add_argument('--d_ff', type=int, default=128,
                       help='Feed-forward 차원 (default: 128)')
    parser.add_argument('--top_k', type=int, default=5,
                       help='FFT에서 선택할 상위 k개 주기 (default: 5)')
    parser.add_argument('--num_kernels', type=int, default=6,
                       help='Inception block의 커널 수 (default: 6)')
    
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
    parser.add_argument('--wandb_project', type=str, default='timesnet-classification',
                       help='Wandb 프로젝트 이름 (default: timesnet-classification)')
    
    args = parser.parse_args()
    
    print("TimesNet 분류 모델 학습 시작")
    print("=" * 50)
    print(f"정규화 방법: {args.scaling}")
    print(f"스케줄러: {args.scheduler}")
    print(f"모델 설정: e_layers={args.e_layers}, d_model={args.d_model}, d_ff={args.d_ff}")
    print(f"학습 설정: lr={args.learning_rate}, batch_size={args.batch_size}, epochs={args.epochs}")
    print(f"Wandb 사용: {args.use_wandb}")
    print("=" * 50)
    
    # 데이터 로드 및 전처리
    X_train, X_val, y_train, y_val, X_test, test_ids, scaler = load_and_preprocess_data(args.scaling)
    
    # 설정 및 트레이너 초기화
    config = TimesNetConfig()
    
    # 명령행 인자로 설정 업데이트
    config.lradj = args.scheduler
    config.e_layers = args.e_layers
    config.d_model = args.d_model
    config.d_ff = args.d_ff
    config.top_k = args.top_k
    config.num_kernels = args.num_kernels
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    config.train_epochs = args.epochs
    config.patience = args.patience
    config.use_wandb = args.use_wandb
    config.wandb_project = args.wandb_project
    
    trainer = TimesNetTrainer(config)
    
    # 데이터 준비 (시계열 변환 후 설정 업데이트)
    train_loader, val_loader, test_loader = trainer.prepare_data(
        X_train, y_train, X_val, y_val, X_test
    )
    
    # 업데이트된 설정으로 모델 빌드
    trainer.build_model()
    
    # 모델 학습
    trainer.train(train_loader, val_loader)
    
    # 최종 검증
    print("\n최종 검증 결과:")
    val_loss, val_acc, val_macro_f1 = trainer.validate(val_loader)
    print(f"검증 정확도: {val_acc:.4f}")
    print(f"검증 Macro F1: {val_macro_f1:.4f}")
    
    # 상세한 평가 수행
    print("\n상세한 평가 수행 중...")
    detailed_eval = trainer.evaluate_detailed(val_loader)
    
    print(f"\n=== 상세 평가 결과 ===")
    print(f"정확도 (Accuracy): {detailed_eval['accuracy']:.4f}")
    print(f"Macro F1: {detailed_eval['macro_f1']:.4f}")
    print(f"Micro F1: {detailed_eval['micro_f1']:.4f}")
    print(f"Weighted F1: {detailed_eval['weighted_f1']:.4f}")
    
    # 클래스별 F1 스코어 출력 (상위 10개만)
    f1_scores = detailed_eval['f1_per_class']
    print(f"\n클래스별 F1 스코어 (상위 10개):")
    sorted_indices = np.argsort(f1_scores)[::-1]
    for i, idx in enumerate(sorted_indices[:10]):
        print(f"  클래스 {idx}: {f1_scores[idx]:.4f}")
    
    # 최종 메트릭을 Wandb에 로깅
    trainer.log_metrics({
        'final_val_loss': val_loss,
        'final_val_accuracy': detailed_eval['accuracy'],
        'final_val_macro_f1': detailed_eval['macro_f1'],
        'final_val_micro_f1': detailed_eval['micro_f1'],
        'final_val_weighted_f1': detailed_eval['weighted_f1']
    })
    
    # 클래스별 F1 스코어를 Wandb에 로깅
    class_f1_dict = {f'f1_class_{i}': score for i, score in enumerate(f1_scores)}
    trainer.log_metrics(class_f1_dict)
    
    # 혼동 행렬 로깅
    class_names = [f'Class_{i}' for i in range(len(f1_scores))]
    trainer.log_confusion_matrix(detailed_eval['true_labels'], detailed_eval['predictions'], class_names)
    
    # 테스트 데이터 예측
    if test_loader is not None:
        print("\n테스트 데이터 예측 중...")
        test_predictions, test_probabilities = trainer.predict(test_loader, return_probabilities=True)
        
        # 결과 저장
        submission = pd.DataFrame({
            'ID': test_ids,
            'target': test_predictions
        })
        submission.to_csv('timesnet_submission.csv', index=False)
        print("예측 결과가 'timesnet_submission.csv'에 저장되었습니다.")
        
        # Wandb에 CSV 업로드
        trainer.upload_csv('timesnet_submission.csv', 'timesnet_submission')
        
        # Wandb에 예측 결과 테이블 로깅
        trainer.log_predictions_table(test_ids, test_predictions, test_probabilities)
        
        # 예측 결과 분포 확인
        print(f"예측 클래스 분포: {np.bincount(test_predictions)}")
        
        # Wandb에 예측 분포 로깅
        class_distribution = np.bincount(test_predictions)
        distribution_dict = {f'pred_class_{i}': count for i, count in enumerate(class_distribution)}
        trainer.log_metrics(distribution_dict)
    
    # Wandb 세션 종료
    if trainer.config.use_wandb and trainer.wandb_run:
        wandb.finish()
        print("Wandb 세션이 종료되었습니다.")

if __name__ == "__main__":
    main()
