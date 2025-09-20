#!/usr/bin/env python3
"""
iTransformer 분류 모델 학습 스크립트
Time-Series-Library의 iTransformer 모델을 사용한 분류 학습
"""

import os
import sys
import argparse
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import math
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import pickle
import os

# 시드 고정 함수
def set_seed(seed=42):
    """재현 가능한 결과를 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"시드 고정 완료: {seed}")

# Time-Series-Library 경로 추가
sys.path.append('./Time-Series-Library')

from models.iTransformer import Model as iTransformer
from utils.tools import EarlyStopping, adjust_learning_rate

# 고급 손실 함수들
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ClassWeightedFocalLoss(nn.Module):
    """Class-weighted Focal Loss"""
    def __init__(self, class_weights, alpha=1, gamma=2):
        super(ClassWeightedFocalLoss, self).__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 클래스별 가중치 적용 (GPU로 이동)
        weights = self.class_weights.to(inputs.device)[targets]
        focal_loss = weights * self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

class PairwiseMarginLoss(nn.Module):
    """Pairwise Margin Loss for difficult class pairs"""
    def __init__(self, margin=0.5, difficult_pairs=[(0, 9), (0, 15), (9, 15)]):
        super(PairwiseMarginLoss, self).__init__()
        self.margin = margin
        self.difficult_pairs = difficult_pairs

    def forward(self, features, targets):
        # features: [batch_size, feature_dim]
        # targets: [batch_size]
        
        loss = 0
        valid_pairs = 0
        for class_a, class_b in self.difficult_pairs:
            mask_a = (targets == class_a)
            mask_b = (targets == class_b)
            
            if mask_a.sum() > 0 and mask_b.sum() > 0:
                features_a = features[mask_a]
                features_b = features[mask_b]
                
                # Cosine similarity between class pairs
                sim_aa = F.cosine_similarity(features_a.unsqueeze(1), features_a.unsqueeze(0), dim=2)
                sim_bb = F.cosine_similarity(features_b.unsqueeze(1), features_b.unsqueeze(0), dim=2)
                sim_ab = F.cosine_similarity(features_a.unsqueeze(1), features_b.unsqueeze(0), dim=2)
                
                # Margin loss: same class should be similar, different classes should be dissimilar
                loss += F.relu(self.margin - sim_aa.mean()) + F.relu(sim_ab.mean() + self.margin)
                valid_pairs += 1
        
        if valid_pairs > 0:
            return loss / valid_pairs
        else:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss"""
    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, targets):
        # features: [batch_size, feature_dim]
        # targets: [batch_size]
        
        batch_size = features.shape[0]
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same class)
        targets = targets.contiguous().view(-1, 1)
        mask = torch.eq(targets, targets.T).float()
        
        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size).to(features.device)
        
        # Check if there are any positive pairs
        num_pos_pairs = mask.sum(1)
        valid_samples = num_pos_pairs > 0
        
        if valid_samples.sum() == 0:
            # No positive pairs in batch, return zero loss
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # Compute contrastive loss only for valid samples
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)  # Add small epsilon
        
        # Only consider positive pairs for valid samples
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)  # Add small epsilon
        
        # Only compute loss for samples with positive pairs
        loss = -mean_log_prob_pos[valid_samples].mean()
        
        # Check for NaN
        if torch.isnan(loss):
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        return loss

# 데이터 증강 기법들
class TimeWarp:
    """Time Warping augmentation"""
    def __init__(self, sigma=0.2):
        self.sigma = sigma
    
    def __call__(self, x):
        # x: [seq_len, features]
        seq_len = x.shape[0]
        warp_steps = np.arange(seq_len)
        
        # Generate warping curve
        warp_curve = np.random.normal(0, self.sigma, seq_len)
        warp_curve = np.cumsum(warp_curve)
        warp_curve = warp_curve - warp_curve[0]
        warp_curve = warp_curve * (seq_len - 1) / warp_curve[-1]
        
        # Apply warping
        warped_x = np.zeros_like(x)
        for i in range(seq_len):
            idx = int(warp_curve[i])
            if 0 <= idx < seq_len:
                warped_x[i] = x[idx]
            else:
                warped_x[i] = x[i]
        
        return warped_x

class Jitter:
    """Jittering augmentation"""
    def __init__(self, sigma=0.03):
        self.sigma = sigma
    
    def __call__(self, x):
        noise = np.random.normal(0, self.sigma, x.shape)
        return x + noise

class WindowSlicing:
    """Window Slicing augmentation"""
    def __init__(self, reduce_ratio=0.9):
        self.reduce_ratio = reduce_ratio
    
    def __call__(self, x):
        seq_len = x.shape[0]
        target_len = int(seq_len * self.reduce_ratio)
        
        if target_len < seq_len:
            start = np.random.randint(0, seq_len - target_len)
            return x[start:start + target_len]
        return x

class TSMixup:
    """Time Series Mixup augmentation"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x1, x2, y1, y2):
        # x1, x2: [seq_len, features]
        # y1, y2: class labels
        
        if np.random.random() < 0.5:
            # Mixup
            lam = np.random.beta(self.alpha, self.alpha)
            mixed_x = lam * x1 + (1 - lam) * x2
            mixed_y = lam * y1 + (1 - lam) * y2
        else:
            # CutMix
            seq_len = x1.shape[0]
            cut_len = int(seq_len * np.random.beta(self.alpha, self.alpha))
            start = np.random.randint(0, seq_len - cut_len)
            
            mixed_x = x1.copy()
            mixed_x[start:start + cut_len] = x2[start:start + cut_len]
            mixed_y = y1  # Keep original label for cutmix
        
        return mixed_x, mixed_y

# Cosine Attention 구현
class CosineAttention(nn.Module):
    """Cosine Attention: normalize(Q)·normalize(K)^T 사용"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(CosineAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)  # 스케일링 팩터
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # 1. Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Cosine similarity: normalize(Q)·normalize(K)^T
        Q_norm = F.normalize(Q, p=2, dim=-1)  # L2 normalization
        K_norm = F.normalize(K, p=2, dim=-1)  # L2 normalization
        
        # 3. Cosine similarity matrix
        scores = torch.matmul(Q_norm, K_norm.transpose(-2, -1))
        
        # 4. Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 5. Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 6. Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # 7. Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 8. Final linear projection
        output = self.w_o(context)
        
        return output, attention_weights

# CV 앙상블 관련 함수들
def save_model_checkpoint(model, scaler, fold, cv_score, save_dir="cv_models"):
    """모델과 스케일러를 저장"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 모델 저장
    model_path = os.path.join(save_dir, f"model_fold_{fold}.pth")
    torch.save(model.state_dict(), model_path)
    
    # 스케일러 저장
    scaler_path = os.path.join(save_dir, f"scaler_fold_{fold}.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # 메타데이터 저장
    metadata = {
        'fold': fold,
        'cv_score': cv_score,
        'model_path': model_path,
        'scaler_path': scaler_path
    }
    
    metadata_path = os.path.join(save_dir, f"metadata_fold_{fold}.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    return metadata

def load_model_checkpoint(model, fold, save_dir="cv_models"):
    """모델과 스케일러를 로드"""
    model_path = os.path.join(save_dir, f"model_fold_{fold}.pth")
    scaler_path = os.path.join(save_dir, f"scaler_fold_{fold}.pkl")
    
    # 모델 로드
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    # 스케일러 로드
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def get_cv_weights(cv_scores):
    """CV 점수 기반 가중치 계산"""
    # 소프트맥스 가중치 (점수가 높을수록 가중치 증가)
    scores = np.array(cv_scores)
    weights = np.exp(scores - np.max(scores))  # 수치 안정성을 위해 최대값 빼기
    weights = weights / np.sum(weights)
    return weights

def ensemble_predict(models, X_test, device, weights=None):
    """앙상블 예측"""
    all_logits = []
    
    for i, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            logits = model(X_test_tensor, None, None, None)
            all_logits.append(logits.cpu().numpy())
    
    # 가중 평균
    if weights is not None:
        weighted_logits = np.average(all_logits, axis=0, weights=weights)
    else:
        weighted_logits = np.mean(all_logits, axis=0)
    
    # 최종 예측
    predictions = np.argmax(weighted_logits, axis=1)
    probabilities = F.softmax(torch.tensor(weighted_logits), dim=1).numpy()
    
    return predictions, probabilities, weighted_logits

def run_cv_ensemble(X_train, y_train, X_test, test_ids, config, args):
    """CV 앙상블 실행"""
    print("=" * 60)
    print("CV 앙상블 학습 시작")
    print(f"Fold 수: {config.n_folds}")
    print("=" * 60)
    
    # K-Fold 설정
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=42)
    
    cv_scores = []
    models = []
    scalers = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\nFold {fold + 1}/{config.n_folds} 학습 중...")
        print("-" * 40)
        
        # 폴드별 데이터 분할
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]
        
        # 정규화 적용
        X_fold_train_scaled, X_fold_val_scaled, X_test_scaled, scaler = fft_friendly_scaling(
            X_fold_train, X_fold_val, X_test, method=args.scaling, scope=args.scaling_scope
        )
        
        # 트레이너 초기화
        fold_config = iTransformerConfig()
        fold_config.__dict__.update(config.__dict__)  # 설정 복사
        fold_config.use_wandb = False  # CV 중에는 Wandb 비활성화
        
        trainer = iTransformerTrainer(fold_config)
        
        # 데이터 준비
        train_loader, val_loader, test_loader = trainer.prepare_data(
            X_fold_train_scaled, y_fold_train, X_fold_val_scaled, y_fold_val, X_test_scaled
        )
        
        # 모델 빌드
        trainer.build_model()
        
        # 학습
        print(f"Fold {fold + 1} 학습 시작...")
        training_time = trainer.train(train_loader, val_loader)
        
        # 검증
        val_loss, val_acc, val_macro_f1 = trainer.validate(val_loader)
        cv_scores.append(val_macro_f1)
        
        print(f"Fold {fold + 1} 완료:")
        print(f"  Val Macro F1: {val_macro_f1:.4f}")
        print(f"  학습 시간: {training_time:.2f}s")
        
        # 모델 저장
        save_model_checkpoint(trainer.model, scaler, fold, val_macro_f1, config.cv_save_dir)
        models.append(trainer.model)
        scalers.append(scaler)
    
    # CV 결과 요약
    print("\n" + "=" * 60)
    print("CV 결과 요약")
    print("=" * 60)
    print(f"CV 점수: {cv_scores}")
    print(f"평균 CV 점수: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"최고 CV 점수: {np.max(cv_scores):.4f}")
    print(f"최저 CV 점수: {np.min(cv_scores):.4f}")
    
    # 가중치 계산
    cv_weights = get_cv_weights(cv_scores)
    print(f"CV 가중치: {cv_weights}")
    
    # 앙상블 예측
    print("\n앙상블 예측 생성 중...")
    X_test_ts = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predictions, probabilities, logits = ensemble_predict(models, X_test_ts, trainer.device, cv_weights)
    
    # Submission 파일 생성
    submission_df = pd.DataFrame({
        'ID': test_ids,
        'target': predictions
    })
    
    submission_path = f"itransformer_cv_ensemble_submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission 파일 저장: {submission_path}")
    
    # 상세 결과 저장
    results_df = pd.DataFrame({
        'ID': test_ids,
        'target': predictions,
        'prob_0': probabilities[:, 0],
        'prob_1': probabilities[:, 1],
        'prob_2': probabilities[:, 2],
        'prob_3': probabilities[:, 3],
        'prob_4': probabilities[:, 4],
        'prob_5': probabilities[:, 5],
        'prob_6': probabilities[:, 6],
        'prob_7': probabilities[:, 7],
        'prob_8': probabilities[:, 8],
        'prob_9': probabilities[:, 9],
        'prob_10': probabilities[:, 10],
        'prob_11': probabilities[:, 11],
        'prob_12': probabilities[:, 12],
        'prob_13': probabilities[:, 13],
        'prob_14': probabilities[:, 14],
        'prob_15': probabilities[:, 15],
        'prob_16': probabilities[:, 16],
        'prob_17': probabilities[:, 17],
        'prob_18': probabilities[:, 18],
        'prob_19': probabilities[:, 19],
        'prob_20': probabilities[:, 20]
    })
    
    detailed_path = f"itransformer_cv_ensemble_detailed.csv"
    results_df.to_csv(detailed_path, index=False)
    print(f"상세 결과 파일 저장: {detailed_path}")
    
    return submission_path, cv_scores, cv_weights

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
        self.train_epochs = 100
        self.patience = 20
        
        # 스케줄러 설정
        self.lradj = 'cosine'  # learning rate adjustment 타입
        
        # 고급 학습 설정
        self.use_class_weights = True      # 클래스 가중치 사용
        self.use_focal_loss = False        # Focal Loss 사용
        self.use_contrastive_loss = False  # Contrastive Loss 사용
        self.use_margin_loss = False       # Margin Loss 사용
        self.use_data_augmentation = False # 데이터 증강 사용
        
        # 손실 함수 가중치
        self.focal_alpha = 1.0
        self.focal_gamma = 2.0
        self.contrastive_weight = 0.001  # 매우 작은 가중치로 조정
        self.margin_weight = 0.01       # 더 작은 가중치로 조정
        
        # 데이터 증강 설정
        self.augmentation_prob = 0.5
        self.timewarp_sigma = 0.2
        self.jitter_sigma = 0.03
        self.window_slice_ratio = 0.9
        self.mixup_alpha = 0.2
        
        # 문제 클래스 설정 (0, 9, 15)
        self.problem_classes = [0, 9, 15]
        self.class_weights = None  # 나중에 계산됨
        
        # CV 앙상블 설정
        self.use_cv = False
        self.n_folds = 5
        self.cv_save_dir = "cv_models"
        self.cv_scores = []
        
        # Attention 메커니즘 설정
        self.use_cosine_attention = False
        
        # Wandb 설정
        self.use_wandb = True
        self.wandb_project = 'itransformer-classification-advanced'
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
        
        # 고급 손실 함수들
        self.focal_loss = None
        self.contrastive_loss = None
        self.margin_loss = None
        
        # 데이터 증강 기법들
        self.timewarp = TimeWarp(sigma=self.config.timewarp_sigma)
        self.jitter = Jitter(sigma=self.config.jitter_sigma)
        self.window_slice = WindowSlicing(reduce_ratio=self.config.window_slice_ratio)
        self.tsmixup = TSMixup(alpha=self.config.mixup_alpha)
        
        # 클래스별 threshold (나중에 계산)
        self.class_thresholds = None
        
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
        
        # class_names가 None이면 기본값 사용
        if class_names is None:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]
        
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
    
    def augment_data(self, X, y):
        """데이터 증강 적용"""
        if not self.config.use_data_augmentation:
            return X, y
        
        augmented_X = []
        augmented_y = []
        
        for i in range(len(X)):
            x = X[i]
            label = y[i]
            
            # 원본 데이터 추가
            augmented_X.append(x)
            augmented_y.append(label)
            
            # 문제 클래스에 대해서만 추가 증강
            if label in self.config.problem_classes and np.random.random() < self.config.augmentation_prob:
                # TimeWarp
                if np.random.random() < 0.3:
                    warped_x = self.timewarp(x)
                    augmented_X.append(warped_x)
                    augmented_y.append(label)
                
                # Jitter
                if np.random.random() < 0.3:
                    jittered_x = self.jitter(x)
                    augmented_X.append(jittered_x)
                    augmented_y.append(label)
                
                # Window Slicing
                if np.random.random() < 0.3:
                    sliced_x = self.window_slice(x)
                    if sliced_x.shape[0] == x.shape[0]:  # 길이가 같을 때만 추가
                        augmented_X.append(sliced_x)
                        augmented_y.append(label)
        
        return np.array(augmented_X), np.array(augmented_y)
    
    
    def prepare_data(self, X_train, y_train, X_val, y_val, X_test):
        """데이터 준비 및 DataLoader 생성"""
        # 데이터 증강 적용 (문제 클래스에 집중)
        if self.config.use_data_augmentation:
            print("데이터 증강 적용 중...")
            X_train_aug, y_train_aug = self.augment_data(X_train, y_train)
            print(f"증강 후 Train 데이터: {X_train_aug.shape}")
        else:
            X_train_aug, y_train_aug = X_train, y_train
        
        # 시계열 데이터로 변환
        X_train_ts = self.create_time_series_data(X_train_aug)
        X_val_ts = self.create_time_series_data(X_val)
        X_test_ts = self.create_time_series_data(X_test)
        
        print(f"시계열 변환 후 형태: {X_train_ts.shape}")
        
        # 설정 업데이트
        self.config.seq_len = X_train_ts.shape[1]
        self.config.enc_in = X_train_ts.shape[2]
        self.config.num_class = len(np.unique(y_train_aug))
        
        print(f"업데이트된 설정 - seq_len: {self.config.seq_len}, enc_in: {self.config.enc_in}")
        
        # 클래스별 샘플링 가중치 계산 (WeightedRandomSampler)
        if self.config.use_class_weights:
            class_counts = np.bincount(y_train_aug)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[y_train_aug]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        # DataLoader 생성
        train_dataset = TensorDataset(
            torch.tensor(X_train_ts, dtype=torch.float32),
            torch.tensor(y_train_aug, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_ts, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test_ts, dtype=torch.float32)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=shuffle,
            sampler=sampler
        )
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def build_model(self):
        """모델 빌드"""
        self.model = iTransformer(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)
        self.scheduler = self._create_scheduler()
        
        # 고급 손실 함수들 초기화
        if self.config.use_focal_loss and self.config.class_weights is not None:
            self.focal_loss = ClassWeightedFocalLoss(
                class_weights=self.config.class_weights,
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma
            ).to(self.device)
            # 클래스 가중치를 GPU로 이동
            self.focal_loss.class_weights = self.focal_loss.class_weights.to(self.device)
        
        if self.config.use_contrastive_loss:
            self.contrastive_loss = SupConLoss(temperature=0.1).to(self.device)
        
        if self.config.use_margin_loss:
            self.margin_loss = PairwiseMarginLoss(
                margin=0.5,
                difficult_pairs=[(0, 9), (0, 15), (9, 15)]
            ).to(self.device)
        
        # 모델 파라미터 수 계산
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"모델 파라미터 수: {total_params:,}")
        print(f"학습 가능한 파라미터 수: {trainable_params:,}")
        
        # 사용 중인 고급 기능들 출력
        print(f"사용 중인 고급 기능:")
        print(f"  - Focal Loss: {self.config.use_focal_loss}")
        print(f"  - Contrastive Loss: {self.config.use_contrastive_loss}")
        print(f"  - Margin Loss: {self.config.use_margin_loss}")
        print(f"  - Data Augmentation: {self.config.use_data_augmentation}")
        print(f"  - Class Weights: {self.config.use_class_weights}")
    
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
        total_focal_loss = 0
        total_contrastive_loss = 0
        total_margin_loss = 0
        
        # 기본 손실 함수 (클래스 가중치 적용)
        if self.config.use_class_weights and self.config.class_weights is not None:
            class_weights_tensor = torch.tensor(self.config.class_weights, dtype=torch.float32).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            criterion = nn.CrossEntropyLoss()
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # iTransformer는 x_mark_enc를 None으로 전달
            outputs = self.model(batch_x, None, None, None)
            
            # 기본 분류 손실
            if self.config.use_focal_loss and self.focal_loss is not None:
                loss = self.focal_loss(outputs, batch_y)
                total_focal_loss += loss.item()
            else:
                loss = criterion(outputs, batch_y)
            
            # Contrastive Loss (특징 추출 필요)
            if self.config.use_contrastive_loss and self.contrastive_loss is not None:
                # 마지막 레이어의 특징 추출 (간단한 방법)
                features = outputs  # 또는 별도 특징 추출
                contrastive_loss = self.contrastive_loss(features, batch_y)
                
                # NaN 체크
                if not torch.isnan(contrastive_loss) and contrastive_loss.item() > 0:
                    loss = loss + self.config.contrastive_weight * contrastive_loss
                    total_contrastive_loss += contrastive_loss.item()
            
            # Margin Loss (특징 추출 필요)
            if self.config.use_margin_loss and self.margin_loss is not None:
                features = outputs  # 또는 별도 특징 추출
                margin_loss = self.margin_loss(features, batch_y)
                
                # NaN 체크
                if not torch.isnan(margin_loss) and margin_loss.item() > 0:
                    loss = loss + self.config.margin_weight * margin_loss
                    total_margin_loss += margin_loss.item() if hasattr(margin_loss, 'item') else margin_loss
            
            total_loss += loss.item()
            
            loss.backward()
            self.optimizer.step()
        
        avg_loss = total_loss / len(train_loader)
        avg_focal = total_focal_loss / len(train_loader) if total_focal_loss > 0 else 0
        avg_contrastive = total_contrastive_loss / len(train_loader) if total_contrastive_loss > 0 else 0
        avg_margin = total_margin_loss / len(train_loader) if total_margin_loss > 0 else 0
        
        return {
            'total_loss': avg_loss,
            'focal_loss': avg_focal,
            'contrastive_loss': avg_contrastive,
            'margin_loss': avg_margin
        }
    
    def validate(self, val_loader):
        """검증"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_trues = []
        
        # 기본 손실 함수 (클래스 가중치 적용)
        if self.config.use_class_weights and self.config.class_weights is not None:
            class_weights_tensor = torch.tensor(self.config.class_weights, dtype=torch.float32).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
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
            train_metrics = self.train_epoch(train_loader)
            train_loss = train_metrics['total_loss']
            
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
            
            # 고급 손실 함수들 로깅
            if train_metrics['focal_loss'] > 0:
                metrics['train_focal_loss'] = train_metrics['focal_loss']
            if train_metrics['contrastive_loss'] > 0:
                metrics['train_contrastive_loss'] = train_metrics['contrastive_loss']
            if train_metrics['margin_loss'] > 0:
                metrics['train_margin_loss'] = train_metrics['margin_loss']
            
            self.log_metrics(metrics)
            
            print(f"Epoch {epoch+1}/{self.config.train_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            if train_metrics['focal_loss'] > 0:
                print(f"  Focal Loss: {train_metrics['focal_loss']:.4f}")
            if train_metrics['contrastive_loss'] > 0:
                print(f"  Contrastive Loss: {train_metrics['contrastive_loss']:.4f}")
            if train_metrics['margin_loss'] > 0:
                print(f"  Margin Loss: {train_metrics['margin_loss']:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro F1: {val_macro_f1:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
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
    
    # 클래스 가중치 계산 (불균형 데이터 처리)
    class_counts = np.bincount(y)
    total_samples = len(y)
    class_weights = total_samples / (len(class_counts) * class_counts)
    print(f"클래스 가중치: {class_weights}")
    
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
    
    return X_train_scaled, X_val_scaled, y_train, y_val, X_test_scaled, test_ids, scaler, class_weights

def main():
    """메인 함수"""
    # 시드 고정
    set_seed(42)
    
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='iTransformer 분류 모델 학습')
    
    # 정규화 방법 선택
    parser.add_argument('--scaling', type=str, default='robust',
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
    parser.add_argument('--e_layers', type=int, default=4,
                       help='인코더 레이어 수 (default: 4)')
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
    parser.add_argument('--epochs', type=int, default=100,
                       help='학습 에포크 수 (default: 100)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience (default: 20)')
    
    # Wandb 설정
    parser.add_argument('--use_wandb', action='store_true', default=True,
                       help='Wandb 로깅 사용 (default: True)')
    parser.add_argument('--no_wandb', dest='use_wandb', action='store_false',
                       help='Wandb 로깅 비활성화')
    parser.add_argument('--wandb_project', type=str, default='itransformer-classification-advanced',
                       help='Wandb 프로젝트 이름 (default: itransformer-classification-advanced)')
    
    # 고급 학습 설정
    parser.add_argument('--use_focal_loss', action='store_true', default=False,
                       help='Focal Loss 사용 (default: False)')
    parser.add_argument('--use_contrastive_loss', action='store_true', default=False,
                       help='Contrastive Loss 사용 (default: False)')
    parser.add_argument('--use_margin_loss', action='store_true', default=False,
                       help='Margin Loss 사용 (default: False)')
    parser.add_argument('--use_data_augmentation', action='store_true', default=False,
                       help='데이터 증강 사용 (default: False)')
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                       help='클래스 가중치 사용 (default: True)')
    
    # 손실 함수 가중치
    parser.add_argument('--focal_alpha', type=float, default=1.0,
                       help='Focal Loss alpha (default: 1.0)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal Loss gamma (default: 2.0)')
    parser.add_argument('--contrastive_weight', type=float, default=0.001,
                       help='Contrastive Loss 가중치 (default: 0.001)')
    parser.add_argument('--margin_weight', type=float, default=0.01,
                       help='Margin Loss 가중치 (default: 0.01)')
    
    # 데이터 증강 설정
    parser.add_argument('--augmentation_prob', type=float, default=0.5,
                       help='데이터 증강 확률 (default: 0.5)')
    parser.add_argument('--timewarp_sigma', type=float, default=0.2,
                       help='TimeWarp 시그마 (default: 0.2)')
    parser.add_argument('--jitter_sigma', type=float, default=0.03,
                       help='Jitter 시그마 (default: 0.03)')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                       help='TSMixup 알파 (default: 0.2)')
    
    # CV 앙상블 설정
    parser.add_argument('--use_cv', action='store_true', default=False,
                       help='CV 앙상블 사용 (default: False)')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='CV 폴드 수 (default: 5)')
    parser.add_argument('--cv_save_dir', type=str, default='cv_models',
                       help='CV 모델 저장 디렉토리 (default: cv_models)')
    
    # Attention 메커니즘 설정
    parser.add_argument('--use_cosine_attention', action='store_true', default=False,
                       help='Cosine Attention 사용 (default: False)')
    
    args = parser.parse_args()
    
    print("iTransformer 분류 모델 학습 시작")
    print("=" * 50)
    print(f"정규화 방법: {args.scaling}")
    print(f"스케일링 범위: {args.scaling_scope}")
    print(f"Attention 메커니즘: {'Cosine Attention' if args.use_cosine_attention else 'Full Attention'}")
    print(f"스케줄러: {args.scheduler}")
    print(f"모델 설정: e_layers={args.e_layers}, d_model={args.d_model}, d_ff={args.d_ff}")
    print(f"학습 설정: lr={args.learning_rate}, batch_size={args.batch_size}, epochs={args.epochs}")
    print(f"Wandb 사용: {args.use_wandb}")
    print("=" * 50)
    
    # 데이터 로드 및 전처리
    X_train, X_val, y_train, y_val, X_test, test_ids, scaler, class_weights = load_and_preprocess_data(args.scaling, args.scaling_scope)
    
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
    
    # 클래스 가중치 설정
    config.class_weights = class_weights
    
    # 고급 학습 설정 적용
    config.use_focal_loss = args.use_focal_loss
    config.use_contrastive_loss = args.use_contrastive_loss
    config.use_margin_loss = args.use_margin_loss
    config.use_data_augmentation = args.use_data_augmentation
    config.use_class_weights = args.use_class_weights
    
    # 손실 함수 가중치 설정
    config.focal_alpha = args.focal_alpha
    config.focal_gamma = args.focal_gamma
    config.contrastive_weight = args.contrastive_weight
    config.margin_weight = args.margin_weight
    
    # 데이터 증강 설정
    config.augmentation_prob = args.augmentation_prob
    config.timewarp_sigma = args.timewarp_sigma
    config.jitter_sigma = args.jitter_sigma
    config.mixup_alpha = args.mixup_alpha
    
    # CV 앙상블 설정
    config.use_cv = args.use_cv
    config.n_folds = args.n_folds
    config.cv_save_dir = args.cv_save_dir
    
    # Attention 메커니즘 설정
    config.use_cosine_attention = args.use_cosine_attention
    
    # CV 앙상블 실행
    if config.use_cv:
        submission_path, cv_scores, cv_weights = run_cv_ensemble(
            X_train, y_train, X_test, test_ids, config, args
        )
        print(f"\nCV 앙상블 완료!")
        print(f"Submission 파일: {submission_path}")
        print(f"CV 점수: {cv_scores}")
        print(f"평균 CV 점수: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        return
    
    # 일반 학습 (CV 사용하지 않는 경우)
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
