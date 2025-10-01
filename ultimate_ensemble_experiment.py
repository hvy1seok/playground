#!/usr/bin/env python3
"""
Ultimate Ensemble Experiment
iTransformer + TabTransformer 통합 앙상블 with 모든 최적화 기법
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import json
import time
import argparse
from datetime import datetime
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Time-Series-Library 경로 추가
sys.path.append('./Time-Series-Library')

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available")

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")

# ----------------------------
# 시드 고정
# ----------------------------
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------
# Loss Functions
# ----------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # [C] tensor

    def forward(self, pred, target):
        ce_loss = nn.functional.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[target]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()

class PairwiseMarginLoss(nn.Module):
    """특정 클래스 쌍 간의 마진 손실"""
    def __init__(self, pairs=[(0, 15), (0, 9), (15, 9)], margin=1.0):
        super().__init__()
        self.pairs = pairs
        self.margin = margin
    
    def forward(self, logits, target):
        loss = 0.0
        count = 0
        
        for c1, c2 in self.pairs:
            mask = (target == c1) | (target == c2)
            if mask.sum() == 0:
                continue
            
            logits_c1 = logits[mask, c1]
            logits_c2 = logits[mask, c2]
            target_masked = target[mask]
            
            # c1이 정답이면 logits_c1 > logits_c2 + margin
            # c2가 정답이면 logits_c2 > logits_c1 + margin
            for i in range(len(target_masked)):
                if target_masked[i] == c1:
                    loss += torch.relu(self.margin - (logits_c1[i] - logits_c2[i]))
                elif target_masked[i] == c2:
                    loss += torch.relu(self.margin - (logits_c2[i] - logits_c1[i]))
                count += 1
        
        return loss / (count + 1e-8)

# ----------------------------
# Data Augmentation
# ----------------------------
class TimeWarp:
    def __init__(self, sigma=0.2):
        self.sigma = sigma
    
    def __call__(self, x):
        length = x.shape[0]
        warp = np.cumsum(np.random.normal(1.0, self.sigma, length))
        warp = warp / warp[-1] * (length - 1)
        warp = np.clip(warp, 0, length - 1)
        indices = np.round(warp).astype(int)
        return x[indices]

class Jitter:
    def __init__(self, sigma=0.03):
        self.sigma = sigma
    
    def __call__(self, x):
        noise = np.random.normal(0, self.sigma, x.shape)
        return x + noise

class TSMixup:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x1, x2):
        lam = np.random.beta(self.alpha, self.alpha)
        return lam * x1 + (1 - lam) * x2

# ----------------------------
# Cosine Attention Layer
# ----------------------------
class CosineAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, L, D = x.shape
        
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Cosine similarity
        Q_norm = Q / (Q.norm(dim=-1, keepdim=True) + 1e-8)
        K_norm = K / (K.norm(dim=-1, keepdim=True) + 1e-8)
        
        attn = torch.matmul(Q_norm, K_norm.transpose(-2, -1))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        
        return out

# ----------------------------
# iTransformer Config (공식 라이브러리용)
# ----------------------------
class iTransformerConfig:
    """iTransformer 공식 라이브러리 설정"""
    def __init__(self, input_dim=52, num_classes=21, d_model=128, n_heads=4, e_layers=4, dropout=0.3):
        # 기본 설정
        self.task_name = 'classification'
        self.seq_len = input_dim  # 시계열 길이
        self.pred_len = 0  # 분류에서는 사용하지 않음
        self.enc_in = 1    # 입력 차원
        self.num_class = num_classes
        
        # 모델 하이퍼파라미터
        self.e_layers = e_layers
        self.d_model = d_model
        self.d_ff = d_model * 2
        self.n_heads = n_heads
        self.factor = 1
        self.dropout = dropout
        self.activation = 'gelu'
        
        # 임베딩 설정
        self.embed = 'timeF'
        self.freq = 'h'
        
        # Cosine Attention 설정
        self.use_cosine_attention = True  # Cosine Attention 사용

# iTransformer 공식 라이브러리 import
try:
    from models.iTransformer import Model as iTransformerOfficial
    ITRANSFORMER_AVAILABLE = True
except ImportError:
    print("⚠️ iTransformer 라이브러리를 찾을 수 없습니다. 간소화 버전을 사용합니다.")
    ITRANSFORMER_AVAILABLE = False
    
    # Fallback: 간소화된 iTransformer
    class iTransformerOfficial(nn.Module):
        def __init__(self, configs):
            super().__init__()
            
            self.embedding = nn.Linear(1, configs.d_model)
            self.pos_encoding = nn.Parameter(torch.randn(1, configs.seq_len, configs.d_model))
            
            self.layers = nn.ModuleList([
                nn.ModuleList([
                    CosineAttention(configs.d_model, configs.n_heads, configs.dropout),
                    nn.LayerNorm(configs.d_model),
                    nn.Sequential(
                        nn.Linear(configs.d_model, configs.d_model * 4),
                        nn.GELU(),
                        nn.Dropout(configs.dropout),
                        nn.Linear(configs.d_model * 4, configs.d_model),
                        nn.Dropout(configs.dropout)
                    ),
                    nn.LayerNorm(configs.d_model)
                ])
                for _ in range(configs.e_layers)
            ])
            
            self.classifier = nn.Sequential(
                nn.Linear(configs.seq_len * configs.d_model, 256),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(128, configs.num_class)
            )
        
        def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
            # x: [B, seq_len, enc_in]
            B = x.shape[0]
            x = self.embedding(x)  # [B, seq_len, d_model]
            x = x + self.pos_encoding
            
            for attn, ln1, ffn, ln2 in self.layers:
                attn_out = attn(x)
                x = ln1(x + attn_out)
                ffn_out = ffn(x)
                x = ln2(x + ffn_out)
            
            x = x.view(B, -1)
            return self.classifier(x)

# ----------------------------
# TabTransformer with Cosine Attention
# ----------------------------
class TabTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, embed_dim=128, num_layers=4, num_heads=4, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        self.layers = nn.ModuleList([
            nn.ModuleList([
                CosineAttention(embed_dim, num_heads, dropout),
                nn.LayerNorm(embed_dim),
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(dropout)
                ),
                nn.LayerNorm(embed_dim)
            ])
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x: [B, input_dim]
        x = self.embedding(x).unsqueeze(1)  # [B, 1, embed_dim]
        
        for attn, ln1, ffn, ln2 in self.layers:
            # Attention
            attn_out = attn(x)
            x = ln1(x + attn_out)
            
            # FFN
            ffn_out = ffn(x)
            x = ln2(x + ffn_out)
        
        x = x.mean(dim=1)  # [B, embed_dim]
        return self.classifier(x)


# ----------------------------
# Threshold Optimization
# ----------------------------
def optimize_thresholds(y_true, y_probs, grid_size=51):
    """클래스별 최적 임계값 탐색"""
    num_classes = y_probs.shape[1]
    best_thresholds = np.ones(num_classes) * 0.5
    best_f1 = -1
    
    # 좌표 강하법
    for iteration in range(3):
        for c in range(num_classes):
            best_score = -1
            best_thresh = 0.5
            
            for t in np.linspace(0.1, 0.9, grid_size):
                temp_thresholds = best_thresholds.copy()
                temp_thresholds[c] = t
                
                # 임계값 기반 예측
                preds = np.argmax(y_probs, axis=1)
                score = f1_score(y_true, preds, average='macro')
                
                if score > best_score:
                    best_score = score
                    best_thresh = t
            
            best_thresholds[c] = best_thresh
            if best_score > best_f1:
                best_f1 = best_score
    
    return best_thresholds, best_f1

# ----------------------------
# Temperature Scaling
# ----------------------------
def optimize_temperature(y_true, logits, init_temp=1.0):
    """온도 보정 최적화"""
    temperature = torch.tensor([init_temp], requires_grad=True)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    target_tensor = torch.tensor(y_true, dtype=torch.long)
    
    def eval():
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(logits_tensor / temperature, target_tensor)
        loss.backward()
        return loss
    
    optimizer.step(eval)
    
    return temperature.item()

# ----------------------------
# Data Augmentation Pipeline
# ----------------------------
def augment_data(X, y, target_classes=[0, 9, 15], augment_ratio=2.0, use_augmentation=True):
    """문제 클래스에 대한 데이터 증강"""
    if not use_augmentation:
        return X, y
    
    X_aug = []
    y_aug = []
    
    time_warp = TimeWarp(sigma=0.2)
    jitter = Jitter(sigma=0.03)
    mixup = TSMixup(alpha=0.2)
    
    for c in target_classes:
        mask = (y == c)
        X_c = X[mask]
        y_c = y[mask]
        
        n_samples = int(len(X_c) * augment_ratio)
        
        for _ in range(n_samples):
            idx = np.random.randint(0, len(X_c))
            x = X_c[idx].copy()
            
            # 랜덤하게 증강 기법 선택
            aug_type = np.random.choice(['warp', 'jitter', 'mixup'])
            
            if aug_type == 'warp':
                x = time_warp(x)
            elif aug_type == 'jitter':
                x = jitter(x)
            elif aug_type == 'mixup' and len(X_c) > 1:
                idx2 = np.random.randint(0, len(X_c))
                x = mixup(x, X_c[idx2])
            
            X_aug.append(x)
            y_aug.append(c)
    
    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)
    
    X_combined = np.vstack([X, X_aug])
    y_combined = np.hstack([y, y_aug])
    
    # 셔플
    indices = np.random.permutation(len(X_combined))
    X_combined = X_combined[indices]
    y_combined = y_combined[indices]
    
    return X_combined, y_combined

# ----------------------------
# Model Training Functions
# ----------------------------
def train_model_itransformer(model, train_loader, val_loader, y_val, device, num_classes, 
                             epochs=100, patience=20, use_focal=True, use_pairwise=True):
    """iTransformer (공식 라이브러리) 학습"""
    
    # Loss functions
    ce_loss = nn.CrossEntropyLoss()
    focal_loss = FocalLoss(gamma=2.0) if use_focal else None
    pairwise_loss = PairwiseMarginLoss(pairs=[(0, 15), (0, 9), (15, 9)], margin=1.0) if use_pairwise else None
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_f1 = 0
    best_state = None
    wait = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            # 공식 라이브러리: x_mark_enc, x_dec, x_mark_dec를 None으로 전달
            logits = model(xb, None, None, None)
            
            # Combined loss
            loss = ce_loss(logits, yb)
            
            if use_focal:
                loss += 0.5 * focal_loss(logits, yb)
            
            if use_pairwise:
                loss += 0.1 * pairwise_loss(logits, yb)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_probs = []
        
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                logits = model(xb, None, None, None)
                probs = torch.softmax(logits, dim=1)
                val_probs.append(probs.cpu().numpy())
                val_preds.append(torch.argmax(probs, dim=1).cpu().numpy())
        
        val_probs = np.vstack(val_probs)
        val_preds = np.hstack(val_preds)
        f1 = f1_score(y_val, val_preds, average='macro')
        
        # 10 epoch마다 또는 개선 시 출력
        if (epoch + 1) % 10 == 0 or f1 > best_f1:
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}, Val F1: {f1:.4f}, Best F1: {best_f1:.4f}, Wait: {wait}/{patience}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch+1}. Best F1: {best_f1:.4f}")
                break
        
        scheduler.step()
    
    model.load_state_dict(best_state)
    return model, best_f1

def train_model(model, train_loader, val_loader, y_val, device, num_classes, 
                epochs=100, patience=20, use_focal=True, use_pairwise=True):
    """TabTransformer 모델 학습"""
    
    # Loss functions
    ce_loss = nn.CrossEntropyLoss()
    focal_loss = FocalLoss(gamma=2.0) if use_focal else None
    pairwise_loss = PairwiseMarginLoss(pairs=[(0, 15), (0, 9), (15, 9)], margin=1.0) if use_pairwise else None
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_f1 = 0
    best_state = None
    wait = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            logits = model(xb)
            
            # Combined loss
            loss = ce_loss(logits, yb)
            
            if use_focal:
                loss += 0.5 * focal_loss(logits, yb)
            
            if use_pairwise:
                loss += 0.1 * pairwise_loss(logits, yb)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_probs = []
        
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1)
                val_probs.append(probs.cpu().numpy())
                val_preds.append(torch.argmax(probs, dim=1).cpu().numpy())
        
        val_probs = np.vstack(val_probs)
        val_preds = np.hstack(val_preds)
        f1 = f1_score(y_val, val_preds, average='macro')
        
        # 10 epoch마다 또는 개선 시 출력
        if (epoch + 1) % 10 == 0 or f1 > best_f1:
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}, Val F1: {f1:.4f}, Best F1: {best_f1:.4f}, Wait: {wait}/{patience}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch+1}. Best F1: {best_f1:.4f}")
                break
        
        scheduler.step()
    
    model.load_state_dict(best_state)
    return model, best_f1


# ----------------------------
# Checkpoint Management
# ----------------------------
def save_checkpoint(fold, seed_results, oof_data, checkpoint_dir='ultimate_checkpoints'):
    """체크포인트 저장"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'fold': fold,
        'seed_results': seed_results,
        'oof_data': oof_data,
        'timestamp': datetime.now().isoformat()
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_fold_{fold}.pkl')
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"  ✅ Checkpoint 저장: {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(fold, checkpoint_dir='ultimate_checkpoints'):
    """체크포인트 로드"""
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_fold_{fold}.pkl')
    
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"  ✅ Checkpoint 로드: {checkpoint_path}")
        return checkpoint
    return None

def checkpoint_exists(fold, checkpoint_dir='ultimate_checkpoints'):
    """체크포인트 존재 여부 확인"""
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_fold_{fold}.pkl')
    return os.path.exists(checkpoint_path)

# ----------------------------
# Main Experiment
# ----------------------------
def run_ultimate_experiment(args):
    """최종 통합 실험"""
    # Seeds 및 설정
    SEEDS = [819, 42, 24]
    N_FOLDS = 5
    CHECKPOINT_DIR = 'ultimate_checkpoints'
    
    print("="*80)
    print("ULTIMATE ENSEMBLE EXPERIMENT")
    print("iTransformer (공식 라이브러리) + TabTransformer (Multi-head Cosine)")
    print("="*80)
    print(f"\n설정:")
    print(f"  - 메타 모델: {args.meta_model}")
    print(f"  - 데이터 증강: {'사용' if args.use_augmentation else '미사용'}")
    print(f"  - Seeds: {SEEDS}")
    print(f"  - Folds: {N_FOLDS}")
    print(f"  - iTransformer: 공식 라이브러리 (Cosine Attention)")
    print(f"  - TabTransformer: Multi-head Cosine Attention")
    print("="*80)
    
    # 데이터 로드
    train_df = pd.read_csv("./datasests/train.csv")
    test_df = pd.read_csv("./datasests/test.csv")
    
    X = train_df.drop(columns=["ID", "target"]).values
    y = train_df["target"].values
    X_test = test_df.drop(columns=["ID"]).values
    test_ids = test_df["ID"].values
    
    num_classes = len(np.unique(y))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n데이터 형태: {X.shape}")
    print(f"테스트 데이터: {X_test.shape}")
    print(f"클래스 수: {num_classes}")
    print(f"Device: {device}")
    
    # Seeds
    SEEDS = [819, 42, 24]
    N_FOLDS = 5
    
    # OOF 저장
    oof_itransformer = np.zeros((len(X), num_classes))
    oof_tabtransformer = np.zeros((len(X), num_classes))
    oof_labels = np.zeros(len(X), dtype=int)
    
    # Test 저장
    test_itransformer_logits = []
    test_tabtransformer_logits = []
    
    # Results
    fold_results = []
    
    # 5-Fold CV
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=123)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f"\n{'='*80}")
        print(f"FOLD {fold}/{N_FOLDS}")
        print(f"{'='*80}")
        
        # 체크포인트 확인
        if checkpoint_exists(fold, CHECKPOINT_DIR):
            print(f"⏩ Fold {fold} 체크포인트 발견! 로드 중...")
            checkpoint = load_checkpoint(fold, CHECKPOINT_DIR)
            
            # 체크포인트에서 데이터 복원
            val_probs_itrans = checkpoint['oof_data']['val_probs_itrans']
            val_probs_tabtrans = checkpoint['oof_data']['val_probs_tabtrans']
            test_logits_itrans_fold = checkpoint['seed_results']['test_logits_itrans']
            test_logits_tabtrans_fold = checkpoint['seed_results']['test_logits_tabtrans']
            
            # OOF 저장
            oof_itransformer[val_idx] = val_probs_itrans
            oof_tabtransformer[val_idx] = val_probs_tabtrans
            oof_labels[val_idx] = y[val_idx]
            
            # Test logits 저장
            test_itransformer_logits.append(test_logits_itrans_fold)
            test_tabtransformer_logits.append(test_logits_tabtrans_fold)
            
            # Fold 결과
            fold_results.append({
                'fold': fold,
                'itrans_f1': f1_score(y[val_idx], np.argmax(val_probs_itrans, axis=1), average='macro'),
                'tabtrans_f1': f1_score(y[val_idx], np.argmax(val_probs_tabtrans, axis=1), average='macro')
            })
            
            print(f"Fold {fold} 결과 (체크포인트):")
            print(f"  iTransformer OOF F1: {fold_results[-1]['itrans_f1']:.4f}")
            print(f"  TabTransformer OOF F1: {fold_results[-1]['tabtrans_f1']:.4f}")
            
            continue  # 다음 fold로
        
        # 체크포인트가 없으면 학습 진행
        print(f"🔄 Fold {fold} 학습 시작...")
        
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        oof_labels[val_idx] = y_val_fold
        
        # Seed 앙상블
        val_probs_itrans_seeds = []
        val_probs_tabtrans_seeds = []
        test_logits_itrans_seeds = []
        test_logits_tabtrans_seeds = []
        
        for seed_idx, seed in enumerate(SEEDS):
            print(f"\n--- Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ---")
            set_seed(seed)
            
            # 데이터 증강
            if args.use_augmentation:
                print("데이터 증강...")
            X_train_aug, y_train_aug = augment_data(
                X_train_fold, y_train_fold, 
                target_classes=[0, 9, 15], 
                augment_ratio=1.5,
                use_augmentation=args.use_augmentation
            )
            
            # RobustScaler (column-wise)
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_aug)
            X_val_scaled = scaler.transform(X_val_fold)
            X_test_scaled = scaler.transform(X_test)
            
            # DataLoader
            train_dataset = TensorDataset(
                torch.tensor(X_train_scaled, dtype=torch.float32),
                torch.tensor(y_train_aug, dtype=torch.long)
            )
            val_dataset = TensorDataset(
                torch.tensor(X_val_scaled, dtype=torch.float32),
                torch.tensor(y_val_fold, dtype=torch.long)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
            
            # ========== iTransformer (공식 라이브러리) ==========
            print("\n[iTransformer] 학습 중...")
            
            # 공식 라이브러리용 Config 생성
            itrans_config = iTransformerConfig(
                input_dim=X.shape[1],
                num_classes=num_classes,
                d_model=128,
                n_heads=4,
                e_layers=4,
                dropout=0.3
            )
            
            # 공식 라이브러리 모델 생성
            itrans_model = iTransformerOfficial(itrans_config).to(device)
            
            # 공식 라이브러리 입력 형태로 변환: [B, input_dim] -> [B, seq_len, enc_in]
            # train_loader와 val_loader를 다시 만들어야 함
            X_train_ts = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
            X_val_ts = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)
            
            train_loader_ts = DataLoader(
                TensorDataset(torch.tensor(X_train_ts, dtype=torch.float32),
                             torch.tensor(y_train_fold, dtype=torch.long)),
                batch_size=64, shuffle=True
            )
            val_loader_ts = DataLoader(
                TensorDataset(torch.tensor(X_val_ts, dtype=torch.float32),
                             torch.tensor(y_val_fold, dtype=torch.long)),
                batch_size=64, shuffle=False
            )
            
            itrans_model, itrans_f1 = train_model_itransformer(
                itrans_model, train_loader_ts, val_loader_ts, y_val_fold, device, num_classes,
                epochs=100, patience=20, use_focal=True, use_pairwise=True
            )
            print(f"iTransformer F1: {itrans_f1:.4f}")
            
            # ========== TabTransformer ==========
            print("\n[TabTransformer] 학습 중...")
            tabtrans_model = TabTransformer(
                input_dim=X.shape[1],
                num_classes=num_classes,
                embed_dim=128,
                num_layers=4,
                num_heads=4,
                dropout=0.3
            ).to(device)
            
            tabtrans_model, tabtrans_f1 = train_model(
                tabtrans_model, train_loader, val_loader, y_val_fold, device, num_classes,
                epochs=100, patience=20, use_focal=True, use_pairwise=True
            )
            print(f"TabTransformer F1: {tabtrans_f1:.4f}")
            
            # Validation 예측
            itrans_model.eval()
            tabtrans_model.eval()
            
            with torch.no_grad():
                val_logits_itrans = []
                val_logits_tabtrans = []
                
                # iTransformer: 시계열 형태로 예측
                for xb, _ in val_loader_ts:
                    xb = xb.to(device)
                    val_logits_itrans.append(itrans_model(xb, None, None, None).cpu().numpy())
                
                # TabTransformer: 원래 형태로 예측
                for xb, _ in val_loader:
                    xb = xb.to(device)
                    val_logits_tabtrans.append(tabtrans_model(xb).cpu().numpy())
                
                val_logits_itrans = np.vstack(val_logits_itrans)
                val_logits_tabtrans = np.vstack(val_logits_tabtrans)
                
                val_probs_itrans_seeds.append(torch.softmax(torch.tensor(val_logits_itrans), dim=1).numpy())
                val_probs_tabtrans_seeds.append(torch.softmax(torch.tensor(val_logits_tabtrans), dim=1).numpy())
            
            # Test 예측
            # iTransformer용: 시계열 형태
            X_test_ts = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
            test_loader_ts = DataLoader(
                TensorDataset(torch.tensor(X_test_ts, dtype=torch.float32)),
                batch_size=64, shuffle=False
            )
            
            # TabTransformer용: 원래 형태
            test_loader = DataLoader(
                TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32)),
                batch_size=64, shuffle=False
            )
            
            with torch.no_grad():
                test_logits_itrans = []
                test_logits_tabtrans = []
                
                # iTransformer: 시계열 형태로 예측
                for xb, in test_loader_ts:
                    xb = xb.to(device)
                    test_logits_itrans.append(itrans_model(xb, None, None, None).cpu().numpy())
                
                # TabTransformer: 원래 형태로 예측
                for xb, in test_loader:
                    xb = xb.to(device)
                    test_logits_tabtrans.append(tabtrans_model(xb).cpu().numpy())
                
                test_logits_itrans = np.vstack(test_logits_itrans)
                test_logits_tabtrans = np.vstack(test_logits_tabtrans)
                
                test_logits_itrans_seeds.append(test_logits_itrans)
                test_logits_tabtrans_seeds.append(test_logits_tabtrans)
        
        # Seed 앙상블 (평균)
        print("\n=== Seed 앙상블 ===")
        val_probs_itrans = np.mean(val_probs_itrans_seeds, axis=0)
        val_probs_tabtrans = np.mean(val_probs_tabtrans_seeds, axis=0)
        
        test_logits_itrans_fold = np.mean(test_logits_itrans_seeds, axis=0)
        test_logits_tabtrans_fold = np.mean(test_logits_tabtrans_seeds, axis=0)
        
        # OOF 저장
        oof_itransformer[val_idx] = val_probs_itrans
        oof_tabtransformer[val_idx] = val_probs_tabtrans
        
        # Test logits 저장
        test_itransformer_logits.append(test_logits_itrans_fold)
        test_tabtransformer_logits.append(test_logits_tabtrans_fold)
        
        # Fold 결과
        fold_results.append({
            'fold': fold,
            'itrans_f1': f1_score(y_val_fold, np.argmax(val_probs_itrans, axis=1), average='macro'),
            'tabtrans_f1': f1_score(y_val_fold, np.argmax(val_probs_tabtrans, axis=1), average='macro')
        })
        
        print(f"\nFold {fold} 결과:")
        print(f"  iTransformer OOF F1: {fold_results[-1]['itrans_f1']:.4f}")
        print(f"  TabTransformer OOF F1: {fold_results[-1]['tabtrans_f1']:.4f}")
        
        # 체크포인트 저장
        checkpoint_data = {
            'val_probs_itrans': val_probs_itrans,
            'val_probs_tabtrans': val_probs_tabtrans,
            'test_logits_itrans': test_logits_itrans_fold,
            'test_logits_tabtrans': test_logits_tabtrans_fold
        }
        
        seed_results_data = {
            'test_logits_itrans': test_logits_itrans_fold,
            'test_logits_tabtrans': test_logits_tabtrans_fold
        }
        
        save_checkpoint(fold, seed_results_data, checkpoint_data, CHECKPOINT_DIR)
    
    # ========== OOF 평가 ==========
    print(f"\n{'='*80}")
    print("OOF 평가")
    print(f"{'='*80}")
    
    oof_pred_itrans = np.argmax(oof_itransformer, axis=1)
    oof_pred_tabtrans = np.argmax(oof_tabtransformer, axis=1)
    
    oof_f1_itrans = f1_score(oof_labels, oof_pred_itrans, average='macro')
    oof_f1_tabtrans = f1_score(oof_labels, oof_pred_tabtrans, average='macro')
    
    print(f"iTransformer OOF Macro-F1: {oof_f1_itrans:.4f}")
    print(f"TabTransformer OOF Macro-F1: {oof_f1_tabtrans:.4f}")
    
    # OOF Confusion Matrix
    print("\n=== iTransformer OOF Confusion Matrix ===")
    cm_itrans = confusion_matrix(oof_labels, oof_pred_itrans)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_itrans, annot=True, fmt='d', cmap='Blues')
    plt.title('iTransformer OOF Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('oof_confusion_matrix_itransformer.png', dpi=150)
    print("저장: oof_confusion_matrix_itransformer.png")
    
    print("\n=== TabTransformer OOF Confusion Matrix ===")
    cm_tabtrans = confusion_matrix(oof_labels, oof_pred_tabtrans)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_tabtrans, annot=True, fmt='d', cmap='Greens')
    plt.title('TabTransformer OOF Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('oof_confusion_matrix_tabtransformer.png', dpi=150)
    print("저장: oof_confusion_matrix_tabtransformer.png")
    
    # ========== Cross-Model 앙상블 ==========
    print(f"\n{'='*80}")
    print("Cross-Model 앙상블")
    print(f"{'='*80}")
    
    # OOF F1 기반 가중치
    weight_itrans = oof_f1_itrans / (oof_f1_itrans + oof_f1_tabtrans)
    weight_tabtrans = oof_f1_tabtrans / (oof_f1_itrans + oof_f1_tabtrans)
    
    print(f"Simple Weighted: iTransformer={weight_itrans:.3f}, TabTransformer={weight_tabtrans:.3f}")
    
    # Test logits 5-Fold 앙상블
    test_logits_itrans_final = np.mean(test_itransformer_logits, axis=0)
    test_logits_tabtrans_final = np.mean(test_tabtransformer_logits, axis=0)
    
    # Simple weighted ensemble
    test_logits_simple = (weight_itrans * test_logits_itrans_final + 
                          weight_tabtrans * test_logits_tabtrans_final)
    
    test_probs_simple = torch.softmax(torch.tensor(test_logits_simple), dim=1).numpy()
    
    # ========== 메타 스태킹 (OOF Leakage 방지) ==========
    print(f"\n{'='*80}")
    print("메타 스태킹 (OOF Leakage 방지)")
    print(f"{'='*80}")
    
    # OOF Leakage 방지를 위한 Hold-out 분할
    print("OOF Leakage 방지: Hold-out validation 사용")
    X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
        np.arange(len(oof_labels)), oof_labels, 
        test_size=0.2, random_state=42, stratify=oof_labels
    )
    
    print(f"메타 학습 데이터: {len(X_meta_train)}개")
    print(f"메타 검증 데이터: {len(X_meta_val)}개")
    
    # OOF 확률을 메타 특징으로 사용
    # Shape: [N_train, 2*C] (iTransformer + TabTransformer)
    oof_meta_features = np.hstack([oof_itransformer, oof_tabtransformer])
    
    # 추가 메타 특징
    # 1. 엔트로피
    entropy_itrans = -np.sum(oof_itransformer * np.log(oof_itransformer + 1e-8), axis=1, keepdims=True)
    entropy_tabtrans = -np.sum(oof_tabtransformer * np.log(oof_tabtransformer + 1e-8), axis=1, keepdims=True)
    
    # 2. Top-1 confidence
    top1_conf_itrans = np.max(oof_itransformer, axis=1, keepdims=True)
    top1_conf_tabtrans = np.max(oof_tabtransformer, axis=1, keepdims=True)
    
    # 3. Margin (Top-1 - Top-2)
    sorted_itrans = np.sort(oof_itransformer, axis=1)
    sorted_tabtrans = np.sort(oof_tabtransformer, axis=1)
    margin_itrans = (sorted_itrans[:, -1] - sorted_itrans[:, -2]).reshape(-1, 1)
    margin_tabtrans = (sorted_tabtrans[:, -1] - sorted_tabtrans[:, -2]).reshape(-1, 1)
    
    # 4. Agreement (두 모델이 같은 클래스를 예측하는지)
    pred_itrans = np.argmax(oof_itransformer, axis=1)
    pred_tabtrans = np.argmax(oof_tabtransformer, axis=1)
    agreement = (pred_itrans == pred_tabtrans).astype(float).reshape(-1, 1)
    
    # 모든 메타 특징 결합
    oof_meta_features_extended = np.hstack([
        oof_meta_features,
        entropy_itrans, entropy_tabtrans,
        top1_conf_itrans, top1_conf_tabtrans,
        margin_itrans, margin_tabtrans,
        agreement
    ])
    
    # Hold-out 분할 적용
    X_meta_train_features = oof_meta_features_extended[X_meta_train]
    y_meta_train_labels = oof_labels[X_meta_train]
    X_meta_val_features = oof_meta_features_extended[X_meta_val]
    y_meta_val_labels = oof_labels[X_meta_val]
    
    print(f"메타 특징 형태: {oof_meta_features_extended.shape}")
    print(f"  - 기본 확률: {num_classes * 2}개")
    print(f"  - 엔트로피: 2개")
    print(f"  - Top-1 Confidence: 2개")
    print(f"  - Margin: 2개")
    print(f"  - Agreement: 1개")
    print(f"  - 총: {oof_meta_features_extended.shape[1]}개")
    
    # 메타 모델 학습 (Hold-out train 데이터만 사용)
    print(f"\n메타 모델 학습 중... (모델: {args.meta_model})")
    
    if args.meta_model == 'logistic':
        meta_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='lbfgs',
            multi_class='multinomial',
            C=1.0,
            class_weight='balanced'
        )
        meta_model.fit(X_meta_train_features, y_meta_train_labels)
        
    elif args.meta_model == 'lightgbm':
        if not LIGHTGBM_AVAILABLE:
            print("⚠️ LightGBM이 설치되지 않았습니다. Logistic Regression으로 대체합니다.")
            meta_model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                solver='lbfgs',
                multi_class='multinomial',
                C=1.0,
                class_weight='balanced'
            )
            meta_model.fit(X_meta_train_features, y_meta_train_labels)
        else:
            # LightGBM 파라미터
            params = {
                'objective': 'multiclass',
                'num_class': num_classes,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': 42,
                'n_jobs': -1
            }
            
            # 클래스 가중치 계산
            class_counts = np.bincount(y_meta_train_labels)
            class_weights = len(y_meta_train_labels) / (num_classes * class_counts)
            sample_weights = class_weights[y_meta_train_labels]
            
            train_data = lgb.Dataset(X_meta_train_features, label=y_meta_train_labels, weight=sample_weights)
            
            meta_model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
            )
    
    elif args.meta_model == 'xgboost':
        if not XGBOOST_AVAILABLE:
            print("⚠️ XGBoost가 설치되지 않았습니다. Logistic Regression으로 대체합니다.")
            meta_model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                solver='lbfgs',
                multi_class='multinomial',
                C=1.0,
                class_weight='balanced'
            )
            meta_model.fit(X_meta_train_features, y_meta_train_labels)
        else:
            # 클래스 가중치 계산
            class_counts = np.bincount(y_meta_train_labels)
            class_weights = len(y_meta_train_labels) / (num_classes * class_counts)
            sample_weights = class_weights[y_meta_train_labels]
            
            dtrain = xgb.DMatrix(X_meta_train_features, label=y_meta_train_labels, weight=sample_weights)
            
            params = {
                'objective': 'multi:softprob',
                'num_class': num_classes,
                'eval_metric': 'mlogloss',
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'seed': 42,
                'tree_method': 'hist'
            }
            
            meta_model = xgb.train(
                params,
                dtrain,
                num_boost_round=200,
                evals=[(dtrain, 'train')],
                early_stopping_rounds=50,
                verbose_eval=50
            )
    
    else:
        raise ValueError(f"Unknown meta_model: {args.meta_model}")
    
    # OOF 메타 예측 (Hold-out validation 데이터로 평가)
    if args.meta_model == 'logistic':
        oof_meta_preds = meta_model.predict(X_meta_val_features)
        oof_meta_probs = meta_model.predict_proba(X_meta_val_features)
    elif args.meta_model == 'lightgbm' and LIGHTGBM_AVAILABLE:
        oof_meta_probs = meta_model.predict(X_meta_val_features)
        oof_meta_preds = np.argmax(oof_meta_probs, axis=1)
    elif args.meta_model == 'xgboost' and XGBOOST_AVAILABLE:
        doof = xgb.DMatrix(X_meta_val_features)
        oof_meta_probs = meta_model.predict(doof)
        oof_meta_preds = np.argmax(oof_meta_probs, axis=1)
    else:
        oof_meta_preds = meta_model.predict(X_meta_val_features)
        oof_meta_probs = meta_model.predict_proba(X_meta_val_features)
    
    oof_meta_f1 = f1_score(y_meta_val_labels, oof_meta_preds, average='macro')
    
    # Simple Weighted 비교 (Hold-out validation 데이터로)
    simple_weighted_val = weight_itrans * oof_itransformer[X_meta_val] + weight_tabtrans * oof_tabtransformer[X_meta_val]
    simple_weighted_preds = np.argmax(simple_weighted_val, axis=1)
    simple_weighted_f1_val = f1_score(y_meta_val_labels, simple_weighted_preds, average='macro')
    
    print(f"메타 모델 Hold-out F1: {oof_meta_f1:.4f}")
    print(f"  vs Simple Weighted Hold-out F1: {simple_weighted_f1_val:.4f}")
    
    # Test 데이터 메타 특징 생성
    test_probs_itrans = torch.softmax(torch.tensor(test_logits_itrans_final), dim=1).numpy()
    test_probs_tabtrans = torch.softmax(torch.tensor(test_logits_tabtrans_final), dim=1).numpy()
    
    test_meta_features = np.hstack([test_probs_itrans, test_probs_tabtrans])
    
    # Test 추가 메타 특징
    entropy_itrans_test = -np.sum(test_probs_itrans * np.log(test_probs_itrans + 1e-8), axis=1, keepdims=True)
    entropy_tabtrans_test = -np.sum(test_probs_tabtrans * np.log(test_probs_tabtrans + 1e-8), axis=1, keepdims=True)
    
    top1_conf_itrans_test = np.max(test_probs_itrans, axis=1, keepdims=True)
    top1_conf_tabtrans_test = np.max(test_probs_tabtrans, axis=1, keepdims=True)
    
    sorted_itrans_test = np.sort(test_probs_itrans, axis=1)
    sorted_tabtrans_test = np.sort(test_probs_tabtrans, axis=1)
    margin_itrans_test = (sorted_itrans_test[:, -1] - sorted_itrans_test[:, -2]).reshape(-1, 1)
    margin_tabtrans_test = (sorted_tabtrans_test[:, -1] - sorted_tabtrans_test[:, -2]).reshape(-1, 1)
    
    pred_itrans_test = np.argmax(test_probs_itrans, axis=1)
    pred_tabtrans_test = np.argmax(test_probs_tabtrans, axis=1)
    agreement_test = (pred_itrans_test == pred_tabtrans_test).astype(float).reshape(-1, 1)
    
    test_meta_features_extended = np.hstack([
        test_meta_features,
        entropy_itrans_test, entropy_tabtrans_test,
        top1_conf_itrans_test, top1_conf_tabtrans_test,
        margin_itrans_test, margin_tabtrans_test,
        agreement_test
    ])
    
    # 메타 모델로 Test 예측
    if args.meta_model == 'logistic':
        test_probs_meta = meta_model.predict_proba(test_meta_features_extended)
    elif args.meta_model == 'lightgbm' and LIGHTGBM_AVAILABLE:
        test_probs_meta = meta_model.predict(test_meta_features_extended)
    elif args.meta_model == 'xgboost' and XGBOOST_AVAILABLE:
        dtest = xgb.DMatrix(test_meta_features_extended)
        test_probs_meta = meta_model.predict(dtest)
    else:
        test_probs_meta = meta_model.predict_proba(test_meta_features_extended)
    
    print(f"\n메타 스태킹 완료!")
    print(f"  - Hold-out F1 (iTransformer): {f1_score(y_meta_val_labels, np.argmax(oof_itransformer[X_meta_val], axis=1), average='macro'):.4f}")
    print(f"  - Hold-out F1 (TabTransformer): {f1_score(y_meta_val_labels, np.argmax(oof_tabtransformer[X_meta_val], axis=1), average='macro'):.4f}")
    print(f"  - Hold-out F1 (Simple Weighted): {simple_weighted_f1_val:.4f}")
    print(f"  - Hold-out F1 (Meta Stacking): {oof_meta_f1:.4f}")
    
    # 메타 모델과 Simple Weighted 중 더 나은 것 선택
    if oof_meta_f1 > simple_weighted_f1_val:
        print(f"\n✅ 메타 스태킹이 더 우수합니다! (Δ = +{oof_meta_f1 - simple_weighted_f1_val:.4f})")
        test_probs_ensemble = test_probs_meta
        use_meta = True
    else:
        print(f"\n⚠️ Simple Weighted가 더 우수합니다. (Δ = -{simple_weighted_f1_val - oof_meta_f1:.4f})")
        test_probs_ensemble = test_probs_simple
        use_meta = False
    
    test_preds_ensemble = np.argmax(test_probs_ensemble, axis=1)
    
    # OOF 예측 (전체 데이터에 대해)
    oof_pred_ensemble = np.argmax(weight_itrans * oof_itransformer + weight_tabtrans * oof_tabtransformer, axis=1)
    
    # ========== 최종 예측 ==========
    print(f"\n{'='*80}")
    print("최종 예측")
    print(f"{'='*80}")
    
    # 메타 스태킹 결과를 최종 예측으로 사용
    test_probs_final = test_probs_ensemble
    test_preds_final = test_preds_ensemble
    
    print("✅ 최종 예측 완료!")
    
    # ========== 결과 저장 ==========
    print(f"\n{'='*80}")
    print("결과 저장")
    print(f"{'='*80}")
    
    # 제출 파일
    submission = pd.DataFrame({
        "ID": test_ids,
        "target": test_preds_final
    })
    submission.to_csv("ultimate_ensemble_submission.csv", index=False)
    print("✅ ultimate_ensemble_submission.csv")
    
    # 상세 파일
    detailed = pd.DataFrame({
        "ID": test_ids,
        "target": test_preds_final,
        **{f"prob_{i}": test_probs_final[:, i] for i in range(num_classes)}
    })
    detailed.to_csv("ultimate_ensemble_detailed.csv", index=False)
    print("✅ ultimate_ensemble_detailed.csv")
    
    # OOF 결과 저장
    oof_df = pd.DataFrame({
        "ID": train_df["ID"],
        "true_label": oof_labels,
        "pred_label_itrans": oof_pred_itrans,
        "pred_label_tabtrans": oof_pred_tabtrans,
        "pred_label_ensemble": oof_pred_ensemble,
        **{f"prob_itrans_{i}": oof_itransformer[:, i] for i in range(num_classes)},
        **{f"prob_tabtrans_{i}": oof_tabtransformer[:, i] for i in range(num_classes)},
        **{f"prob_meta_{i}": oof_itransformer[:, i] for i in range(num_classes)}  # 메타 모델은 Hold-out에서만 평가
    })
    oof_df.to_csv("ultimate_ensemble_oof.csv", index=False)
    print("✅ ultimate_ensemble_oof.csv")
    
    # 메타 모델 정보 저장
    if use_meta:
        meta_info = {
            "method": f"{args.meta_model.upper()} Meta-Stacking",
            "use_augmentation": args.use_augmentation,
            "oof_f1_itransformer": float(oof_f1_itrans),
            "oof_f1_tabtransformer": float(oof_f1_tabtrans),
            "oof_f1_simple_weighted": float(simple_weighted_f1_val),
            "oof_f1_meta_stacking": float(oof_meta_f1),
            "improvement": float(oof_meta_f1 - simple_weighted_f1_val),
            "meta_features": {
                "base_probs": num_classes * 2,
                "entropy": 2,
                "top1_confidence": 2,
                "margin": 2,
                "agreement": 1,
                "total": oof_meta_features_extended.shape[1]
            }
        }
        
        with open("ultimate_ensemble_meta_info.json", "w") as f:
            json.dump(meta_info, f, indent=2)
        print("✅ ultimate_ensemble_meta_info.json")
    
    # 예측 분포
    print(f"\n최종 예측 분포:")
    pred_counts = np.bincount(test_preds_final, minlength=num_classes)
    for i, count in enumerate(pred_counts):
        print(f"  Class {i}: {count} ({count/len(test_preds_final)*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print("실험 완료!")
    print(f"{'='*80}")

def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='Ultimate Ensemble Experiment')
    
    parser.add_argument(
        '--meta_model',
        type=str,
        default='logistic',
        choices=['logistic', 'lightgbm', 'xgboost'],
        help='메타 스태킹 모델 선택 (default: logistic)'
    )
    
    parser.add_argument(
        '--use_augmentation',
        action='store_true',
        help='데이터 증강 사용'
    )
    
    parser.add_argument(
        '--no_augmentation',
        dest='use_augmentation',
        action='store_false',
        help='데이터 증강 사용 안 함'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='체크포인트에서 재개 (default: False)'
    )
    
    parser.add_argument(
        '--clear_checkpoints',
        action='store_true',
        help='기존 체크포인트 삭제 후 새로 시작'
    )
    
    parser.set_defaults(use_augmentation=True)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seed(123)
    
    # 체크포인트 관리
    if args.clear_checkpoints:
        import shutil
        if os.path.exists('ultimate_checkpoints'):
            shutil.rmtree('ultimate_checkpoints')
            print("🗑️ 기존 체크포인트 삭제 완료\n")
    
    print("\n" + "="*80)
    print("실행 설정")
    print("="*80)
    print(f"메타 모델: {args.meta_model}")
    print(f"데이터 증강: {'사용' if args.use_augmentation else '미사용'}")
    print(f"체크포인트: {'재개' if args.resume else '처음부터'}")
    
    if args.meta_model == 'lightgbm' and not LIGHTGBM_AVAILABLE:
        print("\n⚠️ 경고: LightGBM이 설치되지 않아 Logistic Regression으로 대체됩니다.")
    
    if args.meta_model == 'xgboost' and not XGBOOST_AVAILABLE:
        print("\n⚠️ 경고: XGBoost가 설치되지 않아 Logistic Regression으로 대체됩니다.")
    
    if args.resume and os.path.exists('ultimate_checkpoints'):
        checkpoints = [f for f in os.listdir('ultimate_checkpoints') if f.endswith('.pkl')]
        print(f"\n✅ {len(checkpoints)}개의 체크포인트 발견:")
        for cp in sorted(checkpoints):
            print(f"   - {cp}")
    
    print("="*80 + "\n")
    
    run_ultimate_experiment(args)

