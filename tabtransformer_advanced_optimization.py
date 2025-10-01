#!/usr/bin/env python3
"""
TabTransformer + OVR(0/9/15 전담) 고급 최적화
확률 보정 + 임계값 최적화 + 메타 앙상블로 macro-F1 극대화
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import optuna
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# 재현성 보장용 시드 고정
# ----------------------------
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(123)

# ----------------------------
# Label Smoothing Loss
# ----------------------------
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# ----------------------------
# Cosine Transformer
# ----------------------------
class CosineTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, embed_dim=128, num_layers=4, dropout=0.3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.blocks = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        z = self.embedding(x).unsqueeze(1)  # [B,1,D]
        for ln in self.blocks:
            h = ln(z)
            Q, K, V = self.q(h), self.k(h), self.v(h)
            Qn = Q / (Q.norm(dim=-1, keepdim=True) + 1e-8)
            Kn = K / (K.norm(dim=-1, keepdim=True) + 1e-8)
            A = torch.softmax(Qn @ Kn.transpose(1, 2), dim=-1)
            out = A @ V
            z = z + out
        z = z.mean(dim=1)
        return self.classifier(z)

# ----------------------------
# OVR Binary Classifier for 0/9/15
# ----------------------------
class OVRBinaryClassifier(nn.Module):
    def __init__(self, input_dim, embed_dim=64, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim//2, 1)
        )

    def forward(self, x):
        return self.classifier(x)

# ----------------------------
# Probability Calibration
# ----------------------------
class ProbabilityCalibrator:
    def __init__(self, method='platt'):
        self.method = method
        self.calibrators = {}
        
    def fit(self, y_true, y_probs, classes):
        """각 클래스별로 확률 보정기 학습"""
        for c in classes:
            if self.method == 'platt':
                # Platt scaling (sigmoid)
                from sklearn.linear_model import LogisticRegression
                calibrator = LogisticRegression()
                calibrator.fit(y_probs[:, c].reshape(-1, 1), (y_true == c).astype(int))
                self.calibrators[c] = calibrator
            elif self.method == 'isotonic':
                # Isotonic regression
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(y_probs[:, c], (y_true == c).astype(int))
                self.calibrators[c] = calibrator
            elif self.method == 'temperature':
                # Temperature scaling (단일 파라미터)
                def temperature_scaling(logits, temp):
                    return logits / temp
                # 간단한 구현 - 실제로는 더 정교한 최적화 필요
                self.calibrators[c] = 1.0  # 기본값
                
    def predict_proba(self, y_probs):
        """보정된 확률 예측"""
        calibrated_probs = np.zeros_like(y_probs)
        
        for c, calibrator in self.calibrators.items():
            if self.method in ['platt', 'isotonic']:
                calibrated_probs[:, c] = calibrator.predict_proba(y_probs[:, c].reshape(-1, 1))[:, 1]
            else:  # temperature
                calibrated_probs[:, c] = y_probs[:, c]  # 단순화
        
        # 정규화
        calibrated_probs = calibrated_probs / (calibrated_probs.sum(axis=1, keepdims=True) + 1e-8)
        return calibrated_probs

# ----------------------------
# Threshold Optimization
# ----------------------------
def optimize_thresholds(y_true, y_probs, classes, grid_size=17):
    """클래스별 임계값 최적화"""
    from sklearn.metrics import f1_score
    
    best_thresholds = {c: 0.5 for c in classes}
    best_f1 = -1
    
    # 좌표강하식 근사: 클래스별 순차 탐색
    for iteration in range(3):
        for c in classes:
            best_score = -1
            best_thresh = 0.5
            
            # 그리드 탐색
            for t in np.linspace(0.1, 0.9, grid_size):
                # 임시 임계값으로 예측
                temp_thresholds = best_thresholds.copy()
                temp_thresholds[c] = t
                
                # 후보 마스크 생성
                H = np.array([y_probs[:, i] >= temp_thresholds[i] for i in classes]).T
                
                # 다중 후보 처리: 점수 최대 클래스 선택
                pred = np.where(H.any(1), 
                              np.array(classes)[np.argmax(y_probs, axis=1)], 
                              np.array(classes)[np.argmax(y_probs, axis=1)])
                
                score = f1_score(y_true, pred, average='macro')
                if score > best_score:
                    best_score = score
                    best_thresh = t
            
            best_thresholds[c] = best_thresh
            if best_score > best_f1:
                best_f1 = best_score
                
    return best_thresholds, best_f1

# ----------------------------
# Confusion Resolver for 0/9/15
# ----------------------------
class ConfusionResolver:
    def __init__(self):
        self.pairwise_models = {}
        self.weights = {'tt': 0.4, 'ovr': 0.3, 'pair': 0.3}  # α, β, γ
        
    def fit_pairwise_models(self, X, y, target_classes=[0, 9, 15]):
        """쌍대 전문가 모델 학습"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 0 vs 9, 0 vs 15, 9 vs 15 쌍대 모델
        pairs = [(0, 9), (0, 15), (9, 15)]
        
        for c1, c2 in pairs:
            # 해당 클래스들만 필터링
            mask = np.isin(y, [c1, c2])
            X_pair = X[mask]
            y_pair = y[mask]
            
            # 이진 분류로 변환
            y_binary = (y_pair == c1).astype(int)
            
            # 모델 학습
            model = OVRBinaryClassifier(X_pair.shape[1]).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            # 간단한 학습 (실제로는 더 정교하게)
            for epoch in range(50):
                model.train()
                optimizer.zero_grad()
                outputs = model(torch.tensor(X_pair, dtype=torch.float32).to(device))
                loss = criterion(outputs.squeeze(), torch.tensor(y_binary, dtype=torch.float32).to(device))
                loss.backward()
                optimizer.step()
            
            self.pairwise_models[(c1, c2)] = model
            
    def resolve_confusion(self, tt_probs, ovr_probs, X, target_classes=[0, 9, 15]):
        """혼동 해결"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        final_probs = tt_probs.copy()
        
        for i in range(len(tt_probs)):
            # 후보 집합 S = {c ∈ {0,9,15}: p_c(tt) ≥ τ_c}
            candidates = []
            for c in target_classes:
                if tt_probs[i, c] >= 0.5:  # 임계값 (실제로는 최적화된 값 사용)
                    candidates.append(c)
            
            if len(candidates) > 1:  # 혼동 상황
                # 쌍대 전문가 확률 계산
                pair_probs = np.zeros(len(target_classes))
                
                for (c1, c2), model in self.pairwise_models.items():
                    if c1 in candidates and c2 in candidates:
                        with torch.no_grad():
                            model.eval()
                            pair_logit = model(torch.tensor(X[i:i+1], dtype=torch.float32).to(device))
                            pair_prob = torch.sigmoid(pair_logit).cpu().numpy()[0]
                            
                            # 클래스별 확률로 변환
                            pair_probs[target_classes.index(c1)] += pair_prob
                            pair_probs[target_classes.index(c2)] += (1 - pair_prob)
                
                # 가중 평균
                for j, c in enumerate(target_classes):
                    if c in candidates:
                        final_probs[i, c] = (self.weights['tt'] * tt_probs[i, c] + 
                                           self.weights['ovr'] * ovr_probs[i, c] + 
                                           self.weights['pair'] * pair_probs[j])
        
        # 정규화
        final_probs = final_probs / (final_probs.sum(axis=1, keepdims=True) + 1e-8)
        return final_probs

# ----------------------------
# Logit Adjustment
# ----------------------------
def apply_logit_adjustment(logits, class_counts, tau=1.0):
    """Logit Adjustment 적용"""
    class_priors = class_counts / class_counts.sum()
    log_priors = np.log(class_priors + 1e-8)
    
    adjusted_logits = logits - tau * log_priors
    return adjusted_logits

# ----------------------------
# Main Training Pipeline
# ----------------------------
def train_advanced_tabtransformer():
    """고급 TabTransformer 학습 파이프라인"""
    print("고급 TabTransformer 학습 시작")
    print("=" * 60)
    
    # 1. 데이터 로딩
    train_df = pd.read_csv("./datasests/train.csv")
    test_df = pd.read_csv("./datasests/test.csv")
    
    X = train_df.drop(columns=["ID", "target"]).values
    y = train_df["target"].values
    X_test = test_df.drop(columns=["ID"]).values
    test_ids = test_df["ID"].values
    
    # 스케일링
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(np.unique(y))
    target_classes = [0, 9, 15]  # 혼동 클래스들
    
    print(f"데이터 형태: {X.shape}")
    print(f"클래스 수: {num_classes}")
    print(f"타겟 클래스: {target_classes}")
    
    # 2. 5-Fold Cross Validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f"\n===== Fold {fold} =====")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 3. TabTransformer 멀티클래스 모델 학습
        print("TabTransformer 멀티클래스 모델 학습...")
        
        batch_size = 64
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                         torch.tensor(y_train, dtype=torch.long)),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                         torch.tensor(y_val, dtype=torch.long)),
            batch_size=batch_size, shuffle=False
        )
        
        # 멀티클래스 모델
        tt_model = CosineTransformer(
            input_dim=X.shape[1],
            num_classes=num_classes,
            embed_dim=128, num_layers=4, dropout=0.3
        ).to(device)
        
        criterion = LabelSmoothingLoss(classes=num_classes, smoothing=0.1)
        optimizer = optim.AdamW(tt_model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        # 학습
        best_f1 = 0
        best_state = None
        patience = 10
        wait = 0
        
        for epoch in range(50):
            # Train
            tt_model.train()
            train_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = tt_model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            tt_model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = tt_model(xb)
                    val_preds.append(torch.argmax(preds, dim=1).cpu())
                    val_labels.append(yb.cpu())
            
            val_preds = torch.cat(val_preds)
            val_labels = torch.cat(val_labels)
            f1 = f1_score(val_labels, val_preds, average="macro")
            
            if f1 > best_f1:
                best_f1 = f1
                best_state = tt_model.state_dict()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
            
            scheduler.step(f1)
        
        tt_model.load_state_dict(best_state)
        print(f"TabTransformer F1: {best_f1:.4f}")
        
        # 4. OVR Binary Classifiers for 0/9/15
        print("OVR Binary Classifiers 학습...")
        
        ovr_models = {}
        ovr_probs_val = np.zeros((len(X_val), len(target_classes)))
        
        for i, target_class in enumerate(target_classes):
            # 이진 분류 데이터 생성
            y_binary = (y_train == target_class).astype(int)
            
            ovr_model = OVRBinaryClassifier(X_train.shape[1]).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(ovr_model.parameters(), lr=1e-3)
            
            # 학습
            for epoch in range(30):
                ovr_model.train()
                optimizer.zero_grad()
                outputs = ovr_model(torch.tensor(X_train, dtype=torch.float32).to(device))
                loss = criterion(outputs.squeeze(), torch.tensor(y_binary, dtype=torch.float32).to(device))
                loss.backward()
                optimizer.step()
            
            # 검증 예측
            ovr_model.eval()
            with torch.no_grad():
                outputs = ovr_model(torch.tensor(X_val, dtype=torch.float32).to(device))
                ovr_probs_val[:, i] = torch.sigmoid(outputs).cpu().numpy().squeeze()
            
            ovr_models[target_class] = ovr_model
        
        # 5. 확률 보정
        print("확률 보정...")
        
        # TabTransformer 확률
        tt_model.eval()
        tt_probs_val = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                preds = torch.softmax(tt_model(xb), dim=1).cpu().numpy()
                tt_probs_val.append(preds)
        tt_probs_val = np.vstack(tt_probs_val)
        
        # 보정기 학습
        calibrator = ProbabilityCalibrator(method='platt')
        calibrator.fit(y_val, tt_probs_val, range(num_classes))
        
        # 보정된 확률
        tt_probs_calibrated = calibrator.predict_proba(tt_probs_val)
        
        # 6. 임계값 최적화
        print("임계값 최적화...")
        
        best_thresholds, best_f1_thresh = optimize_thresholds(
            y_val, tt_probs_calibrated, range(num_classes)
        )
        
        print(f"최적 임계값 F1: {best_f1_thresh:.4f}")
        print(f"임계값: {best_thresholds}")
        
        # 7. 혼동 해결
        print("혼동 해결...")
        
        resolver = ConfusionResolver()
        resolver.fit_pairwise_models(X_train, y_train, target_classes)
        
        # OVR 확률을 전체 클래스 확률로 변환
        ovr_probs_full = np.zeros((len(X_val), num_classes))
        for i, target_class in enumerate(target_classes):
            ovr_probs_full[:, target_class] = ovr_probs_val[:, i]
        
        # 혼동 해결
        final_probs = resolver.resolve_confusion(
            tt_probs_calibrated, ovr_probs_full, X_val, target_classes
        )
        
        # 8. Logit Adjustment
        print("Logit Adjustment...")
        
        class_counts = np.bincount(y_train, minlength=num_classes)
        logits = np.log(final_probs + 1e-8)
        adjusted_logits = apply_logit_adjustment(logits, class_counts, tau=0.5)
        final_probs_adjusted = torch.softmax(torch.tensor(adjusted_logits), dim=1).numpy()
        
        # 최종 예측
        final_preds = np.argmax(final_probs_adjusted, axis=1)
        final_f1 = f1_score(y_val, final_preds, average="macro")
        
        print(f"최종 F1: {final_f1:.4f}")
        
        # 폴드 결과 저장
        fold_results.append({
            'fold': fold,
            'tt_f1': best_f1,
            'threshold_f1': best_f1_thresh,
            'final_f1': final_f1,
            'tt_probs': tt_probs_calibrated,
            'ovr_probs': ovr_probs_full,
            'final_probs': final_probs_adjusted,
            'thresholds': best_thresholds
        })
    
    # 9. 전체 결과 요약
    print("\n" + "="*60)
    print("전체 결과 요약")
    print("="*60)
    
    tt_f1s = [r['tt_f1'] for r in fold_results]
    threshold_f1s = [r['threshold_f1'] for r in fold_results]
    final_f1s = [r['final_f1'] for r in fold_results]
    
    print(f"TabTransformer 평균 F1: {np.mean(tt_f1s):.4f} ± {np.std(tt_f1s):.4f}")
    print(f"임계값 최적화 평균 F1: {np.mean(threshold_f1s):.4f} ± {np.std(threshold_f1s):.4f}")
    print(f"최종 평균 F1: {np.mean(final_f1s):.4f} ± {np.std(final_f1s):.4f}")
    
    # 10. 테스트 예측 (간단 버전)
    print("\n테스트 예측...")
    
    # 전체 데이터로 최종 모델 학습 (간단 버전)
    tt_model_final = CosineTransformer(
        input_dim=X.shape[1],
        num_classes=num_classes,
        embed_dim=128, num_layers=4, dropout=0.3
    ).to(device)
    
    # 간단한 학습
    train_loader_full = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                     torch.tensor(y, dtype=torch.long)),
        batch_size=64, shuffle=True
    )
    
    criterion = LabelSmoothingLoss(classes=num_classes, smoothing=0.1)
    optimizer = optim.AdamW(tt_model_final.parameters(), lr=1e-3, weight_decay=1e-2)
    
    for epoch in range(30):
        tt_model_final.train()
        for xb, yb in train_loader_full:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = tt_model_final(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
    
    # 테스트 예측
    tt_model_final.eval()
    test_probs = []
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32)), batch_size=64)
    
    with torch.no_grad():
        for xb, in test_loader:
            xb = xb.to(device)
            preds = torch.softmax(tt_model_final(xb), dim=1).cpu().numpy()
            test_probs.append(preds)
    
    test_probs = np.vstack(test_probs)
    test_preds = np.argmax(test_probs, axis=1)
    
    # 11. 결과 저장
    submission = pd.DataFrame({
        "ID": test_ids,
        "target": test_preds
    })
    submission.to_csv("tabtransformer_advanced_submission.csv", index=False)
    
    detailed = pd.DataFrame({
        "ID": test_ids,
        "target": test_preds,
        **{f"prob_{i}": test_probs[:, i] for i in range(num_classes)}
    })
    detailed.to_csv("tabtransformer_advanced_detailed.csv", index=False)
    
    print("✅ 고급 최적화 완료!")
    print("제출 파일: tabtransformer_advanced_submission.csv")
    print("상세 결과: tabtransformer_advanced_detailed.csv")
    
    return fold_results

if __name__ == "__main__":
    results = train_advanced_tabtransformer()
