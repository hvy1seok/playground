#!/usr/bin/env python3
"""
TabTransformer + iTransformer 최종 앙상블 스크립트
TabTransformer 5폴드 소프트 보팅 + iTransformer Specialist 앙상블 결합
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score
import random
import time
import os
import sys

# 시드 고정
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(123)

# TabTransformer 모델 (tabtransformer_classification.py에서 가져옴)
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

def train_tabtransformer_5fold():
    """TabTransformer 5폴드 학습 및 예측"""
    print("=" * 60)
    print("TabTransformer 5폴드 학습 시작")
    print("=" * 60)
    
    # 데이터 로딩
    train_df = pd.read_csv("./datasests/train.csv")
    test_df = pd.read_csv("./datasests/test.csv")
    
    X = train_df.drop(columns=["ID", "target"]).values
    y = train_df["target"].values
    X_test = test_df.drop(columns=["ID"]).values
    test_ids = test_df["ID"].values
    
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(np.unique(y))
    
    # 5-Fold Cross Validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    fold_f1 = []
    test_probs_all = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f"\n===== TabTransformer Fold {fold} =====")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.long)),
            batch_size=64, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                          torch.tensor(y_val, dtype=torch.long)),
            batch_size=256, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                          torch.zeros(len(X_test))),
            batch_size=256, shuffle=False
        )
        
        model = CosineTransformer(
            input_dim=X.shape[1],
            num_classes=num_classes,
            embed_dim=128, num_layers=4, dropout=0.3
        ).to(device)
        
        criterion = LabelSmoothingLoss(classes=num_classes, smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        
        best_f1, best_state = 0, None
        patience, wait = 7, 0
        max_epochs = 30
        
        for epoch in range(1, max_epochs + 1):
            # Train
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            # Validation
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    all_preds.append(torch.argmax(preds, dim=1).cpu())
                    all_labels.append(yb.cpu())
            f1 = f1_score(torch.cat(all_labels), torch.cat(all_preds), average="macro")
            
            if f1 > best_f1:
                best_f1 = f1
                best_state = model.state_dict()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:02d} | Macro-F1: {f1:.4f}")
        
        print(f"[TabTransformer Fold {fold}] Best Macro-F1: {best_f1:.4f}")
        fold_f1.append(best_f1)
        
        # Inference on test
        model.load_state_dict(best_state)
        model.eval()
        test_probs = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                preds = torch.softmax(model(xb), dim=1).cpu().numpy()
                test_probs.append(preds)
        test_probs_all.append(np.vstack(test_probs))
    
    # Soft Voting Ensemble
    ensemble_probs = np.mean(test_probs_all, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    
    # TabTransformer 결과 저장
    tabtransformer_submission = pd.DataFrame({"ID": test_ids, "target": ensemble_preds})
    tabtransformer_submission.to_csv("tabtransformer_5fold_submission.csv", index=False)
    
    # 상세 결과 저장 (확률 포함)
    tabtransformer_detailed = pd.DataFrame({
        "ID": test_ids,
        "target": ensemble_preds,
        **{f"prob_{i}": ensemble_probs[:, i] for i in range(num_classes)}
    })
    tabtransformer_detailed.to_csv("tabtransformer_5fold_detailed.csv", index=False)
    
    print(f"\nTabTransformer 5폴드 완료:")
    print(f"Fold별 F1: {fold_f1}")
    print(f"평균 F1: {np.mean(fold_f1):.4f}")
    print(f"Submission 파일: tabtransformer_5fold_submission.csv")
    print(f"상세 결과 파일: tabtransformer_5fold_detailed.csv")
    
    return ensemble_probs, ensemble_preds, test_ids, fold_f1

def load_itransformer_results():
    """iTransformer Specialist 앙상블 결과 로드"""
    print("\n" + "=" * 60)
    print("iTransformer Specialist 앙상블 결과 로드")
    print("=" * 60)
    
    # iTransformer Specialist 앙상블 상세 결과 로드
    if os.path.exists("itransformer_specialist_ensemble_detailed.csv"):
        itransformer_detailed = pd.read_csv("itransformer_specialist_ensemble_detailed.csv")
        itransformer_probs = itransformer_detailed[[f'prob_{i}' for i in range(21)]].values
        itransformer_preds = itransformer_detailed['target'].values
        test_ids = itransformer_detailed['ID'].values
        
        print(f"iTransformer Specialist 앙상블 결과 로드 완료")
        print(f"데이터 형태: {itransformer_probs.shape}")
        print(f"예측 분포: {np.bincount(itransformer_preds)}")
        
        return itransformer_probs, itransformer_preds, test_ids
    else:
        print("❌ iTransformer Specialist 앙상블 결과를 찾을 수 없습니다.")
        print("itransformer_classification.py를 먼저 실행하세요.")
        return None, None, None

def create_final_ensemble(tabtransformer_probs, itransformer_probs, test_ids, 
                         tabtransformer_weight=0.4, itransformer_weight=0.6):
    """최종 앙상블 생성"""
    print("\n" + "=" * 60)
    print("최종 앙상블 생성")
    print("=" * 60)
    
    # 가중 평균 (TabTransformer + iTransformer)
    final_probs = (tabtransformer_weight * tabtransformer_probs + 
                   itransformer_weight * itransformer_probs)
    
    # 최종 예측
    final_preds = np.argmax(final_probs, axis=1)
    
    # 결과 저장
    final_submission = pd.DataFrame({
        "ID": test_ids,
        "target": final_preds
    })
    final_submission.to_csv("final_ensemble_submission.csv", index=False)
    
    # 상세 결과 저장
    final_detailed = pd.DataFrame({
        "ID": test_ids,
        "target": final_preds,
        **{f"prob_{i}": final_probs[:, i] for i in range(21)}
    })
    final_detailed.to_csv("final_ensemble_detailed.csv", index=False)
    
    print(f"최종 앙상블 완료:")
    print(f"TabTransformer 가중치: {tabtransformer_weight}")
    print(f"iTransformer 가중치: {itransformer_weight}")
    print(f"예측 분포: {np.bincount(final_preds)}")
    print(f"Submission 파일: final_ensemble_submission.csv")
    print(f"상세 결과 파일: final_detailed.csv")
    
    return final_probs, final_preds

def analyze_ensemble_results(tabtransformer_probs, itransformer_probs, final_probs, 
                           tabtransformer_preds, itransformer_preds, final_preds):
    """앙상블 결과 분석"""
    print("\n" + "=" * 60)
    print("앙상블 결과 분석")
    print("=" * 60)
    
    # 예측 일치도 분석
    tab_itransformer_agreement = np.mean(tabtransformer_preds == itransformer_preds)
    tab_final_agreement = np.mean(tabtransformer_preds == final_preds)
    itransformer_final_agreement = np.mean(itransformer_preds == final_preds)
    
    print(f"TabTransformer vs iTransformer 일치도: {tab_itransformer_agreement:.4f}")
    print(f"TabTransformer vs Final 일치도: {tab_final_agreement:.4f}")
    print(f"iTransformer vs Final 일치도: {itransformer_final_agreement:.4f}")
    
    # 확률 분포 분석
    tab_entropy = -np.sum(tabtransformer_probs * np.log(tabtransformer_probs + 1e-8), axis=1).mean()
    itransformer_entropy = -np.sum(itransformer_probs * np.log(itransformer_probs + 1e-8), axis=1).mean()
    final_entropy = -np.sum(final_probs * np.log(final_probs + 1e-8), axis=1).mean()
    
    print(f"\n평균 엔트로피 (불확실성):")
    print(f"TabTransformer: {tab_entropy:.4f}")
    print(f"iTransformer: {itransformer_entropy:.4f}")
    print(f"Final Ensemble: {final_entropy:.4f}")
    
    # 클래스별 예측 분포
    print(f"\n클래스별 예측 분포:")
    print(f"TabTransformer: {np.bincount(tabtransformer_preds, minlength=21)}")
    print(f"iTransformer: {np.bincount(itransformer_preds, minlength=21)}")
    print(f"Final Ensemble: {np.bincount(final_preds, minlength=21)}")

def main():
    """메인 함수"""
    print("TabTransformer + iTransformer 최종 앙상블 시작")
    print("=" * 80)
    
    # 1. TabTransformer 5폴드 학습 및 예측
    tabtransformer_probs, tabtransformer_preds, test_ids, tabtransformer_f1 = train_tabtransformer_5fold()
    
    # 2. iTransformer Specialist 앙상블 결과 로드
    itransformer_probs, itransformer_preds, _ = load_itransformer_results()
    
    if itransformer_probs is None:
        print("❌ iTransformer 결과를 로드할 수 없습니다. 프로그램을 종료합니다.")
        return
    
    # 3. 최종 앙상블 생성 (여러 가중치 조합 실험)
    print("\n" + "=" * 60)
    print("가중치 조합 실험")
    print("=" * 60)
    
    weight_combinations = [
        (0.3, 0.7),  # iTransformer에 더 높은 가중치
        (0.4, 0.6),  # 균형
        (0.5, 0.5),  # 동일 가중치
        (0.6, 0.4),  # TabTransformer에 더 높은 가중치
    ]
    
    best_combination = None
    best_entropy = float('inf')
    
    for tab_weight, itransformer_weight in weight_combinations:
        final_probs, final_preds = create_final_ensemble(
            tabtransformer_probs, itransformer_probs, test_ids,
            tab_weight, itransformer_weight
        )
        
        # 엔트로피 계산 (낮을수록 확신도가 높음)
        entropy = -np.sum(final_probs * np.log(final_probs + 1e-8), axis=1).mean()
        
        print(f"가중치 ({tab_weight:.1f}, {itransformer_weight:.1f}): 엔트로피 = {entropy:.4f}")
        
        if entropy < best_entropy:
            best_entropy = entropy
            best_combination = (tab_weight, itransformer_weight)
    
    print(f"\n최적 가중치 조합: {best_combination} (엔트로피: {best_entropy:.4f})")
    
    # 4. 최종 결과 생성
    final_probs, final_preds = create_final_ensemble(
        tabtransformer_probs, itransformer_probs, test_ids,
        best_combination[0], best_combination[1]
    )
    
    # 5. 결과 분석
    analyze_ensemble_results(
        tabtransformer_probs, itransformer_probs, final_probs,
        tabtransformer_preds, itransformer_preds, final_preds
    )
    
    print("\n" + "=" * 80)
    print("최종 앙상블 완료!")
    print("=" * 80)
    print(f"TabTransformer 5폴드 평균 F1: {np.mean(tabtransformer_f1):.4f}")
    print(f"최종 제출 파일: final_ensemble_submission.csv")
    print(f"상세 결과 파일: final_ensemble_detailed.csv")

if __name__ == "__main__":
    main()
