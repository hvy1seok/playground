#!/usr/bin/env python3
"""
TabTransformer 메타 스태킹 + 모델 다양성 앙상블
LightGBM/LogReg 메타 모델로 최종 성능 극대화
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import warnings
warnings.filterwarnings('ignore')

# LightGBM 설치 확인
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("LightGBM이 설치되지 않았습니다. pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

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
# Cosine Transformer (다양한 구성)
# ----------------------------
class CosineTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, embed_dim=128, num_layers=4, dropout=0.3, variant='base'):
        super().__init__()
        self.variant = variant
        
        if variant == 'wide':
            embed_dim = embed_dim * 2
        elif variant == 'deep':
            num_layers = num_layers + 2
        elif variant == 'narrow':
            embed_dim = embed_dim // 2
            
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.blocks = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        # 다양한 분류기 구성
        if variant == 'wide':
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes)
            )
        elif variant == 'deep':
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, num_classes)
            )
        else:  # base, narrow
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
# OVR Binary Classifier
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
# Meta Stacking Models
# ----------------------------
class MetaStacker:
    def __init__(self, base_models=['logistic', 'lightgbm', 'random_forest']):
        self.base_models = base_models
        self.meta_models = {}
        self.feature_names = []
        
    def create_meta_features(self, base_predictions, y_true=None):
        """메타 특징 생성"""
        meta_features = []
        feature_names = []
        
        # 1. 기본 확률 특징
        for i, (model_name, probs) in enumerate(base_predictions.items()):
            for j in range(probs.shape[1]):
                meta_features.append(probs[:, j])
                feature_names.append(f"{model_name}_prob_{j}")
        
        # 2. 확률 통계 특징
        all_probs = np.array(list(base_predictions.values()))
        
        # 평균 확률
        mean_probs = np.mean(all_probs, axis=0)
        for j in range(mean_probs.shape[1]):
            meta_features.append(mean_probs[:, j])
            feature_names.append(f"mean_prob_{j}")
        
        # 확률 분산
        var_probs = np.var(all_probs, axis=0)
        for j in range(var_probs.shape[1]):
            meta_features.append(var_probs[:, j])
            feature_names.append(f"var_prob_{j}")
        
        # 3. 샤프니스 지표
        for i, (model_name, probs) in enumerate(base_predictions.items()):
            # 엔트로피
            entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
            meta_features.append(entropy)
            feature_names.append(f"{model_name}_entropy")
            
            # 마진 (top1 - top2)
            sorted_probs = np.sort(probs, axis=1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            meta_features.append(margin)
            feature_names.append(f"{model_name}_margin")
            
            # 최대 확률
            max_prob = np.max(probs, axis=1)
            meta_features.append(max_prob)
            feature_names.append(f"{model_name}_max_prob")
        
        # 4. 모델 간 일치도
        if len(base_predictions) > 1:
            predictions = [np.argmax(probs, axis=1) for probs in base_predictions.values()]
            
            # 예측 일치도
            agreement = np.mean([pred == predictions[0] for pred in predictions[1:]], axis=0)
            meta_features.append(agreement)
            feature_names.append("prediction_agreement")
            
            # 확률 유사도 (코사인 유사도)
            from sklearn.metrics.pairwise import cosine_similarity
            prob_similarity = []
            for i in range(len(probs)):
                similarities = []
                for j in range(len(base_predictions)):
                    for k in range(j+1, len(base_predictions)):
                        sim = cosine_similarity(
                            list(base_predictions.values())[j][i:i+1],
                            list(base_predictions.values())[k][i:i+1]
                        )[0, 0]
                        similarities.append(sim)
                prob_similarity.append(np.mean(similarities))
            meta_features.append(prob_similarity)
            feature_names.append("prob_similarity")
        
        self.feature_names = feature_names
        return np.array(meta_features).T
    
    def fit(self, base_predictions, y_true):
        """메타 모델 학습"""
        X_meta = self.create_meta_features(base_predictions, y_true)
        
        for model_name in self.base_models:
            if model_name == 'logistic':
                meta_model = LogisticRegression(
                    random_state=42, 
                    max_iter=1000,
                    C=1.0
                )
            elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                meta_model = lgb.LGBMClassifier(
                    random_state=42,
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    verbose=-1
                )
            elif model_name == 'random_forest':
                meta_model = RandomForestClassifier(
                    random_state=42,
                    n_estimators=100,
                    max_depth=5
                )
            else:
                continue
            
            meta_model.fit(X_meta, y_true)
            self.meta_models[model_name] = meta_model
    
    def predict_proba(self, base_predictions):
        """메타 모델 예측"""
        X_meta = self.create_meta_features(base_predictions)
        
        predictions = {}
        for model_name, meta_model in self.meta_models.items():
            pred_proba = meta_model.predict_proba(X_meta)
            predictions[model_name] = pred_proba
        
        return predictions

# ----------------------------
# Main Training Pipeline
# ----------------------------
def train_meta_stacking_tabtransformer():
    """메타 스태킹 TabTransformer 학습"""
    print("메타 스태킹 TabTransformer 학습 시작")
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
    target_classes = [0, 9, 15]
    
    print(f"데이터 형태: {X.shape}")
    print(f"클래스 수: {num_classes}")
    print(f"타겟 클래스: {target_classes}")
    
    # 2. 5-Fold Cross Validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    fold_results = []
    
    # OOF 예측 저장용
    oof_predictions = {
        'tt_base': np.zeros((len(X), num_classes)),
        'tt_wide': np.zeros((len(X), num_classes)),
        'tt_deep': np.zeros((len(X), num_classes)),
        'tt_narrow': np.zeros((len(X), num_classes)),
        'ovr_0': np.zeros((len(X), num_classes)),
        'ovr_9': np.zeros((len(X), num_classes)),
        'ovr_15': np.zeros((len(X), num_classes))
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f"\n===== Fold {fold} =====")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
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
        
        # 3. 다양한 TabTransformer 모델 학습
        model_variants = ['base', 'wide', 'deep', 'narrow']
        models = {}
        model_scores = {}
        
        for variant in model_variants:
            print(f"TabTransformer {variant} 모델 학습...")
            
            model = CosineTransformer(
                input_dim=X.shape[1],
                num_classes=num_classes,
                embed_dim=128,
                num_layers=4,
                dropout=0.3,
                variant=variant
            ).to(device)
            
            criterion = LabelSmoothingLoss(classes=num_classes, smoothing=0.1)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
            
            # 학습
            best_f1 = 0
            best_state = None
            patience = 8
            wait = 0
            
            for epoch in range(30):
                # Train
                model.train()
                train_loss = 0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_preds, val_labels = [], []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        preds = model(xb)
                        val_preds.append(torch.argmax(preds, dim=1).cpu())
                        val_labels.append(yb.cpu())
                
                val_preds = torch.cat(val_preds)
                val_labels = torch.cat(val_labels)
                f1 = f1_score(val_labels, val_preds, average="macro")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_state = model.state_dict()
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break
                
                scheduler.step(f1)
            
            model.load_state_dict(best_state)
            models[f'tt_{variant}'] = model
            model_scores[f'tt_{variant}'] = best_f1
            print(f"TabTransformer {variant} F1: {best_f1:.4f}")
        
        # 4. OVR Binary Classifiers 학습
        print("OVR Binary Classifiers 학습...")
        
        for target_class in target_classes:
            y_binary = (y_train == target_class).astype(int)
            
            ovr_model = OVRBinaryClassifier(X_train.shape[1]).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(ovr_model.parameters(), lr=1e-3)
            
            # 학습
            for epoch in range(20):
                ovr_model.train()
                optimizer.zero_grad()
                outputs = ovr_model(torch.tensor(X_train, dtype=torch.float32).to(device))
                loss = criterion(outputs.squeeze(), torch.tensor(y_binary, dtype=torch.float32).to(device))
                loss.backward()
                optimizer.step()
            
            models[f'ovr_{target_class}'] = ovr_model
        
        # 5. 검증 예측
        print("검증 예측...")
        
        val_predictions = {}
        
        # TabTransformer 모델들
        for model_name, model in models.items():
            if model_name.startswith('tt_'):
                model.eval()
                val_probs = []
                with torch.no_grad():
                    for xb, _ in val_loader:
                        xb = xb.to(device)
                        preds = torch.softmax(model(xb), dim=1).cpu().numpy()
                        val_probs.append(preds)
                val_probs = np.vstack(val_probs)
                val_predictions[model_name] = val_probs
                
                # OOF 저장
                oof_predictions[model_name][val_idx] = val_probs
        
        # OVR 모델들
        for target_class in target_classes:
            model = models[f'ovr_{target_class}']
            model.eval()
            with torch.no_grad():
                outputs = model(torch.tensor(X_val, dtype=torch.float32).to(device))
                ovr_probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
            
            # 전체 클래스 확률로 변환
            full_probs = np.zeros((len(X_val), num_classes))
            full_probs[:, target_class] = ovr_probs
            val_predictions[f'ovr_{target_class}'] = full_probs
            
            # OOF 저장
            oof_predictions[f'ovr_{target_class}'][val_idx] = full_probs
        
        # 6. 메타 스태킹
        print("메타 스태킹...")
        
        meta_stacker = MetaStacker()
        meta_stacker.fit(val_predictions, y_val)
        
        # 메타 모델 예측
        meta_predictions = meta_stacker.predict_proba(val_predictions)
        
        # 최종 예측 (가중 평균)
        final_probs = np.zeros((len(X_val), num_classes))
        weights = {'logistic': 0.4, 'lightgbm': 0.4, 'random_forest': 0.2}
        
        for model_name, probs in meta_predictions.items():
            if model_name in weights:
                final_probs += weights[model_name] * probs
        
        final_preds = np.argmax(final_probs, axis=1)
        final_f1 = f1_score(y_val, final_preds, average="macro")
        
        print(f"메타 스태킹 F1: {final_f1:.4f}")
        
        # 폴드 결과 저장
        fold_results.append({
            'fold': fold,
            'model_scores': model_scores,
            'meta_f1': final_f1,
            'meta_predictions': meta_predictions
        })
    
    # 7. 전체 OOF 메타 스태킹
    print("\n전체 OOF 메타 스태킹...")
    
    # OOF 예측을 메타 특징으로 변환
    meta_stacker_final = MetaStacker()
    meta_stacker_final.fit(oof_predictions, y)
    
    # 8. 테스트 예측
    print("테스트 예측...")
    
    # 전체 데이터로 최종 모델들 학습 (간단 버전)
    test_models = {}
    
    # TabTransformer 모델들
    for variant in model_variants:
        model = CosineTransformer(
            input_dim=X.shape[1],
            num_classes=num_classes,
            embed_dim=128,
            num_layers=4,
            dropout=0.3,
            variant=variant
        ).to(device)
        
        # 간단한 학습
        train_loader_full = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                         torch.tensor(y, dtype=torch.long)),
            batch_size=64, shuffle=True
        )
        
        criterion = LabelSmoothingLoss(classes=num_classes, smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        
        for epoch in range(20):
            model.train()
            for xb, yb in train_loader_full:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
        
        test_models[f'tt_{variant}'] = model
    
    # OVR 모델들
    for target_class in target_classes:
        y_binary = (y == target_class).astype(int)
        
        model = OVRBinaryClassifier(X.shape[1]).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(15):
            model.train()
            optimizer.zero_grad()
            outputs = model(torch.tensor(X, dtype=torch.float32).to(device))
            loss = criterion(outputs.squeeze(), torch.tensor(y_binary, dtype=torch.float32).to(device))
            loss.backward()
            optimizer.step()
        
        test_models[f'ovr_{target_class}'] = model
    
    # 테스트 예측
    test_predictions = {}
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32)), batch_size=64)
    
    # TabTransformer 모델들
    for model_name, model in test_models.items():
        if model_name.startswith('tt_'):
            model.eval()
            test_probs = []
            with torch.no_grad():
                for xb, in test_loader:
                    xb = xb.to(device)
                    preds = torch.softmax(model(xb), dim=1).cpu().numpy()
                    test_probs.append(preds)
            test_probs = np.vstack(test_probs)
            test_predictions[model_name] = test_probs
    
    # OVR 모델들
    for target_class in target_classes:
        model = test_models[f'ovr_{target_class}']
        model.eval()
        with torch.no_grad():
            outputs = model(torch.tensor(X_test, dtype=torch.float32).to(device))
            ovr_probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
        
        # 전체 클래스 확률로 변환
        full_probs = np.zeros((len(X_test), num_classes))
        full_probs[:, target_class] = ovr_probs
        test_predictions[f'ovr_{target_class}'] = full_probs
    
    # 메타 모델 예측
    test_meta_predictions = meta_stacker_final.predict_proba(test_predictions)
    
    # 최종 예측
    final_test_probs = np.zeros((len(X_test), num_classes))
    weights = {'logistic': 0.4, 'lightgbm': 0.4, 'random_forest': 0.2}
    
    for model_name, probs in test_meta_predictions.items():
        if model_name in weights:
            final_test_probs += weights[model_name] * probs
    
    final_test_preds = np.argmax(final_test_probs, axis=1)
    
    # 9. 결과 저장
    submission = pd.DataFrame({
        "ID": test_ids,
        "target": final_test_preds
    })
    submission.to_csv("tabtransformer_meta_stacking_submission.csv", index=False)
    
    detailed = pd.DataFrame({
        "ID": test_ids,
        "target": final_test_preds,
        **{f"prob_{i}": final_test_probs[:, i] for i in range(num_classes)}
    })
    detailed.to_csv("tabtransformer_meta_stacking_detailed.csv", index=False)
    
    print("✅ 메타 스태킹 완료!")
    print("제출 파일: tabtransformer_meta_stacking_submission.csv")
    print("상세 결과: tabtransformer_meta_stacking_detailed.csv")
    
    # 10. 결과 요약
    print("\n" + "="*60)
    print("결과 요약")
    print("="*60)
    
    for fold_result in fold_results:
        print(f"Fold {fold_result['fold']}:")
        for model_name, score in fold_result['model_scores'].items():
            print(f"  {model_name}: {score:.4f}")
        print(f"  Meta Stacking: {fold_result['meta_f1']:.4f}")
        print()
    
    return fold_results

if __name__ == "__main__":
    results = train_meta_stacking_tabtransformer()
