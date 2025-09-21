import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random

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
# 데이터 로딩
# ----------------------------
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

X = train_df.drop(columns=["ID", "target"]).values
y = train_df["target"].values
X_test = test_df.drop(columns=["ID"]).values

scaler = RobustScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_tensor = torch.tensor(X_test, dtype=torch.float32)
test_loader = DataLoader(TensorDataset(test_tensor, torch.zeros(len(test_tensor))), batch_size=256)

num_classes = len(np.unique(y))

# ----------------------------
# 5-Fold Cross Validation
# ----------------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
fold_f1 = []
test_probs_all = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\n===== Fold {fold} =====")
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
        print(f"Epoch {epoch:02d} | Macro-F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    print(f"[Fold {fold}] Best Macro-F1: {best_f1:.4f}")
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

# ----------------------------
# Soft Voting Ensemble
# ----------------------------
ensemble_probs = np.mean(test_probs_all, axis=0)
ensemble_preds = np.argmax(ensemble_probs, axis=1)

submission = pd.DataFrame({"ID": test_df["ID"], "target": ensemble_preds})
submission.to_csv("submission_cosine_5fold.csv", index=False)

print("\nFold별 F1:", fold_f1)
print("평균 F1:", np.mean(fold_f1))
print("submission_cosine_5fold.csv 저장 완료")