# =========================================
# 1. 라이브러리 임포트
# =========================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================================
# 2. 시드 고정
# =========================================
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(123)

# =========================================
# 3. Label Smoothing Loss
# =========================================
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

# =========================================
# 4. 모델 정의
# =========================================
# ---- 1) Base Cosine Transformer ----
class CosineBase(nn.Module):
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
        z = self.embedding(x).unsqueeze(1)
        for ln in self.blocks:
            h = ln(z)
            Q, K, V = self.q(h), self.k(h), self.v(h)
            Qn = Q / (Q.norm(dim=-1, keepdim=True) + 1e-8)
            Kn = K / (K.norm(dim=-1, keepdim=True) + 1e-8)
            A = torch.softmax(Qn @ Kn.transpose(1, 2), dim=-1)
            z = z + A @ V
        z = z.mean(dim=1)
        return self.classifier(z)

# ---- 2) MultiHead Cosine Attention ----
class CosineMultiHead(nn.Module):
    def __init__(self, input_dim, num_classes, embed_dim=128, num_heads=4, num_layers=4, dropout=0.3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
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
        z = self.embedding(x).unsqueeze(1)
        B, T, D = z.shape
        for ln in self.blocks:
            h = ln(z)
            Q, K, V = self.q(h), self.k(h), self.v(h)
            Q = Q.view(B, T, self.num_heads, self.head_dim)
            K = K.view(B, T, self.num_heads, self.head_dim)
            V = V.view(B, T, self.num_heads, self.head_dim)
            Qn = Q / (Q.norm(dim=-1, keepdim=True) + 1e-8)
            Kn = K / (K.norm(dim=-1, keepdim=True) + 1e-8)
            A = torch.softmax(torch.einsum("bthd,bThd->bhtT", Qn, Kn), dim=-1)
            out = torch.einsum("bhtT,bThd->bthd", A, V).reshape(B, T, D)
            z = z + out
        z = z.mean(dim=1)
        return self.classifier(z)

# ---- 3) Cosine + FFN ----
class CosineFFN(nn.Module):
    def __init__(self, input_dim, num_classes, embed_dim=128, num_layers=4, dropout=0.3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(embed_dim),
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim*4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim*4, embed_dim),
                    nn.Dropout(dropout)
                )
            ]) for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        z = self.embedding(x).unsqueeze(1)
        for ln, ffn in self.blocks:
            h = ln(z)
            Q, K, V = self.q(h), self.k(h), self.v(h)
            Qn = Q / (Q.norm(dim=-1, keepdim=True) + 1e-8)
            Kn = K / (K.norm(dim=-1, keepdim=True) + 1e-8)
            A = torch.softmax(Qn @ Kn.transpose(1, 2), dim=-1)
            attn_out = A @ V
            z = z + attn_out
            z = z + ffn(z)
        z = z.mean(dim=1)
        return self.classifier(z)

# ---- 4) Full Transformer-style ----
class CosineFull(nn.Module):
    def __init__(self, input_dim, num_classes, embed_dim=128, num_heads=4, num_layers=4, dropout=0.3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.blocks = nn.ModuleList([])
        for _ in range(num_layers):
            self.blocks.append(nn.ModuleList([
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim*4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim*4, embed_dim),
                    nn.Dropout(dropout)
                )
            ]))
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        z = self.embedding(x).unsqueeze(1)
        B, T, D = z.shape
        for ln1, q, k, v, ln2, ffn in self.blocks:
            h = ln1(z)
            Q, K, V = q(h), k(h), v(h)
            Q = Q.view(B, T, self.num_heads, self.head_dim)
            K = K.view(B, T, self.num_heads, self.head_dim)
            V = V.view(B, T, self.num_heads, self.head_dim)
            Qn = Q / (Q.norm(dim=-1, keepdim=True) + 1e-8)
            Kn = K / (K.norm(dim=-1, keepdim=True) + 1e-8)
            A = torch.softmax(torch.einsum("bthd,bThd->bhtT", Qn, Kn), dim=-1)
            attn_out = torch.einsum("bhtT,bThd->bthd", A, V).reshape(B, T, D)
            z = z + attn_out
            z = z + ffn(ln2(z))
        z = z.mean(dim=1)
        return self.classifier(z)

# =========================================
# 5. 데이터 로딩
# =========================================
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

X = train_df.drop(columns=["ID","target"]).values
y = train_df["target"].values
X_test = test_df.drop(columns=["ID"]).values
test_ids = test_df["ID"].values
num_classes = len(np.unique(y))

scaler = RobustScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)
train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                        torch.tensor(y_train, dtype=torch.long)), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                      torch.tensor(y_val, dtype=torch.long)), batch_size=256, shuffle=False)

# =========================================
# 6. 학습 함수
# =========================================
def train_model(model, name, epochs=20, lr=1e-3):
    criterion = LabelSmoothingLoss(classes=num_classes, smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_f1 = 0
    for epoch in range(1, epochs+1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

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
        print(f"[{name}] Epoch {epoch:02d} | Val F1: {f1:.4f}")
    return best_f1

# =========================================
# 7. 네 가지 모델 비교
# =========================================
results = {}
models = {
    "Base": CosineBase(X.shape[1], num_classes).to(device),
    "MultiHead": CosineMultiHead(X.shape[1], num_classes).to(device),
    "FFN": CosineFFN(X.shape[1], num_classes).to(device),
    "Full": CosineFull(X.shape[1], num_classes).to(device)
}
for name, model in models.items():
    print(f"\n=== Training {name} ===")
    best_f1 = train_model(model, name, epochs=20, lr=1e-3)
    results[name] = best_f1

print("\n=== 최종 비교 테이블 ===")
for k,v in results.items():
    print(f"{k:10s} | Best Macro-F1: {v:.4f}")
