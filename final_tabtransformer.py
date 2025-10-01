import os, random, numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, top_k_accuracy_score
import matplotlib.pyplot as plt

# ----------------------------
# Seed 고정
# ----------------------------
def set_seed(seed=220):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Label Smoothing Loss
# ----------------------------
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.05, dim=-1):
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
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
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
        z = self.embedding(x).unsqueeze(1)
        for ln in self.blocks:
            h = ln(z)
            Q,K,V = self.q(h), self.k(h), self.v(h)
            Qn = Q / (Q.norm(dim=-1,keepdim=True)+1e-8)
            Kn = K / (K.norm(dim=-1,keepdim=True)+1e-8)
            A = torch.softmax(Qn @ Kn.transpose(1,2), dim=-1)
            z = z + (A @ V)
        return self.classifier(z.mean(1))

# ----------------------------
# Lookahead Optimizer
# ----------------------------
class Lookahead(optim.Optimizer):
    def __init__(self, base_optimizer, alpha=0.3, k=5):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("Invalid alpha")
        if not 1 <= k:
            raise ValueError("Invalid k")

        defaults = dict(alpha=alpha, k=k)
        super().__init__(base_optimizer.param_groups, defaults)

        self.base_optimizer = base_optimizer
        self.alpha = alpha
        self.k = k
        self.param_groups = self.base_optimizer.param_groups

        self.state = {}
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = {'slow_param': p.data.clone()}
        self.step_counter = 0

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self.step_counter += 1
        if self.step_counter % self.k != 0:
            return loss
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                slow = self.state[p]['slow_param']
                slow += self.alpha * (p.data - slow)
                p.data.copy_(slow)
        return loss

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

# ----------------------------
# 데이터 로드
# ----------------------------
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

X = train_df.drop(columns=["ID","target"]).values
y = train_df["target"].values
X_test = test_df.drop(columns=["ID"]).values
test_ids = test_df["ID"].values
num_classes = len(np.unique(y))

# ----------------------------
# 학습 함수 (Seed × Fold)
# ----------------------------
def train_and_save(seed=220, epochs=50, batch_size=64, out_dir="926"):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_probs = np.zeros((len(X), num_classes))
    test_probs_folds = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X,y),1):
        print(f"\n===== Seed {seed} | Fold {fold} =====")

        scaler = RobustScaler()
        X_tr, X_va = scaler.fit_transform(X[tr_idx]), scaler.transform(X[va_idx])
        y_tr_, y_va_ = y[tr_idx], y[va_idx]
        X_te = scaler.transform(X_test)

        tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr,dtype=torch.float32),
                                             torch.tensor(y_tr_,dtype=torch.long)),
                               batch_size=batch_size, shuffle=True)
        va_loader = DataLoader(TensorDataset(torch.tensor(X_va,dtype=torch.float32),
                                             torch.tensor(y_va_,dtype=torch.long)),
                               batch_size=256, shuffle=False)
        te_loader = DataLoader(torch.tensor(X_te,dtype=torch.float32),
                               batch_size=256, shuffle=False)

        model = CosineTransformer(X.shape[1], num_classes).to(device)
        criterion = LabelSmoothingLoss(classes=num_classes, smoothing=0.05)
        base_optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
        optimizer = Lookahead(base_optimizer, alpha=0.3, k=5)

        steps_per_epoch = len(tr_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            base_optimizer, max_lr=2e-3,
            steps_per_epoch=steps_per_epoch, epochs=epochs,
            pct_start=0.3, anneal_strategy="cos",
            div_factor=25.0, final_div_factor=1e4
        )

        best_f1, best_state = 0, None
        for epoch in range(1, epochs+1):
            model.train(); train_losses=[]
            for xb,yb in tr_loader:
                xb,yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward(); optimizer.step()
                train_losses.append(loss.item())
                scheduler.step()

            # Validation
            model.eval(); va_logits=[]
            with torch.no_grad():
                for xb,yb in va_loader:
                    xb,yb = xb.to(device), yb.to(device)
                    va_logits.append(model(xb).cpu())
            va_logits = torch.cat(va_logits)
            va_probs = torch.softmax(va_logits,1).numpy()
            va_preds = va_probs.argmax(1)
            f1 = f1_score(y_va_, va_preds, average="macro")
            print(f"Epoch {epoch:02d} | Loss={np.mean(train_losses):.4f} | Val_F1={f1:.4f}")

            if f1 > best_f1:
                best_f1, best_state = f1, model.state_dict()

        # Save best
        model.load_state_dict(best_state)

        # Validation 저장
        model.eval(); va_logits=[]
        with torch.no_grad():
            for xb in torch.tensor(X_va,dtype=torch.float32).split(256):
                xb=xb.to(device); va_logits.append(model(xb).cpu())
        va_logits = torch.cat(va_logits)
        va_probs = torch.softmax(va_logits,1).numpy()
        oof_probs[va_idx] = va_probs
        np.save(f"{out_dir}/probs_seed{seed}_fold{fold}.npy", va_probs)

        # Test 저장
        te_logits=[]
        with torch.no_grad():
            for xb in te_loader:
                xb=xb.to(device); te_logits.append(model(xb).cpu())
        te_logits = torch.cat(te_logits)
        te_probs = torch.softmax(te_logits,1).numpy()
        np.save(f"{out_dir}/test_probs_seed{seed}_fold{fold}.npy", te_probs)
        test_probs_folds.append(te_probs)

        print(f"[Seed {seed} | Fold {fold}] Best F1={best_f1:.4f}")

    return oof_probs, test_probs_folds

# ----------------------------
# Seed Ensemble 실행
# ----------------------------
def run_seed_ensemble(seeds=[220,518,819,77,452], epochs=50, out_dir="926"):
    all_oof = []
    all_test = []

    for seed in seeds:
        oof_probs, test_probs_folds = train_and_save(seed=seed, epochs=epochs, out_dir=out_dir)
        all_oof.append(oof_probs)
        all_test.append(np.mean(test_probs_folds, axis=0))  # fold 평균

    # ---- Seed Ensemble ----
    oof_probs_ens = np.mean(all_oof, axis=0)
    test_probs_ens = np.mean(all_test, axis=0)

    # 평가
    oof_preds = oof_probs_ens.argmax(1)
    oof_f1 = f1_score(y, oof_preds, average="macro")
    oof_top3 = top_k_accuracy_score(y, oof_probs_ens, k=3, labels=np.arange(num_classes))

    cm = confusion_matrix(y, oof_preds, labels=np.arange(num_classes))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title("OOF Confusion Matrix (Seed Ensemble)")
    plt.savefig(f"{out_dir}/cm_oof_seedensemble.png"); plt.close()

    np.save(f"{out_dir}/oof_probs_seedensemble.npy", oof_probs_ens)
    np.save(f"{out_dir}/test_probs_seedensemble.npy", test_probs_ens)

    print("\n===== 최종 Seed Ensemble 결과 =====")
    print("OOF Macro-F1:", oof_f1)
    print("OOF Top-3 Accuracy:", oof_top3)

    # ---- Submission ----
    test_preds = test_probs_ens.argmax(1)
    submission = pd.DataFrame({"ID": test_ids, "target": test_preds})
    submission.to_csv(f"{out_dir}/submission.csv", index=False)
    print(f"{out_dir}/submission.csv 저장 완료")

# ----------------------------
# 실행
# ----------------------------
set_seed(220)
run_seed_ensemble(seeds=[220,518,819,77,452], epochs=60, out_dir="926")