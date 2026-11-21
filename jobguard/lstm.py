"""BiLSTM baseline in PyTorch, for comparison with the linear models.

Usage:
    pip install torch
    python -m jobguard.lstm --data data/fake_job_postings.csv

Uses the same train/test split as train.py (same seed), so the numbers are
directly comparable. Writes reports/lstm_metrics.json, which train.py adds to
the results table. The web app does not need PyTorch.
"""
import argparse
import json
import time
from collections import Counter

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split

from .data import load_dataset
from .train import ROOT, SEED, best_threshold, scores

MAX_LEN, VOCAB, EMB, HID = 300, 30000, 128, 64


def encode(texts, vocab):
    out = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, 1) for w in t.split()[:MAX_LEN]]
        out[i, :len(ids)] = ids
    return torch.tensor(out)


class BiLSTM(nn.Module):
    def __init__(self, n_words):
        super().__init__()
        self.emb = nn.Embedding(n_words, EMB, padding_idx=0)
        self.lstm = nn.LSTM(EMB, HID, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(2 * HID, 1)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        h = h.masked_fill((x == 0).unsqueeze(-1), -1e4).max(dim=1).values   # max-pool over real tokens
        return self.out(self.drop(h)).squeeze(-1)


def predict(model, X, batch=256):
    model.eval()
    with torch.no_grad():
        return torch.cat([torch.sigmoid(model(X[i:i + batch])) for i in range(0, len(X), batch)]).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "fake_job_postings.csv"))
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X, y = load_dataset(args.data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=SEED)

    counts = Counter(w for t in X_tr["text"] for w in t.split()[:MAX_LEN])
    vocab = {w: i + 2 for i, (w, _) in enumerate(counts.most_common(VOCAB - 2))}   # 0 = pad, 1 = unknown
    tr, val, te = (encode(d["text"].tolist(), vocab) for d in (X_tr, X_val, X_test))
    ytr = torch.tensor(y_tr.values, dtype=torch.float32)

    model = BiLSTM(VOCAB)
    pos_weight = torch.tensor((len(ytr) - ytr.sum()) / ytr.sum())          # fake posts are rare
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    best_auc, best_state = -1, None
    from sklearn.metrics import average_precision_score
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, order = time.time(), torch.randperm(len(tr))
        for i in range(0, len(tr), 64):
            idx = order[i:i + 64]
            opt.zero_grad()
            loss = loss_fn(model(tr[idx]), ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_auc = average_precision_score(y_val, predict(model, val))
        print(f"epoch {epoch}: val PR-AUC {val_auc:.3f} ({time.time() - t0:.0f}s)")
        if val_auc > best_auc:                       # keep the best epoch (early stopping)
            best_auc, best_state = val_auc, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    thr = best_threshold(y_val.values, predict(model, val))
    result = scores(y_test.values, predict(model, te), thr)
    print("test:", result)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "lstm_metrics.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
