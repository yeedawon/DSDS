"""
8단계: 목표 도메인 × 학습 전략 (per-target 평가) — OOP 재설계.
연구 질문: 목표 풀에서 잘하려면 (targeted) 그 풀만 학습 vs (broad) 전체 학습?
평가는 '목표 도메인의 test 슬라이스'로만 수행한다.
"""
import os, json, time, random
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
import torch
from datasets import load_dataset


# ============================ OOP: 도메인 / 전략 ============================
class Domain(ABC):
    """소스 집합으로 정의되는 데이터 도메인. 학습 풀이자 동시에 test 슬라이스."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def contains(self, source: str) -> bool: ...

    def test_mask(self, sources) -> np.ndarray:
        return np.array([self.contains(s) for s in sources])


class SourceSetDomain(Domain):
    def __init__(self, name, sources):
        self._name = name
        self.sources = set(sources)

    @property
    def name(self):
        return self._name

    def contains(self, source):
        return source in self.sources


class UniverseDomain(Domain):
    """모든 소스를 포함하는 도메인 (전체 학습 풀)."""

    @property
    def name(self):
        return "combined"

    def contains(self, source):
        return True


class TrainingStrategy(ABC):
    """목표가 주어졌을 때 어느 풀로 학습할지 결정 (전략 패턴)."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def pool(self, target: Domain) -> Domain: ...


class TargetedStrategy(TrainingStrategy):
    @property
    def name(self):
        return "targeted"

    def pool(self, target):
        return target  # 목표 풀에만 한정


class BroadStrategy(TrainingStrategy):
    def __init__(self, universe: Domain):
        self._u = universe

    @property
    def name(self):
        return "broad"

    def pool(self, target):
        return self._u  # 전체


FORMAL = SourceSetDomain("formal", {"wikipedia", "wikinews", "wikitree", "policy"})
COLLOQUIAL = SourceSetDomain("colloquial", {"NSMC", "airbnb"})
ALL = UniverseDomain()


# ============================ 지표 / 유틸 ============================
def evaluate(y_true, y_pred, n=3):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    acc = float((y_true == y_pred).mean())
    f1s = []
    for c in range(n):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return acc, float(np.mean(f1s))


def stratified_sample(indices, n_total, labels, seed=42):
    rng = random.Random(seed)
    by = {0: [], 1: [], 2: []}
    for i in indices:
        by[labels[i]].append(i)
    out = []
    for c in [0, 1, 2]:
        pool = by[c][:]
        rng.shuffle(pool)
        out.extend(pool[: n_total // 3])
    rng.shuffle(out)
    return out


def stratified_holdout(X, y, frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    dev = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        dev.extend(ci[: int(len(ci) * frac)])
    mask = np.zeros(len(y), bool)
    mask[dev] = True
    return X[~mask], y[~mask], X[mask], y[mask]


# ============================ 분류기 (노트북과 동일) ============================
class KNNClassifier:
    def __init__(self, k=5):
        self.k = int(k)

    @staticmethod
    def _l2(X):
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    def fit(self, X, y):
        self._X = self._l2(np.asarray(X, np.float32))
        self._y = np.asarray(y)
        return self

    def predict_many(self, X, batch=512):
        Xn = self._l2(np.asarray(X, np.float32))
        preds = np.empty(len(Xn), np.int64)
        for i in range(0, len(Xn), batch):
            sims = Xn[i:i + batch] @ self._X.T
            topk = np.argpartition(-sims, self.k - 1, axis=1)[:, : self.k]
            for j, row in enumerate(topk):
                preds[i + j] = np.bincount(self._y[row], minlength=3).argmax()
        return preds


class NNClassifier:
    def __init__(self, hidden=256, lr=0.1, epochs=150, batch=128, l2=1e-4,
                 momentum=0.9, patience=15, seed=42):
        self.hidden, self.lr, self.epochs, self.batch = hidden, lr, epochs, batch
        self.l2, self.momentum, self.patience, self.seed = l2, momentum, patience, seed

    def _std_fit(self, X):
        self._mu = X.mean(0, keepdims=True)
        self._sig = X.std(0, keepdims=True) + 1e-8

    def _std(self, X):
        return (X - self._mu) / self._sig

    def _init(self, d_in, d_out):
        rng = np.random.RandomState(self.seed)
        self.W1 = (rng.randn(d_in, self.hidden) * np.sqrt(2 / d_in)).astype(np.float32)
        self.b1 = np.zeros((1, self.hidden), np.float32)
        self.W2 = (rng.randn(self.hidden, d_out) * np.sqrt(2 / self.hidden)).astype(np.float32)
        self.b2 = np.zeros((1, d_out), np.float32)
        self._vW1 = np.zeros_like(self.W1); self._vb1 = np.zeros_like(self.b1)
        self._vW2 = np.zeros_like(self.W2); self._vb2 = np.zeros_like(self.b2)

    @staticmethod
    def _softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def _fwd(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        p = self._softmax(a1 @ self.W2 + self.b2)
        return z1, a1, p

    def fit(self, X, y, X_dev=None, y_dev=None):
        self._std_fit(np.asarray(X, np.float32))
        Xs = self._std(np.asarray(X, np.float32))
        y = np.asarray(y)
        n, d_in = Xs.shape
        d_out = int(y.max()) + 1
        Y = np.eye(d_out, dtype=np.float32)[y]
        self._init(d_in, d_out)
        rng = np.random.RandomState(self.seed)
        best_f1, best_ep, best, wait = -1, self.epochs, None, 0
        for ep in range(1, self.epochs + 1):
            order = rng.permutation(n)
            for s in range(0, n, self.batch):
                idx = order[s:s + self.batch]
                xb, yb = Xs[idx], Y[idx]
                m = len(idx)
                z1, a1, p = self._fwd(xb)
                dz2 = (p - yb) / m
                dW2 = a1.T @ dz2 + self.l2 * self.W2
                db2 = dz2.sum(0, keepdims=True)
                dz1 = (dz2 @ self.W2.T) * (z1 > 0)
                dW1 = xb.T @ dz1 + self.l2 * self.W1
                db1 = dz1.sum(0, keepdims=True)
                self._vW2 = self.momentum * self._vW2 - self.lr * dW2
                self._vb2 = self.momentum * self._vb2 - self.lr * db2
                self._vW1 = self.momentum * self._vW1 - self.lr * dW1
                self._vb1 = self.momentum * self._vb1 - self.lr * db1
                self.W2 += self._vW2; self.b2 += self._vb2
                self.W1 += self._vW1; self.b1 += self._vb1
            if X_dev is not None:
                _, f1 = evaluate(y_dev, self.predict_many(X_dev))
                if f1 > best_f1:
                    best_f1, best_ep, wait = f1, ep, 0
                    best = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
        if best:
            self.W1, self.b1, self.W2, self.b2 = best
        self.best_epoch_ = best_ep
        return self

    def predict_many(self, X):
        _, _, p = self._fwd(self._std(np.asarray(X, np.float32)))
        return p.argmax(1)


class BertClassifier:
    def __init__(self, model_name="klue/bert-base", epochs=3, batch=32, lr=2e-5,
                 max_length=128, seed=42):
        self.model_name, self.epochs, self.batch = model_name, epochs, batch
        self.lr, self.max_length, self.seed = lr, max_length, seed
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    def _enc(self, prem, hyp):
        return self._tok(list(prem), list(hyp), truncation=True, padding=True,
                         max_length=self.max_length, return_tensors="pt")

    def fit(self, prem, hyp, y):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        torch.manual_seed(self.seed)
        y = np.asarray(y); n = len(prem)
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=int(y.max()) + 1).to(self.device)
        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr)
        rng = np.random.RandomState(self.seed)
        for ep in range(1, self.epochs + 1):
            self._model.train()
            order = rng.permutation(n)
            run = 0.0
            for s in range(0, n, self.batch):
                idx = order[s:s + self.batch]
                enc = self._enc([prem[i] for i in idx], [hyp[i] for i in idx]).to(self.device)
                lab = torch.tensor(y[idx], dtype=torch.long, device=self.device)
                out = self._model(**enc, labels=lab)
                out.loss.backward(); opt.step(); opt.zero_grad()
                run += out.loss.item() * len(idx)
            print(f"    [collo-BERT] epoch {ep}: train_loss={run / n:.4f}", flush=True)
        return self

    @torch.no_grad()
    def predict_many(self, prem, hyp):
        self._model.eval()
        preds = []
        for s in range(0, len(prem), self.batch):
            enc = self._enc(prem[s:s + self.batch], hyp[s:s + self.batch]).to(self.device)
            preds.append(self._model(**enc).logits.argmax(-1).cpu().numpy())
        return np.concatenate(preds)


# ============================ 실행 ============================
print("데이터 로드...", flush=True)
ds = load_dataset("klue/klue", "nli")
train, val = ds["train"], ds["validation"]
labels = train["label"]; srcs = train["source"]
val_src = np.array(val["source"])
y_val = np.load("emb/val_y.npy")
val_X = np.load("emb/val_X.npy")

# --- 구어체 학습 풀 구성 + 임베딩 (캐시 있으면 재사용) ---
if os.path.exists("emb/colloquial_X.npy"):
    collo_X = np.load("emb/colloquial_X.npy"); collo_y = np.load("emb/colloquial_y.npy")
    print("구어체 임베딩 캐시 로드", flush=True)
else:
    collo_idx = [i for i, s in enumerate(srcs) if s in COLLOQUIAL.sources]
    samp = stratified_sample(collo_idx, 6000, labels)
    collo = train.select(samp)
    print(f"구어체 학습셋: {len(collo)}건, 라벨 {dict(Counter(collo['label']))}", flush=True)
    from sentence_transformers import SentenceTransformer
    sb = SentenceTransformer("jhgan/ko-sroberta-multitask", device="mps")

    def emb(dset):
        u = sb.encode(list(dset["premise"]), batch_size=128, convert_to_numpy=True,
                      normalize_embeddings=True, show_progress_bar=False)
        v = sb.encode(list(dset["hypothesis"]), batch_size=128, convert_to_numpy=True,
                      normalize_embeddings=True, show_progress_bar=False)
        return np.concatenate([u, v, np.abs(u - v), u * v], -1).astype(np.float32)

    collo_X = emb(collo); collo_y = np.array(collo["label"], np.int64)
    np.save("emb/colloquial_X.npy", collo_X); np.save("emb/colloquial_y.npy", collo_y)
    print(f"구어체 임베딩 저장: {collo_X.shape}", flush=True)

# --- KNN / NN 구어체 학습 → val 전체 예측 저장 ---
Xf, yf, Xd, yd = stratified_holdout(collo_X, collo_y)
best_k = max([1, 3, 5, 7, 9, 15, 25, 51],
             key=lambda k: evaluate(yd, KNNClassifier(k).fit(Xf, yf).predict_many(Xd))[1])
np.save("preds/KNN_colloquial.npy", KNNClassifier(best_k).fit(collo_X, collo_y).predict_many(val_X))
print(f"KNN 구어체 학습 완료 (k={best_k})", flush=True)

probe = NNClassifier().fit(Xf, yf, Xd, yd)
np.save("preds/NN_colloquial.npy",
        NNClassifier(epochs=probe.best_epoch_).fit(collo_X, collo_y).predict_many(val_X))
print(f"NN 구어체 학습 완료 (epochs={probe.best_epoch_})", flush=True)

# --- BERT 구어체 학습 → val 전체 예측 저장 ---
if os.path.exists("preds/BERT_colloquial.npy"):
    print("BERT 구어체 예측 캐시 로드", flush=True)
else:
    collo_idx = [i for i, s in enumerate(srcs) if s in COLLOQUIAL.sources]
    samp = stratified_sample(collo_idx, 6000, labels)
    collo = train.select(samp)
    print("BERT 구어체 학습 시작 (~13분)...", flush=True)
    t = time.time()
    clf = BertClassifier(epochs=3).fit(list(collo["premise"]), list(collo["hypothesis"]),
                                       np.array(collo["label"]))
    pred = clf.predict_many(list(val["premise"]), list(val["hypothesis"]))
    np.save("preds/BERT_colloquial.npy", pred)
    print(f"BERT 구어체 학습 완료 ({(time.time() - t) / 60:.1f}분)", flush=True)

# ============================ per-target 2×2 평가 ============================
MODELS = ["KNN", "NN", "BERT"]
targeted, broad = TargetedStrategy(), BroadStrategy(ALL)
summary = {}
print("\n" + "=" * 70, flush=True)
print("2x2 per-target: 목표 슬라이스에서 (targeted=목표풀만) vs (broad=전체)", flush=True)
print("=" * 70, flush=True)
for target in [FORMAL, COLLOQUIAL]:
    mask = target.test_mask(val_src)
    print(f"\n[목표={target.name}]  test 슬라이스 {int(mask.sum())}건", flush=True)
    print(f"{'model':6s} | {'targeted (acc/mF1)':>20s} | {'broad (acc/mF1)':>20s} | 승자", flush=True)
    for m in MODELS:
        p_t = np.load(f"preds/{m}_{targeted.pool(target).name}.npy")
        p_b = np.load(f"preds/{m}_{broad.pool(target).name}.npy")
        at, ft = evaluate(y_val[mask], p_t[mask])
        ab, fb = evaluate(y_val[mask], p_b[mask])
        win = "targeted" if ft > fb else ("broad" if fb > ft else "tie")
        summary[f"{m}/{target.name}"] = {
            "targeted": {"acc": at, "mf1": ft}, "broad": {"acc": ab, "mf1": fb}, "winner": win}
        print(f"{m:6s} | {at:6.4f} / {ft:6.4f}      | {ab:6.4f} / {fb:6.4f}      | {win}", flush=True)

json.dump(summary, open("results_pertarget.json", "w"), ensure_ascii=False, indent=2)
print("\n→ results_pertarget.json 저장 완료", flush=True)
