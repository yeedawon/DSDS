"""
통합 가산 증강 실험 (KNN / NN / BERT) — 증분 체크포인트 방식.
- 목표 풀에서 2,000 베이스라인 → 한계까지 데이터를 '한 번' 진행하며,
  total=2000,3000,4000,6000,8000,... 체크포인트마다 목표 슬라이스 macro-F1 기록.
  (각 크기 독립 재학습 아님 — 하나의 점진 경로에서 중간 결과 보고)
- 세 모델 모두 핵심 하이퍼파라미터를 held-out dev에서 선택:
    KNN: k,  NN: 학습률(발산 자동 제외),  BERT: 학습률.
- KNN/NN은 시드 8개 평균, BERT는 비용상 단일 시드.
출력: results_augmentation.json  (키: "MODEL/target/strategy/total" -> {mean,std})
"""
import json, numpy as np, torch, time
from abc import ABC, abstractmethod
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

FORMAL = {"wikipedia", "wikinews", "wikitree", "policy"}
COLLO = {"NSMC", "airbnb"}
dev_t = "mps" if torch.backends.mps.is_available() else "cpu"

ds = load_dataset("klue/klue", "nli"); tr = ds["train"]; va = ds["validation"]
tr_src = np.array(tr["source"]); tr_y = np.array(tr["label"], np.int64)
tr_p = list(tr["premise"]); tr_h = list(tr["hypothesis"])
va_src = np.array(va["source"]); va_y = np.load("emb/val_y.npy")
va_p = list(va["premise"]); va_h = list(va["hypothesis"])
tr_X = np.load("emb/train_all_X.npy"); va_X = np.load("emb/val_X.npy")

def cpool(srcset):
    m = np.array([s in srcset for s in tr_src])
    return {c: np.where(m & (tr_y == c))[0] for c in range(3)}
WHOLE = {c: np.where(tr_y == c)[0] for c in range(3)}
TARGETS = {"formal": FORMAL, "colloquial": COLLO}
CKPTS = [2001, 3000, 3999, 6000, 8001, 9999, 12000, 14001]
BASE_PC = 667
SEEDS = list(range(8))

def macro_f1(yt, yp):
    f = []
    for c in range(3):
        tp = int(((yp == c) & (yt == c)).sum()); fp = int(((yp == c) & (yt != c)).sum()); fn = int(((yp != c) & (yt == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
        f.append(2 * p * r / (p + r) if p + r else 0)
    return float(np.mean(f))

def build_sequence(tset, strat, seed):
    """베이스라인(목표 2000, 인터리브) + 증강(targeted=목표 / broad=전체), 클래스 인터리브 → 프리픽스가 항상 라벨 균형."""
    rng = np.random.RandomState(7000 + seed)
    tp = cpool(tset)
    base = {c: rng.permutation(tp[c]) for c in range(3)}
    used = {c: set(base[c][:BASE_PC].tolist()) for c in range(3)}
    if strat == "targeted":
        rest = {c: list(base[c][BASE_PC:]) for c in range(3)}
        cap_pc = min(len(tp[c]) for c in range(3))
    else:
        wsh = {c: rng.permutation(WHOLE[c]) for c in range(3)}
        rest = {c: [i for i in wsh[c].tolist() if i not in used[c]] for c in range(3)}
        cap_pc = 4667
    seq = []
    for i in range(BASE_PC):
        for c in range(3): seq.append(int(base[c][i]))
    for i in range(cap_pc - BASE_PC):
        for c in range(3):
            if i < len(rest[c]): seq.append(int(rest[c][i]))
    return seq

def slice_ckpts(maxtot):
    return [c for c in CKPTS if c <= maxtot]

val_mask = {t: np.array([s in TARGETS[t] for s in va_src]) for t in TARGETS}


# ============================ KNN (증분 프리픽스, k는 dev 선택) ============================
def knn_predict(Xtr, ytr, Xq, k):
    Xn = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-12)
    Qn = Xq / (np.linalg.norm(Xq, axis=1, keepdims=True) + 1e-12)
    out = np.empty(len(Qn), np.int64)
    for i in range(0, len(Qn), 1024):
        s = Qn[i:i + 1024] @ Xn.T
        tk = np.argpartition(-s, k - 1, axis=1)[:, :k]
        for j, row in enumerate(tk): out[i + j] = np.bincount(ytr[row], minlength=3).argmax()
    return out

def run_knn(target, strat, mask):
    curves = {N: [] for N in CKPTS}
    for sd in SEEDS:
        seq = build_sequence(TARGETS[target], strat, sd)
        full = np.array(seq); Xf = tr_X[full]; yf = tr_y[full]
        # k를 전체 시퀀스 dev(20%)에서 한 번 선택 → 모든 프리픽스에 적용
        rng = np.random.RandomState(sd); perm = rng.permutation(len(full)); nd = max(30, int(0.2 * len(full)))
        dvi, tni = perm[:nd], perm[nd:]
        best_k, best_f = 5, -1
        for k in [1, 5, 15, 25, 51]:
            pf = knn_predict(Xf[tni], yf[tni], Xf[dvi], k)
            f = macro_f1(yf[dvi], pf)
            if f > best_f: best_f, best_k = f, k
        for N in slice_ckpts(len(seq)):
            idx = full[:N]
            pred = knn_predict(tr_X[idx], tr_y[idx], va_X, best_k)
            curves[N].append(macro_f1(va_y[mask], pred[mask]))
    return curves


# ============================ NN (증분 단일 패스, 발산 가드) ============================
def nn_incremental(seq, lr, seed, ckpts, mask, hidden=128, batch=64, l2=1e-4, mom=0.9):
    rng = np.random.RandomState(seed)
    # 표준화: 전체 시퀀스 기준(고정)
    Xseq = tr_X[seq]; mu = Xseq.mean(0, keepdims=True); sg = Xseq.std(0, keepdims=True) + 1e-8
    Vv = (va_X - mu) / sg
    d = tr_X.shape[1]
    W1 = (rng.randn(d, hidden) * np.sqrt(2 / d)).astype(np.float32); b1 = np.zeros((1, hidden), np.float32)
    W2 = (rng.randn(hidden, 3) * np.sqrt(2 / hidden)).astype(np.float32); b2 = np.zeros((1, 3), np.float32)
    vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1); vW2 = np.zeros_like(W2); vb2 = np.zeros_like(b2)
    def fwd(X):
        z1 = X @ W1 + b1; a1 = np.maximum(z1, 0); z2 = a1 @ W2 + b2
        z2 = z2 - z2.max(1, keepdims=True); e = np.exp(z2); return z1, a1, e / e.sum(1, keepdims=True)
    rec = {}; ci = 0; ck = list(ckpts)
    try:
        with np.errstate(over="raise", invalid="raise"):
            for s in range(0, len(seq), batch):
                idx = seq[s:s + batch]
                xb = (tr_X[idx] - mu) / sg; yb = np.eye(3, dtype=np.float32)[tr_y[idx]]; m = len(idx)
                z1, a1, p = fwd(xb)
                dz2 = (p - yb) / m; dW2 = a1.T @ dz2 + l2 * W2; db2 = dz2.sum(0, keepdims=True)
                dz1 = (dz2 @ W2.T) * (z1 > 0); dW1 = xb.T @ dz1 + l2 * W1; db1 = dz1.sum(0, keepdims=True)
                vW2 = mom * vW2 - lr * dW2; vb2 = mom * vb2 - lr * db2
                vW1 = mom * vW1 - lr * dW1; vb1 = mom * vb1 - lr * db1
                W2 += vW2; b2 += vb2; W1 += vW1; b1 += vb1
                processed = s + m
                while ci < len(ck) and processed >= ck[ci]:
                    _, _, pv = fwd(Vv); rec[ck[ci]] = macro_f1(va_y[mask], pv.argmax(1)[mask]); ci += 1
    except FloatingPointError:
        return None
    return rec

def select_nn_lr():
    """대표 시퀀스(구어체 broad, 시드 0)에서 dev로 lr 선택."""
    seq = build_sequence(COLLO, "broad", 0)
    rng = np.random.RandomState(0); perm = rng.permutation(len(seq))
    nd = int(0.2 * len(seq)); dev_idx = [seq[i] for i in perm[:nd]]; tr_idx = [seq[i] for i in perm[nd:]]
    Xd = tr_X[dev_idx]; yd = tr_y[dev_idx]
    best_lr, best_f = 0.01, -1
    for lr in [0.003, 0.01, 0.03, 0.1]:
        # tr_idx로 증분 패스 후 dev 평가 (val mask 대신 dev)
        Xseq = tr_X[tr_idx]; mu = Xseq.mean(0, keepdims=True); sg = Xseq.std(0, keepdims=True) + 1e-8
        d = tr_X.shape[1]; rng2 = np.random.RandomState(0)
        W1 = (rng2.randn(d, 128) * np.sqrt(2 / d)).astype(np.float32); b1 = np.zeros((1, 128), np.float32)
        W2 = (rng2.randn(128, 3) * np.sqrt(2 / 128)).astype(np.float32); b2 = np.zeros((1, 3), np.float32)
        vs = [np.zeros_like(x) for x in (W1, b1, W2, b2)]
        def fwd(X):
            z1 = X @ W1 + b1; a1 = np.maximum(z1, 0); z2 = a1 @ W2 + b2
            z2 = z2 - z2.max(1, keepdims=True); e = np.exp(z2); return z1, a1, e / e.sum(1, keepdims=True)
        ok = True
        try:
            with np.errstate(over="raise", invalid="raise"):
                for s in range(0, len(tr_idx), 64):
                    idx = tr_idx[s:s + 64]; xb = (tr_X[idx] - mu) / sg; yb = np.eye(3, dtype=np.float32)[tr_y[idx]]; m = len(idx)
                    z1, a1, p = fwd(xb); dz2 = (p - yb) / m
                    dW2 = a1.T @ dz2 + 1e-4 * W2; db2 = dz2.sum(0, keepdims=True)
                    dz1 = (dz2 @ W2.T) * (z1 > 0); dW1 = xb.T @ dz1 + 1e-4 * W1; db1 = dz1.sum(0, keepdims=True)
                    vs[2] = 0.9 * vs[2] - lr * dW2; vs[3] = 0.9 * vs[3] - lr * db2
                    vs[0] = 0.9 * vs[0] - lr * dW1; vs[1] = 0.9 * vs[1] - lr * db1
                    W2 += vs[2]; b2 += vs[3]; W1 += vs[0]; b1 += vs[1]
        except FloatingPointError:
            ok = False
        if not ok: continue
        _, _, pd = fwd((Xd - mu) / sg); f = macro_f1(yd, pd.argmax(1))
        if f > best_f: best_f, best_lr = f, lr
    return best_lr

def run_nn(target, strat, mask, lr):
    curves = {N: [] for N in CKPTS}
    for sd in SEEDS:
        seq = build_sequence(TARGETS[target], strat, sd)
        rec = nn_incremental(seq, lr, sd, slice_ckpts(len(seq)), mask)
        if rec is None: continue   # 발산(이 lr에선 드묾)
        for N, f in rec.items(): curves[N].append(f)
    return curves


# ============================ BERT (증분 단일 패스, lr는 dev 선택) ============================
tok = AutoTokenizer.from_pretrained("klue/bert-base")

@torch.no_grad()
def bert_eval(model, P, H, y, sel):
    model.eval(); preds = []
    for s in range(0, len(sel), 64):
        ii = sel[s:s + 64]
        enc = tok([P[i] for i in ii], [H[i] for i in ii], truncation=True, padding=True, max_length=128, return_tensors="pt").to(dev_t)
        preds.append(model(**enc).logits.argmax(-1).cpu().numpy())
    return macro_f1(y[sel], np.concatenate(preds))

def bert_incremental(seq, lr, ckpts, mask_idx, eval_P, eval_H, eval_y, seed=42):
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained("klue/bert-base", num_labels=3).to(dev_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rec = {}; ci = 0; ck = list(ckpts)
    model.train()
    for s in range(0, len(seq), 32):
        idx = seq[s:s + 32]
        enc = tok([tr_p[i] for i in idx], [tr_h[i] for i in idx], truncation=True, padding=True, max_length=128, return_tensors="pt").to(dev_t)
        lab = torch.tensor([int(tr_y[i]) for i in idx], device=dev_t)
        out = model(**enc, labels=lab); out.loss.backward(); opt.step(); opt.zero_grad()
        processed = s + len(idx)
        while ci < len(ck) and processed >= ck[ci]:
            rec[ck[ci]] = bert_eval(model, eval_P, eval_H, eval_y, mask_idx); model.train(); ci += 1
    return rec

def select_bert_lr():
    """구어체 broad 시퀀스를 6000까지만 단일패스로, dev로 lr 선택."""
    seq = build_sequence(COLLO, "broad", 0)
    # dev: 시퀀스에서 안 쓰는 구어체 예제 1000개
    used = set(seq); pool = [i for i in np.where(np.array([s in COLLO for s in tr_src]))[0] if i not in used][:1002]
    dev_P = [tr_p[i] for i in pool]; dev_H = [tr_h[i] for i in pool]; dev_y = tr_y[pool]
    seq6 = seq[:6000]
    best_lr, best_f = 2e-5, -1
    for lr in [2e-5, 3e-5, 5e-5]:
        torch.manual_seed(42)
        model = AutoModelForSequenceClassification.from_pretrained("klue/bert-base", num_labels=3).to(dev_t)
        opt = torch.optim.AdamW(model.parameters(), lr=lr); model.train()
        for s in range(0, len(seq6), 32):
            idx = seq6[s:s + 32]
            enc = tok([tr_p[i] for i in idx], [tr_h[i] for i in idx], truncation=True, padding=True, max_length=128, return_tensors="pt").to(dev_t)
            lab = torch.tensor([int(tr_y[i]) for i in idx], device=dev_t)
            out = model(**enc, labels=lab); out.loss.backward(); opt.step(); opt.zero_grad()
        f = bert_eval(model, dev_P, dev_H, dev_y, list(range(len(pool))))
        print(f"  [BERT lr={lr}] dev macro-F1={f:.4f}", flush=True)
        if f > best_f: best_f, best_lr = f, lr
    print(f"  → BERT 선택 lr={best_lr}", flush=True)
    return best_lr

def run_bert(target, strat, lr):
    sel = np.where(val_mask[target])[0].tolist()
    curves = {N: [] for N in CKPTS}
    for sd in SEEDS:                 # KNN/NN과 동일 시드 → 공정 비교
        seq = build_sequence(TARGETS[target], strat, sd)
        rec = bert_incremental(seq, lr, slice_ckpts(len(seq)), sel, va_p, va_h, va_y, seed=sd)
        for N, f in rec.items(): curves[N].append(f)
        print(f"      [BERT {target}/{strat} seed{sd}] 완료", flush=True)
    return curves


# ============================ 실행 ============================
results = {}
def store(model, target, strat, curves):
    for N, vals in curves.items():
        if vals:
            results[f"{model}/{target}/{strat}/{N}"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

def main():
    import os
    # 기존 결과(KNN/NN/_hparams) 있으면 재사용 — 비싼 부분만 다시 계산
    have = {}
    if os.path.exists("results_augmentation.json"):
        have = json.load(open("results_augmentation.json")); results.update(have)
    nn_lr = have.get("_hparams", {}).get("nn_lr") or select_nn_lr()
    bert_lr = have.get("_hparams", {}).get("bert_lr") or select_bert_lr()
    results["_hparams"] = {"nn_lr": nn_lr, "bert_lr": bert_lr, "knn_k": "dev-selected", "seeds": len(SEEDS)}
    print(f"hparams: nn_lr={nn_lr}, bert_lr={bert_lr}, seeds={len(SEEDS)}", flush=True)

    for target in ["formal", "colloquial"]:
        mask = val_mask[target]
        for strat in ["targeted", "broad"]:
            if f"KNN/{target}/{strat}/2001" not in results:      # 캐시 없으면만
                store("KNN", target, strat, run_knn(target, strat, mask))
            if f"NN/{target}/{strat}/2001" not in results:
                store("NN", target, strat, run_nn(target, strat, mask, nn_lr))
            if results.get(f"BERT/{target}/{strat}/2001", {}).get("n", 0) != len(SEEDS):   # 8시드로 끝난 것만 건너뜀(재개용)
                t0 = time.time()
                store("BERT", target, strat, run_bert(target, strat, bert_lr))
                json.dump(results, open("results_augmentation.json", "w"), ensure_ascii=False, indent=2)  # 중간 저장
                print(f"  [{target}/{strat}] BERT {len(SEEDS)}시드 완료·저장 ({(time.time()-t0)/60:.1f}분)", flush=True)
            else:
                print(f"  [{target}/{strat}] BERT 캐시됨 — 건너뜀", flush=True)

    print("\n→ results_augmentation.json 저장 (BERT 다중 시드 반영)", flush=True)

if __name__ == "__main__":
    main()
