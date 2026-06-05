"""
저데이터 + 다중 시드 데이터량 스윕으로 targeted vs broad 의 통계적 유의성 검증.
- N ∈ {500,1000,2000,4000,6000}, 시드 8개
- targeted: 목표 풀에서 N건 / broad: 전체에서 N건 (둘 다 라벨 균형) — 같은 N (양 통제)
- 평가: 목표 도메인의 test 슬라이스 (격식체 1800 / 구어체 1200)
- 모델: KNN, NN(NumPy). 통계: 시드별 paired diff(targeted-broad) → t-검정
"""
import os, json, math
from abc import ABC, abstractmethod
from collections import Counter
import numpy as np
from datasets import load_dataset


# ===== OOP: Domain / Strategy (노트북과 동일 추상) =====
class Domain(ABC):
    @property
    @abstractmethod
    def name(self): ...
    @abstractmethod
    def contains(self, source): ...
    def test_mask(self, sources):
        return np.array([self.contains(s) for s in sources])

class SourceSetDomain(Domain):
    def __init__(self, name, sources): self._n=name; self.sources=set(sources)
    @property
    def name(self): return self._n
    def contains(self, s): return s in self.sources

class UniverseDomain(Domain):
    @property
    def name(self): return "all"
    def contains(self, s): return True

class TrainingStrategy(ABC):
    @property
    @abstractmethod
    def name(self): ...
    @abstractmethod
    def pool(self, target): ...

class TargetedStrategy(TrainingStrategy):
    @property
    def name(self): return "targeted"
    def pool(self, target): return target

class BroadStrategy(TrainingStrategy):
    def __init__(self, u): self._u=u
    @property
    def name(self): return "broad"
    def pool(self, target): return self._u

FORMAL = SourceSetDomain("formal", {"wikipedia","wikinews","wikitree","policy"})
COLLOQUIAL = SourceSetDomain("colloquial", {"NSMC","airbnb"})
ALL = UniverseDomain()


# ===== 지표 / 분류기 (고정 하이퍼파라미터로 공정 비교) =====
def macro_f1(yt, yp, n=3):
    yt, yp = np.asarray(yt), np.asarray(yp)
    f1s=[]
    for c in range(n):
        tp=int(((yp==c)&(yt==c)).sum()); fp=int(((yp==c)&(yt!=c)).sum()); fn=int(((yp!=c)&(yt==c)).sum())
        p=tp/(tp+fp) if tp+fp else 0.0; r=tp/(tp+fn) if tp+fn else 0.0
        f1s.append(2*p*r/(p+r) if p+r else 0.0)
    return float(np.mean(f1s))

class KNN:
    def __init__(self,k=15): self.k=k
    @staticmethod
    def _n(X): return X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-12)
    def fit(self,X,y): self._X=self._n(X.astype(np.float32)); self._y=y; return self
    def predict(self,X,batch=1024):
        Xn=self._n(X.astype(np.float32)); out=np.empty(len(Xn),np.int64)
        for i in range(0,len(Xn),batch):
            s=Xn[i:i+batch]@self._X.T
            tk=np.argpartition(-s,self.k-1,axis=1)[:,:self.k]
            for j,row in enumerate(tk): out[i+j]=np.bincount(self._y[row],minlength=3).argmax()
        return out

class NN:
    def __init__(self,hidden=128,lr=0.1,epochs=60,batch=64,l2=1e-4,mom=0.9,patience=10,seed=0):
        self.h,self.lr,self.ep,self.b,self.l2,self.mom,self.pat,self.seed=hidden,lr,epochs,batch,l2,mom,patience,seed
    def fit(self,X,y):
        rng=np.random.RandomState(self.seed)
        # 20% dev 분리(early stopping)
        idx=rng.permutation(len(X)); ndev=max(30,int(0.2*len(X)))
        dev,tr=idx[:ndev],idx[ndev:]
        self._mu=X[tr].mean(0,keepdims=True); self._sg=X[tr].std(0,keepdims=True)+1e-8
        Xt=(X[tr]-self._mu)/self._sg; yt=y[tr]; Xd=(X[dev]-self._mu)/self._sg; yd=y[dev]
        n,d=Xt.shape; Y=np.eye(3,dtype=np.float32)[yt]
        W1=(rng.randn(d,self.h)*np.sqrt(2/d)).astype(np.float32); b1=np.zeros((1,self.h),np.float32)
        W2=(rng.randn(self.h,3)*np.sqrt(2/self.h)).astype(np.float32); b2=np.zeros((1,3),np.float32)
        vW1=np.zeros_like(W1);vb1=np.zeros_like(b1);vW2=np.zeros_like(W2);vb2=np.zeros_like(b2)
        def fwd(Xx):
            z1=Xx@W1+b1; a1=np.maximum(z1,0); z2=a1@W2+b2
            z2=z2-z2.max(1,keepdims=True); e=np.exp(z2); p=e/e.sum(1,keepdims=True)
            return z1,a1,p
        best=-1; bestp=None; wait=0
        for ep in range(self.ep):
            o=rng.permutation(n)
            for s in range(0,n,self.b):
                ii=o[s:s+self.b]; xb=Xt[ii]; yb=Y[ii]; m=len(ii)
                z1,a1,p=fwd(xb)
                dz2=(p-yb)/m; dW2=a1.T@dz2+self.l2*W2; db2=dz2.sum(0,keepdims=True)
                dz1=(dz2@W2.T)*(z1>0); dW1=xb.T@dz1+self.l2*W1; db1=dz1.sum(0,keepdims=True)
                vW2=self.mom*vW2-self.lr*dW2; vb2=self.mom*vb2-self.lr*db2
                vW1=self.mom*vW1-self.lr*dW1; vb1=self.mom*vb1-self.lr*db1
                W2+=vW2;b2+=vb2;W1+=vW1;b1+=vb1
            _,_,pd=fwd(Xd); f=macro_f1(yd,pd.argmax(1))
            if f>best: best=f; bestp=(W1.copy(),b1.copy(),W2.copy(),b2.copy()); wait=0
            else:
                wait+=1
                if wait>=self.pat: break
        self._W1,self._b1,self._W2,self._b2=bestp
        return self
    def predict(self,X):
        Xx=(X-self._mu)/self._sg
        z1=Xx@self._W1+self._b1; a1=np.maximum(z1,0); z2=a1@self._W2+self._b2
        return z2.argmax(1)


# ===== 전체 train 임베딩 (서브샘플 재사용) =====
print("데이터 로드...", flush=True)
ds=load_dataset("klue/klue","nli")
train,val=ds["train"],ds["validation"]
tr_src=np.array(train["source"]); tr_y=np.array(train["label"],np.int64)
val_src=np.array(val["source"]); val_y=np.load("emb/val_y.npy"); val_X=np.load("emb/val_X.npy")

if os.path.exists("emb/train_all_X.npy"):
    tr_X=np.load("emb/train_all_X.npy")
    print("전체 train 임베딩 캐시 로드:", tr_X.shape, flush=True)
else:
    from sentence_transformers import SentenceTransformer
    sb=SentenceTransformer("jhgan/ko-sroberta-multitask",device="mps")
    print("전체 train 임베딩 중 (~4분)...", flush=True)
    u=sb.encode(list(train["premise"]),batch_size=128,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
    v=sb.encode(list(train["hypothesis"]),batch_size=128,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
    tr_X=np.concatenate([u,v,np.abs(u-v),u*v],-1).astype(np.float32)
    np.save("emb/train_all_X.npy",tr_X)
    print("저장:", tr_X.shape, flush=True)

# 풀별·클래스별 인덱스
def pool_by_class(mask):
    return {c: np.where(mask & (tr_y==c))[0] for c in [0,1,2]}
POOLS = {
    "formal": pool_by_class(np.array([s in FORMAL.sources for s in tr_src])),
    "colloquial": pool_by_class(np.array([s in COLLOQUIAL.sources for s in tr_src])),
    "all": pool_by_class(np.ones(len(tr_src),bool)),
}

def sample_balanced(pool_name, N, seed):
    rng=np.random.RandomState(1000+seed)
    per=N//3; idx=[]
    for c in [0,1,2]:
        p=POOLS[pool_name][c]
        idx.extend(rng.choice(p, size=per, replace=False))
    rng.shuffle(idx)
    return np.array(idx)


# ===== 스윕 =====
N_GRID=[500,1000,2000,4000,6000]
SEEDS=list(range(8))
TARGETS=[FORMAL, COLLOQUIAL]
targeted, broad = TargetedStrategy(), BroadStrategy(ALL)
val_mask={t.name: t.test_mask(val_src) for t in TARGETS}

def run_model(make, name):
    print("\n"+"="*78, flush=True)
    print(f"모델: {name}   (시드 {len(SEEDS)}개, paired t-검정)", flush=True)
    print("="*78, flush=True)
    res={}
    for target in TARGETS:
        mask=val_mask[target.name]
        print(f"\n[목표={target.name}]  (macro-F1, 목표 슬라이스 {int(mask.sum())}건)", flush=True)
        print(f"  {'N':>5} | {'targeted':>14} | {'broad':>14} | {'diff(t-b)':>16} | t  | 유의", flush=True)
        for N in N_GRID:
            tf=[]; bf=[]
            for sd in SEEDS:
                ti=sample_balanced(targeted.pool(target).name, N, sd)
                bi=sample_balanced(broad.pool(target).name, N, sd)
                tf.append(macro_f1(val_y[mask], make(sd).fit(tr_X[ti],tr_y[ti]).predict(val_X)[mask]))
                bf.append(macro_f1(val_y[mask], make(sd).fit(tr_X[bi],tr_y[bi]).predict(val_X)[mask]))
            tf=np.array(tf); bf=np.array(bf); d=tf-bf
            K=len(d); sd_d=d.std(ddof=1)
            t=d.mean()/(sd_d/math.sqrt(K)) if sd_d>0 else float('inf')
            sig="**" if abs(t)>2.36 else ("*" if abs(t)>1.9 else "")  # df=7: t.975≈2.36
            res[f"{name}/{target.name}/N{N}"]={
                "targeted_mean":float(tf.mean()),"targeted_std":float(tf.std()),
                "broad_mean":float(bf.mean()),"broad_std":float(bf.std()),
                "diff_mean":float(d.mean()),"diff_std":float(sd_d),"t":float(t),"win_targeted":int((d>0).sum())}
            print(f"  {N:>5} | {tf.mean():.4f}±{tf.std():.3f} | {bf.mean():.4f}±{bf.std():.3f} | "
                  f"{d.mean():+.4f}±{sd_d:.3f} | {t:+.1f} | {sig}", flush=True)
    return res

results={}
results.update(run_model(lambda sd: KNN(k=15), "KNN"))
results.update(run_model(lambda sd: NN(seed=sd), "NN"))
json.dump(results, open("results_significance.json","w"), ensure_ascii=False, indent=2)
print("\n** = p<0.05 (|t|>2.36, df=7),  * = p<0.1.  diff>0 이면 targeted 우세.", flush=True)
print("→ results_significance.json 저장 완료", flush=True)
