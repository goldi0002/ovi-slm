# Experiment (negative result): see README of findings in chat. Run with PYTHONPATH=. python experiments/<file>.py
import time, math, numpy as np, torch
from ovi.reservoir import Reservoir
from ovi.ngram import NGram
N=400000; D=512; V=4096
b=torch.from_numpy(np.load("data/books.npy").astype(np.int64))
tr=b[:N]; val=b[-30000:][:5000]
# --- PPMI + SVD embeddings from ALL train-side text excluding val (closed form, no gradients)
t=time.time()
allb=b[:-30000]
C=torch.zeros(V,V)
for d in range(1,5):
    C.index_put_((allb[:-d],allb[d:]),torch.ones(len(allb)-d),accumulate=True)
C=C+C.T
rs=C.sum(1,keepdim=True); cs=C.sum(0,keepdim=True); tot=C.sum()
pmi=torch.log((C*tot/(rs*cs+1e-9)).clamp(min=1e-9)).clamp(min=0).nan_to_num()
U,S,_=torch.svd_lowrank(pmi,q=128,niter=4)
E=U*S.sqrt(); E=E/E.std()
print("ppmi+svd",round(time.time()-t,1),"s")
def run(Win_override):
    r=Reservoir(D=D)
    if Win_override is not None: r.Win=Win_override
    B=200;T=len(tr)//B; xs,ys=[],[]
    r.run(tr[:B*T].view(B,T),16,lambda X,y:(xs.append(X),ys.append(y)))
    X0=torch.cat(xs);Y=torch.cat(ys); vx,vy=[],[]
    r.run(val.view(10,500),16,lambda X_,y:(vx.append(X_),vy.append(y)))
    return X0,Y,torch.cat(vx),torch.cat(vy)
g=torch.Generator().manual_seed(1)
mix=torch.randn(128,D,generator=g)/math.sqrt(128)
for name,W in (("random",None),("ppmi-svd",0.6*(E@mix))):
    X0,Y,VX0,VY=run(W)
    uni=torch.bincount(Y,minlength=V).float()+1; uni/=uni.sum()
    mu=X0.mean(0); K=torch.nn.functional.normalize(X0-mu,dim=1); Q=torch.nn.functional.normalize(VX0-mu,dim=1)
    v,i=(Q@K.T).topk(64,dim=1); yy=Y[i]
    for tau in (10,30):
        p=torch.zeros(len(Q),V).scatter_add_(1,yy,torch.softmax(v*tau,1))
        pk=p[torch.arange(len(VY)),VY]
        print(f"{name} tau{tau}: knn+uni {-torch.log(0.6*pk+0.4*uni[VY]).mean().item():.3f}")
print("(n-gram on same 400k: 6.313)")
