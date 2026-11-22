# Experiment (negative result): see README of findings in chat. Run with PYTHONPATH=. python experiments/<file>.py
import sys, time, math, numpy as np, torch
from ovi.reservoir import Reservoir
from ovi.ngram import NGram
N=400000; D=512
b=torch.from_numpy(np.load("data/books.npy").astype(np.int64))
tr=b[:N]; val=b[-30000:][:5000]
r=Reservoir(D=D); B=200; T=len(tr)//B
xs,ys=[],[]
r.run(tr[:B*T].view(B,T),16,lambda X,y:(xs.append(X),ys.append(y)))
X0=torch.cat(xs); Y=torch.cat(ys)
vx,vy=[],[]
r.run(val.view(10,500),16,lambda X_,y:(vx.append(X_),vy.append(y))); VX0=torch.cat(vx); VY=torch.cat(vy)
# global positions of each val target (stream-major then time order interleaved by batches of 64 steps -> recompute)
pos=[]
def order():
    # run() emits blocks of up to 64 timesteps, each block time-major: [t0 all streams, t1 all streams,...]
    out=[]; t=16; T_=500
    ts=list(range(16,T_-1))
    for s in range(0,len(ts),64):
        for tt in ts[s:s+64]:
            for j in range(10): out.append(j*500+tt+1)
    return out
pos=order(); assert len(pos)==len(VY)
assert all(val[p].item()==VY[i].item() for i,p in enumerate(pos[:2000]))
ng=NGram(3); ng.add(tr.tolist())
vl=val.tolist()
png=torch.tensor([ng.prob(vl[max(0,p-2):p], vl[p]) for p in pos])
print("ngram(400k) alone:", -torch.log(png).mean().item())
uni=torch.bincount(Y,minlength=4096).float()+1; uni/=uni.sum()
fast=r.leak>0.3
for name,msk in (("all",torch.ones(D,dtype=torch.bool)),("fast",fast)):
    mu=X0[:,msk].mean(0)
    K=torch.nn.functional.normalize(X0[:,msk]-mu,dim=1); Q=torch.nn.functional.normalize(VX0[:,msk]-mu,dim=1)
    v,i=(Q@K.T).topk(64,dim=1); yy=Y[i]
    for tau in (10,30):
        p=torch.zeros(len(Q),4096).scatter_add_(1,yy,torch.softmax(v*tau,1))
        pk=p[torch.arange(len(VY)),VY]
        alone=-torch.log(0.6*pk+0.4*uni[VY]).mean().item()
        best=min((-torch.log(l*pk+(1-l)*png).mean().item(),l) for l in (0.1,0.2,0.3,0.5))
        print(f"{name} tau{tau}: knn alone {alone:.3f} | mix with ngram best {best[0]:.3f} at lam {best[1]}")
