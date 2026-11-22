# Experiment (negative result): see README of findings in chat. Run with PYTHONPATH=. python experiments/<file>.py
import sys, time, numpy as np, torch
from ovi.reservoir import Reservoir
D=int(sys.argv[1]); N=int(sys.argv[2])
b=torch.from_numpy(np.load("data/books.npy").astype(np.int64))
val=b[-30000:][:10*500]; tr=b[:N]
r=Reservoir(D=D); B=200; T=len(tr)//B
xs,ys=[],[]
t=time.time(); r.run(tr[:B*T].view(B,T),16,lambda X,y:(xs.append(X),ys.append(y)))
X=torch.cat(xs); Y=torch.cat(ys); X=torch.nn.functional.normalize(X-X.mean(0),dim=1); mu=torch.cat(xs).mean(0)
print("states",X.shape,round(time.time()-t,1),"s")
vx,vy=[],[]
r.run(val.view(10,500),16,lambda X_,y:(vx.append(X_),vy.append(y))); VX=torch.nn.functional.normalize(torch.cat(vx)-mu,dim=1); VY=torch.cat(vy)
uni=torch.bincount(Y,minlength=4096).float()+1; uni/=uni.sum()
t=time.time(); S=VX@X.T; print("sims",round(time.time()-t,1),"s")
for k in (16,64):
    v,i=S.topk(k,dim=1); yy=Y[i]
    for tau in (10,30,60):
        w=torch.softmax(v*tau,1)
        p=torch.zeros(len(VX),4096).scatter_add_(1,yy,w)
        for lam in (0.6,0.9):
            pr=lam*p+(1-lam)*uni
            print(f"k{k} tau{tau} lam{lam}: {-torch.log(pr[torch.arange(len(VY)),VY]).mean().item():.3f}")
