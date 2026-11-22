# Experiment (negative result): see README of findings in chat. Run with PYTHONPATH=. python experiments/<file>.py
import sys, time, numpy as np, torch
from ovi.reservoir import Reservoir
D=int(sys.argv[1]); N=int(sys.argv[2])
b=torch.from_numpy(np.load("data/books.npy").astype(np.int64))
val=b[-30000:]; tr=b[:N]
r=Reservoir(D=D); B=200; T=len(tr)//B
t=time.time(); r.learn(tr[:B*T].view(B,T)); print("learn",round(time.time()-t,1),"s")
vs=val[:20*1500].view(20,1500)
xs,ys=[],[]
r.run(vs,16,lambda X,y:(xs.append(X),ys.append(y))); X,y=torch.cat(xs),torch.cat(ys)
for lam in (1,100):
    r.solve(lam); lg=r.logits(X)
    print("lam",lam,{tau:round(torch.nn.functional.cross_entropy(lg*tau,y).item(),3) for tau in (5,10,20,40,80)})
