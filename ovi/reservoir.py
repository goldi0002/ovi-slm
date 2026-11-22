"""Multi-timescale Reservoir LM (Ovi's own architecture).

  * Core: a FIXED random recurrent network. Every unit has its own leak rate (log-spaced), so the
    state holds the last few tokens (fast units) AND the gist of the whole sentence (slow units).
  * Learned part: ONE linear readout solved in closed form (ridge regression). No backprop, no GPU.
  * Learning new text = accumulate two small matrices and re-solve.
"""
import math, time, numpy as np, torch

torch.set_num_threads(4)


class Reservoir:
    def __init__(self, V=4096, D=768, K=256, rho=0.9, density=0.1, leak=(0.03, 1.0), seed=0):
        g = torch.Generator().manual_seed(seed)
        self.V, self.D, self.K = V, D, K
        self.E = torch.randn(V, K, generator=g) / math.sqrt(K)   # fixed random token code (targets)
        self.Win = torch.randn(V, D, generator=g) * 0.6          # token -> input drive (fixed)
        W = torch.randn(D, D, generator=g) * (torch.rand(D, D, generator=g) < density)
        W *= rho / torch.linalg.eigvals(W).abs().max()           # echo-state property
        self.W = W
        self.leak = torch.exp(torch.empty(D).uniform_(math.log(leak[0]), math.log(leak[1]), generator=g))
        self.A = torch.zeros(D + 1, D + 1, dtype=torch.float64)  # sum x x^T
        self.B = torch.zeros(D + 1, K, dtype=torch.float64)      # sum x e(y)^T
        self.Wout = None

    def run(self, streams, warm=16, collect=None):
        """streams: LongTensor (B, T). Calls collect(X, y) with states predicting the NEXT token."""
        B, T = streams.shape
        s = torch.zeros(B, self.D)
        buf_x, buf_y = [], []
        for t in range(T - 1):
            s = (1 - self.leak) * s + self.leak * torch.tanh(self.Win[streams[:, t]] + s @ self.W)
            if t >= warm:
                buf_x.append(s); buf_y.append(streams[:, t + 1])
            if len(buf_x) == 64 or t == T - 2:
                if buf_x:
                    collect(torch.cat(buf_x), torch.cat(buf_y)); buf_x, buf_y = [], []

    def _acc(self, X, y):
        X = torch.cat([X, torch.ones(len(X), 1)], 1)
        self.A += (X.T @ X).double()
        self.B += X.T.double() @ self.E[y].double()

    def learn(self, streams, warm=16):
        self.run(streams, warm, self._acc)

    def solve(self, lam=1.0):
        A = self.A + lam * torch.eye(self.D + 1, dtype=torch.float64)
        self.Wout = torch.linalg.solve(A, self.B).float()

    def logits(self, x):
        return (torch.cat([x, torch.ones(len(x), 1)], 1) @ self.Wout) @ self.E.T

    def eval_loss(self, streams, taus=(10, 20, 40, 80), warm=16):
        xs, ys = [], []
        self.run(streams, warm, lambda X, y: (xs.append(X), ys.append(y)))
        X, y = torch.cat(xs), torch.cat(ys)
        lg = self.logits(X)
        return {tau: torch.nn.functional.cross_entropy(lg * tau, y).item() for tau in taus}
