"""CrossPMI reply model -- Ovi's own CPU-trained "dual encoder".

Learned from (message -> reply) pairs by counting + one SVD (no gradients, no GPU):
  1. count how often message-word w co-occurs with reply-word v across all pairs
  2. PMI(w, v) = log P(w, v) / (P(w) P(v))   (positive part; smoothed reply marginals)
  3. truncated SVD:  PMI ~ U S V^T   ->  a message vector (from U) and a reply vector (from V)
score(message, reply) = cosine of the two vectors, so ALL stored replies can be ranked with one matmul.
It learns e.g. "puppy" -> "cute", "failed exam" -> "sorry", "tired" -> "rest".
"""
import math, re
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from .memory import toks

STOP = set("a an the and or of to in on at for with from as is are was were be am do does did i you he she it we they me my your his her our their this that there".split())


def words(s):
    return [w for w in toks(s)]


class CrossPMI:
    def __init__(self, dim=128, vocab=7000, alpha=0.75):
        self.dim, self.vmax, self.alpha = dim, vocab, alpha

    def fit(self, pairs):
        from collections import Counter
        df = Counter()
        for q, a, *_ in pairs:
            df.update(set(words(q)) | set(words(a)))
        self.vocab = {w: i for i, (w, _) in enumerate(df.most_common(self.vmax))}
        V = len(self.vocab); N = len(pairs)
        def inc(texts):
            r, c = [], []
            for i, t in enumerate(texts):
                for w in set(words(t)):
                    j = self.vocab.get(w)
                    if j is not None:
                        r.append(i); c.append(j)
            return sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(texts), V))
        M, R = inc([p[0] for p in pairs]), inc([p[1] for p in pairs])
        C = (M.T @ R).toarray()                       # V x V co-occurrence counts
        pm = np.asarray(M.sum(0)).ravel(); pr = np.asarray(R.sum(0)).ravel() ** self.alpha
        pr = pr / pr.sum(); pm = pm / pm.sum()
        P = C / C.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(P / (pm[:, None] * pr[None, :]))
        pmi = np.nan_to_num(pmi, neginf=0, posinf=0); pmi[pmi < 0] = 0
        pmi[C < 2] = 0                                # ignore one-off co-occurrences (noise)
        U, S, Vt = svds(sp.csr_matrix(pmi.astype(np.float32)), k=self.dim)
        o = np.argsort(-S); U, S, Vt = U[:, o], S[o], Vt[o]
        self.Wm = (U * np.sqrt(S)).astype(np.float32)
        self.Wr = (Vt.T * np.sqrt(S)).astype(np.float32)
        # idf weights so rare words matter more than "the"
        self.idf = np.zeros(V, np.float32)
        for w, i in self.vocab.items():
            self.idf[i] = math.log((N + 1) / (df[w] + 1)) * (0.3 if w in STOP else 1.0)
        return self

    def _embed(self, text, W):
        ids = [self.vocab[w] for w in set(words(text)) if w in self.vocab]
        if not ids:
            return np.zeros(self.dim, np.float32)
        v = (W[ids] * self.idf[ids, None]).sum(0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def msg_vec(self, text):
        return self._embed(text, self.Wm)

    def reply_matrix(self, replies):
        return np.stack([self._embed(r, self.Wr) for r in replies])

    def scores(self, msg, reply_mat):
        return reply_mat @ self.msg_vec(msg)


# ---------------- learned ranker over reply-fit features (logistic regression, numpy, CPU) ----------------
NEG = set("sad tired lost failed fail hate angry upset sick hurt cry crying lonely stress stressed worried scared afraid bad terrible awful died death broke broken sorry miss missed".split())
POS = set("happy great good excited love loved awesome amazing wonderful proud glad won passed new yay congrats nice fun".split())
SORRY = set("sorry sad hard tough rough sucks awful terrible hope okay feel understand".split())
GLAD = set("great wonderful congratulations congrats glad happy awesome nice fantastic amazing cool fun".split())
NC = 5
NFEAT = 13 + NC


class Ranker:
    def __init__(self, cross):
        self.cross = cross
        self.w = np.zeros(NFEAT, np.float32)

    def prep_pool(self, replies):
        """Per-reply features that don't depend on the message."""
        from collections import Counter
        if not hasattr(self, "rdf"):
            self.rdf = Counter(w for r in replies for w in set(words(r)))
        out = []
        for r in replies:
            ws = words(r); s = set(ws)
            gen = 0.0   # NOT df-based: a stored reply counts itself in the df table, a new one doesn't -> it leaks the answer in evaluations
            out.append((r.strip().endswith("?"), math.log(1 + len(ws)), gen, bool(s & SORRY), bool(s & GLAD), s))
        return out

    def feats(self, msg, rvecs, rinfo, hist=None):
        """rvecs: (K, d) reply vectors; rinfo: list from prep_pool.
        hist: the last turns INCLUDING msg (hist[-1] == msg). Returns (K, NFEAT)."""
        mw = set(words(msg)); mq = msg.strip().endswith("?"); mneg = bool(mw & NEG); mpos = bool(mw & POS)
        cos = rvecs @ self.cross.msg_vec(msg)
        cmsg = {w for w in mw if w not in STOP and len(w) > 1}
        F = np.zeros((len(rinfo), 13), np.float32)
        for k, (rq, ln, gen, sorry, glad, rs) in enumerate(rinfo):
            ov = len(cmsg & rs)
            F[k] = [cos[k], ov, ov / (1 + len(rs)), rq and not mq, rq and mq, (not rq) and mq, ln, gen,
                    mneg and sorry, mpos and glad, mneg and glad, mpos and sorry, 1.0]
        return np.hstack([F[:, :-1], self.ctx_feats(hist or [msg], rvecs, rinfo), F[:, -1:]])

    def ctx_feats(self, hist, rvecs, rinfo):
        """How well does each reply fit the conversation so far (not just the last message)?"""
        cc = rvecs @ self.cross._embed(" ".join(hist[-3:]), self.cross.Wm)
        cwd = lambda t: {w for w in words(t) if w not in STOP and len(w) > 1}
        prev = hist[:-1]
        p1 = cwd(prev[-1]) if len(prev) >= 1 else set(); p2 = cwd(prev[-2]) if len(prev) >= 2 else set()
        out = np.zeros((len(rinfo), NC), np.float32)
        for k, info in enumerate(rinfo):
            rs = info[5]
            out[k] = [cc[k], len(p1 & rs), len(p2 & rs), len(p1 & rs) / (1 + len(rs)), float(len(p1 & rs) > 0)]
        return out

    def fit_logreg(self, X, y, iters=300, lr=0.5, l2=1e-3):
        mu, sd = X[:, :-1].mean(0), X[:, :-1].std(0) + 1e-6
        self.mu, self.sd = mu, sd
        Z = np.hstack([(X[:, :-1] - mu) / sd, X[:, -1:]])
        w = np.zeros(Z.shape[1], np.float32)
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(Z @ w)))
            w -= lr * (Z.T @ (p - y) / len(y) + l2 * w)
        self.w = w
        return self

    def score(self, F):
        Z = np.hstack([(F[:, :-1] - self.mu) / self.sd, F[:, -1:]])
        return Z @ self.w


def _fit_listwise(self, groups, iters=400, lr=0.3, l2=1e-3):
    """Softmax over each candidate group; the real reply is index 0 of every group."""
    allX = np.vstack(groups)
    self.mu, self.sd = allX[:, :-1].mean(0), allX[:, :-1].std(0) + 1e-6
    Zs = [np.hstack([(X[:, :-1] - self.mu) / self.sd, X[:, -1:]]) for X in groups]
    w = np.zeros(Zs[0].shape[1], np.float32)
    for _ in range(iters):
        g = np.zeros_like(w)
        for Z in Zs:
            s = Z @ w; s -= s.max(); p = np.exp(s); p /= p.sum()
            g += Z.T @ p - Z[0]                     # d(-log p_gold)/dw
        w -= lr * (g / len(Zs) + l2 * w)
    self.w = w
    return self


Ranker.fit_listwise = _fit_listwise


def save(cross, rk, reply_vecs, path="ovi_rank.npz"):
    np.savez_compressed(path, Wm=cross.Wm.astype(np.float16), Wr=cross.Wr.astype(np.float16), idf=cross.idf,
                        vocab=np.array(list(cross.vocab.keys())), dim=cross.dim, w=rk.w, mu=rk.mu, sd=rk.sd,
                        rv=reply_vecs.astype(np.float16))


def load(path="ovi_rank.npz"):
    z = np.load(path, allow_pickle=False)
    c = CrossPMI(dim=int(z["dim"]))
    c.vocab = {w: i for i, w in enumerate(z["vocab"].tolist())}
    c.Wm, c.Wr, c.idf = z["Wm"].astype(np.float32), z["Wr"].astype(np.float32), z["idf"]
    rk = Ranker(c); rk.w, rk.mu, rk.sd = z["w"], z["mu"], z["sd"]
    return rk, z["rv"].astype(np.float32)


def build(path="ovi_rank.npz"):
    """Fit CrossPMI on train conversations -> train the ranker on validation conversations (never seen by CrossPMI)
    -> refit CrossPMI on ALL shipped pairs and store its reply vectors. ~30 s on a laptop CPU."""
    import random
    from . import dialogue as D, dataset
    from .retrieval import Index
    random.seed(0); np.random.seed(0)
    train = D.dedupe(D.dailydialog("train") + D.empathetic("train"))
    cross = CrossPMI().fit(train)
    reps = [a for _, a, _ in train]; RM = cross.reply_matrix(reps)
    rk = Ranker(cross); info = rk.prep_pool(reps)
    ix = Index()
    for q, a, _ in train: ix.add(q, a)
    _WORD, _COMMON = D._WORD, D._COMMON
    from collections import Counter
    rdf = Counter(w for r in reps for w in set(words(r)))
    va = [t for t in D.triples("validation", "daily") + D.triples("valid", "emp")      # real replies must pass the same
          if all(rdf.get(w, 0) >= 5 for w in _WORD.findall(t[1].lower()) if w not in _COMMON)]   # filter as the pool

    groups = []
    for hist, gold in random.sample(va, min(len(va), 6000)):
        top = [i for i, _ in ix.topk(hist[-1], 32) if reps[i] != gold][:29]
        if len(top) < 20:
            continue
        rv = np.vstack([cross._embed(gold, cross.Wr)[None], RM[top]]); ri = rk.prep_pool([gold]) + [info[i] for i in top]
        groups.append(rk.feats(hist[-1], rv, ri, hist))
    rk.fit_listwise(groups)
    pairs = [(q, a) for k, q, a in dataset.load() if k == "d"]       # exactly the dialogue pairs Ovi ships, in order
    final = CrossPMI().fit(pairs)
    rk.cross = final
    save(final, rk, final.reply_matrix([a for _, a in pairs]), path)
    import os
    print(f"ranker trained on {len(groups)} validation turns; wrote {path} ({os.path.getsize(path)/1e6:.1f} MB)")


if __name__ == "__main__":
    build()
