"""Logistic regression and gradient boosted trees, written from scratch with numpy/scipy."""
import numpy as np
from scipy.optimize import minimize


def auc(y, s):
    r = np.argsort(np.argsort(s, kind="mergesort"), kind="mergesort") + 1.0
    # average ranks for ties
    order = np.argsort(s, kind="mergesort"); ss = s[order]
    ranks = np.empty(len(s)); i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    n1 = y.sum(); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


class Logit:
    def __init__(self, l2=1.0):
        self.l2 = l2

    def fit(self, X, y):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-9
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        pen = np.r_[0, np.ones(Z.shape[1] - 1)] * self.l2

        def f(w):
            p = sigmoid(Z @ w)
            ll = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
            g = Z.T @ (p - y) / len(y) + pen * w / len(y)
            return ll + 0.5 * (pen * w * w).sum() / len(y), g
        self.w = minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
        return self

    def predict(self, X):
        return sigmoid(np.c_[np.ones(len(X)), (X - self.mu) / self.sd] @ self.w)


class GBM:
    """Binary gradient boosting with histogram trees (second-order, like XGBoost's objective)."""

    def __init__(self, n_trees=200, depth=3, lr=0.1, bins=32, lam=10.0, min_leaf=50):
        self.n, self.d, self.lr, self.B, self.lam, self.ml = n_trees, depth, lr, bins, lam, min_leaf

    def _bin(self, X):
        out = np.empty(X.shape, dtype=np.int16)
        for j in range(X.shape[1]):
            out[:, j] = np.searchsorted(self.edges[j], X[:, j], side="right")
        return out

    def _grow(self, Xb, g, h, idx, depth):
        G, H = g[idx].sum(), h[idx].sum()
        leaf = ("leaf", -G / (H + self.lam))
        if depth == 0 or len(idx) < 2 * self.ml:
            return leaf
        best = (0.0, None, None)
        base = G * G / (H + self.lam)
        for j in range(Xb.shape[1]):
            b = Xb[idx, j]
            gh = np.bincount(b, g[idx], self.B + 1).cumsum()
            hh = np.bincount(b, h[idx], self.B + 1).cumsum()
            cn = np.bincount(b, minlength=self.B + 1).cumsum()
            GL, HL = gh[:-1], hh[:-1]; GR, HR = G - GL, H - HL
            ok = (cn[:-1] >= self.ml) & (len(idx) - cn[:-1] >= self.ml)
            gain = GL ** 2 / (HL + self.lam) + GR ** 2 / (HR + self.lam) - base
            gain[~ok] = 0
            k = gain.argmax()
            if gain[k] > best[0]:
                best = (gain[k], j, k)
        if best[1] is None:
            return leaf
        _, j, k = best
        m = Xb[idx, j] <= k
        return ("split", j, k, self._grow(Xb, g, h, idx[m], depth - 1), self._grow(Xb, g, h, idx[~m], depth - 1))

    def _pred_tree(self, t, Xb):
        out = np.zeros(len(Xb))
        def rec(node, rows):
            if node[0] == "leaf":
                out[rows] = node[1]; return
            m = Xb[rows, node[1]] <= node[2]
            rec(node[3], rows[m]); rec(node[4], rows[~m])
        rec(t, np.arange(len(Xb)))
        return out

    def fit(self, X, y):
        self.edges = [np.unique(np.quantile(X[:, j], np.linspace(0, 1, self.B + 1)[1:-1])) for j in range(X.shape[1])]
        Xb = self._bin(X)
        p0 = y.mean(); self.b0 = np.log(p0 / (1 - p0))
        F = np.full(len(y), self.b0); self.trees = []
        for _ in range(self.n):
            p = sigmoid(F); g = p - y; h = p * (1 - p)
            t = self._grow(Xb, g, h, np.arange(len(y)), self.d)
            F += self.lr * self._pred_tree(t, Xb); self.trees.append(t)
        return self

    def predict(self, X):
        Xb = self._bin(X); F = np.full(len(X), self.b0)
        for t in self.trees:
            F += self.lr * self._pred_tree(t, Xb)
        return sigmoid(F)
