"""A small MLP with Adam, in NumPy. Ported from the family unchanged.

It is the member of the residual ensemble. Hidden SiLU layers, linear
output, early stopping on a validation split with the best weights kept.
"""

from __future__ import annotations

import math

import numpy as np


class Adam:
    def __init__(self, params, lr=2e-3):
        self.params = list(params)
        self.lr = lr
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        b1 = 1.0 - 0.9 ** self.t
        b2 = 1.0 - 0.999 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m[:] = 0.9 * m + 0.1 * g
            v[:] = 0.999 * v + 0.001 * g ** 2
            p -= self.lr * (m / b1) / (np.sqrt(v / b2) + 1e-8)


class MLP:
    def __init__(self, d_in, d_out, hidden=(64, 64), seed=0):
        r = np.random.default_rng(seed)
        sizes = [d_in, *hidden, d_out]
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            sc = math.sqrt(2.0 / (sizes[i] + sizes[i + 1]))
            self.W.append(r.normal(0, sc, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))

    @staticmethod
    def _act(z):                       # SiLU
        s = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return z * s

    @staticmethod
    def _act_grad(z):
        s = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return s * (1 + z * (1 - s))

    def forward(self, X, cache=False):
        a, acts, zs = X, [X], []
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            zs.append(z)
            a = self._act(z) if i < len(self.W) - 1 else z
            acts.append(a)
        return (a, acts, zs) if cache else a

    def _backward(self, acts, zs, out, yb, wd):
        delta = 2.0 * (out - yb) / out.shape[0]
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta + wd * self.W[i]
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * self._act_grad(zs[i - 1])
        return gW, gb

    def train(self, X, Y, Xv, Yv, epochs=400, lr=3e-3, batch=32,
              weight_decay=1e-5, patience=40, seed=0):
        r = np.random.default_rng(seed)
        opt = Adam(self.W + self.b, lr=lr)
        best_val, best_state, since = np.inf, None, 0
        n = X.shape[0]
        for _ep in range(epochs):
            idx = r.permutation(n)
            for s in range(0, n, batch):
                bi = idx[s:s + batch]
                out, acts, zs = self.forward(X[bi], cache=True)
                gW, gb = self._backward(acts, zs, out, Y[bi], weight_decay)
                opt.step(gW + gb)
            val = float(np.mean((self.forward(Xv) - Yv) ** 2))
            if val < best_val - 1e-7:
                best_val, since = val, 0
                best_state = ([w.copy() for w in self.W],
                              [b.copy() for b in self.b])
            else:
                since += 1
                if since >= patience:
                    break
        if best_state:
            self.W, self.b = best_state
        return best_val
