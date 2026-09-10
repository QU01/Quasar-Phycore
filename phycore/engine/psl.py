"""Phy-2's layer 3: a learned, amortised Pareto front. Needs torch.

    critic  = anchored deep ensemble on the L1-L0 residual, plus a
              cheap-physics surrogate with a violation head when the L0
              is not differentiable (section 7.5 of the report);
    actor   = pi(theta | spec, lambda): a preference-conditioned Gaussian
              on the pre-sigmoid box, trained by PATHWISE gradients through
              the critic (and through the exact L0 when it is torch or
              JAX via ``backend.torch_from_jax``);
    signal  = augmented Tchebycheff of the PESSIMISTIC objectives + linear
              barrier on predicted g + entropy bonus + KL trust region to
              the previous actor + Stein repulsion between preferences +
              likelihood of the exact-L0 elites (the CEM anchor);
    batch   = greedy hypervolume gain on the pessimistic predictions over
              the verified front, exact L0 filtering feasibility FIRST,
              explore slots for the widest bands and for what the support
              gate rejected, frontier slots at g ~ 0;
    front   = only real expensive evaluations, as in the family.

Ported from Phy-Bench ``psl.py`` (v4, the version that won the three
tracks) onto the plugin contract. Three pieces the benchmark showed are
not optional: the elite anchor, the bounded actor mean, and a cheap
critic trained on the CURRENT spec's data.

CUDA: ``PSLConfig.device="auto"`` puts every tensor on the GPU when torch
sees one. JIT: ``PSLConfig.compile=True`` runs the actor body and the
critic members through ``torch.compile`` (guarded - a machine without a
working inductor keeps eager mode and says so in ``PSLState.jit``).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ..backend import compile_module, compile_status, torch_device
from ..plugin import (EVALUATION_ERRORS, fidelity_label, plugin_cell, plugin_label,
                      plugin_violation, reached, repair, spec_vector)
from .dominance import Individual
from .hypervolume import hypervolume
from .sampling import lhs


# =============================================================================
# 1. Networks
# =============================================================================


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=(64, 64)):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AnchoredEnsemble:
    """K MLPs, each with a frozen random prior network (Osband 2018).

    Diversity outside the data comes from the priors, not only from the
    bootstrap - which is where the bootstrap ensemble of the family is
    weakest. Conformal quantile on a held-out split turns the spread into
    a band with finite-sample coverage. Differentiable w.r.t. its input.
    """

    def __init__(self, d_in, d_out, k=5, hidden=(64, 64), prior_scale=0.5,
                 seed=0, device="cpu", compile_nets: bool = False):
        g = torch.Generator().manual_seed(seed)
        self.k, self.device = k, device
        self.nets = [MLP(d_in, d_out, hidden).to(device) for _ in range(k)]
        self.priors = []
        for _ in range(k):
            p = MLP(d_in, d_out, hidden).to(device)
            for prm in p.parameters():
                prm.requires_grad_(False)
                with torch.no_grad():
                    prm.copy_(torch.randn(prm.shape, generator=g).to(device)
                              * 0.6 / math.sqrt(max(prm.shape[-1], 1)))
            self.priors.append(p)
        if compile_nets:
            sample = torch.zeros(4, d_in, device=device)
            self.nets = [compile_module(n, sample=sample) for n in self.nets]
        self.prior_scale = prior_scale
        self.xm = self.xs = self.ym = self.ys = None
        self.q = torch.ones(d_out, device=device)
        self.metrics = {}
        self.seed = seed

    def _member(self, i, xn):
        return self.nets[i](xn) + self.prior_scale * self.priors[i](xn)

    def fit(self, X, Y, epochs=150, lr=3e-3, wd=1e-5, batch=256, calib_frac=0.2,
            seed=None):
        rng = np.random.default_rng(self.seed if seed is None else seed)
        X = torch.as_tensor(np.asarray(X, np.float32), device=self.device)
        Y = torch.as_tensor(np.asarray(Y, np.float32), device=self.device)
        if Y.ndim == 1:
            Y = Y[:, None]
        n = X.shape[0]
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-6
        self.ym, self.ys = Y.mean(0), Y.std(0) + 1e-6
        Xn, Yn = (X - self.xm) / self.xs, (Y - self.ym) / self.ys
        perm = rng.permutation(n)
        n_cal = int(calib_frac * n) if n >= 25 else 0
        ci, ti = perm[:n_cal], perm[n_cal:]
        for i in range(self.k):
            boot = rng.integers(0, len(ti), len(ti))
            idx = torch.as_tensor(ti[boot], device=self.device)
            opt = torch.optim.Adam(self.nets[i].parameters(), lr=lr, weight_decay=wd)
            m = len(idx)
            for _ep in range(epochs):
                p = torch.as_tensor(rng.permutation(m), device=self.device)
                for s in range(0, m, batch):
                    b = idx[p[s:s + batch]]
                    pred = self._member(i, Xn[b])
                    loss = ((pred - Yn[b]) ** 2).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
        with torch.no_grad():
            if n_cal >= 8:
                mu, sd = self._predict_n(Xn[ci])
                score = ((Yn[ci] - mu).abs() / (sd + 1e-6))
                qi = min(int(math.ceil(0.9 * (n_cal + 1))) - 1, n_cal - 1)
                self.q = torch.sort(score, 0).values[qi].clamp(0.5, 20.0)
                cov = ((Yn[ci] - mu).abs() <= self.q * sd).float().mean(0)
                self.metrics = {"n": n, "n_cal": n_cal, "coverage": cov.tolist(),
                                "q": self.q.tolist()}
            else:
                self.q = torch.full((Y.shape[1],), 2.0, device=self.device)
                self.metrics = {"n": n, "n_cal": n_cal}
        return self.metrics

    def _predict_n(self, Xn):
        preds = torch.stack([self._member(i, Xn) for i in range(self.k)])
        return preds.mean(0), preds.std(0) + 1e-6

    def predict(self, X, conformal=True):
        """Differentiable w.r.t. X. Returns (mu, sd) in ORIGINAL units."""
        Xn = (X - self.xm) / self.xs
        mu, sd = self._predict_n(Xn)
        if conformal:
            sd = sd * self.q
        return mu * self.ys + self.ym, sd * self.ys


class Actor(nn.Module):
    """pi(theta | spec, lambda) as a Gaussian on the pre-sigmoid box."""

    def __init__(self, spec_dim, n_obj, dim, hidden=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(spec_dim + n_obj, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU())
        self.mean = nn.Linear(hidden, dim)
        self.logstd = nn.Linear(hidden, dim)
        nn.init.zeros_(self.mean.weight); nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.logstd.weight)
        nn.init.constant_(self.logstd.bias, -1.0)      # sigma 0.37 on the pre-sigmoid axis

    def params(self, s, lam):
        h = self.body(torch.cat([s, lam], -1))
        # |mean| <= 4 on the pre-sigmoid axis: the sigmoid never saturates, so
        # the pathwise gradient never vanishes (a carried actor was dying here)
        m = 4.0 * torch.tanh(self.mean(h) / 4.0)
        # floor at sigma = 0.14 on the pre-sigmoid axis: a cloud, never a point
        return m, self.logstd(h).clamp(-2.0, 0.5)

    def forward(self, s, lam):
        return self.params(s, lam)

    def sample(self, s, lam, eps, temp=1.0):
        m, ls = self.params(s, lam)
        pre = m + temp * torch.exp(ls) * eps
        return torch.sigmoid(pre), m, ls

    @staticmethod
    def kl(m1, ls1, m2, ls2):
        v1, v2 = torch.exp(2 * ls1), torch.exp(2 * ls2)
        return (ls2 - ls1 + (v1 + (m1 - m2) ** 2) / (2 * v2) - 0.5).sum(-1)


# =============================================================================
# 2. Configuration and state (what survives from one spec to the next)
# =============================================================================


@dataclass
class Pools:
    cheap_X: list = field(default_factory=list)     # [thn, s]
    cheap_Y: list = field(default_factory=list)     # [F (log where positive), G, build]
    exp_X: list = field(default_factory=list)       # [thn, s, F_cheap_t]
    exp_r: list = field(default_factory=list)       # residual on f0
    exp_spec: list = field(default_factory=list)


@dataclass
class PSLConfig:
    k_ensemble: int = 5
    actor_steps: int = 400
    actor_batch: int = 256
    actor_lr: float = 2e-3
    kappa: float = 1.0
    w_pen: float = 5.0
    w_kl: float = 0.05
    w_ent: float = 0.03
    w_stein: float = 0.02
    w_elite: float = 1.0             # log-likelihood of the exact-L0 elites under the actor (CEM step)
    n_elite: int = 48
    n_candidates: int = 1200
    exploit_frac: float = 0.5
    explore_frac: float = 0.25
    replay_frac: float = 0.0         # the actor's memory is its weights; the critics carry the data
    min_dist: float = 0.03
    n_cheap_init: int = 600          # free L0 evaluations that seed the cheap critic
    feas_target: float = 0.20        # below this candidate feasibility, keep iterating cheaply
    inner_fixed: int = 2             # cheap inner iterations always taken (surrogate-L0 path)
    max_inner: int = 4
    exact_cheap: bool = True         # use the differentiable L0 when the plugin has one
    device: str = "auto"             # "auto" -> cuda when available
    compile: bool = False            # torch.compile on the actor body and critic members
    threads: int | None = None       # torch.set_num_threads, None = leave alone


class PSLState:
    """Actor, critics and data pools. Carried across specs of ONE plugin."""

    def __init__(self, plugin, cfg: PSLConfig, seed: int, dev: str):
        self.plugin_id = plugin.plugin_id
        self.model_revision = int(getattr(plugin, "model_revision", 0))
        self.dim = int(np.asarray(plugin.bounds).shape[0])
        self.n_obj = len(plugin.objective_names)
        self.cfg, self.seed, self.dev = cfg, seed, dev
        self.actor = None                 # built when the spec dim is known
        self.actor_prev = None
        self.cheap_critic = None
        self.build_critic = None
        self.res_critic = None
        self.pools = Pools()
        self.prev_specs: list = []
        self.obj_log = None
        self.jit = "none"
        self.jit_error = None

    def ensure_actor(self, spec_dim: int):
        if self.actor is None:
            self.actor = Actor(spec_dim, self.n_obj, self.dim).to(self.dev)
            if self.cfg.compile:
                sample = torch.zeros(4, spec_dim + self.n_obj, device=self.dev)
                self.actor.body = compile_module(self.actor.body, sample=sample)
                self.jit = getattr(self.actor.body, "__phycore_jit__", "none")
                self.jit_error = compile_status()["error"]


# =============================================================================
# 3. The plugin seen as the bench contract
# =============================================================================


class _Task:
    def __init__(self, plugin, spec, fidelity: int):
        self.plugin, self.spec, self.fidelity = plugin, spec, int(fidelity)
        self.bounds = np.asarray(plugin.bounds, dtype=float)
        self.dim = self.bounds.shape[0]
        self.n_obj = len(plugin.objective_names)
        self.s_vec = spec_vector(plugin, spec).astype(np.float32)
        self.spec_dim = len(self.s_vec)
        self.torch_cheap = getattr(plugin, "torch_cheap", None)
        self.has_torch = self.torch_cheap is not None
        self.n_con = None

    def cheap(self, theta):
        rec = self.plugin.evaluate(repair(self.plugin, theta), self.spec, 0)
        F = np.asarray(self.plugin.objectives(rec, self.spec), float)
        G = np.asarray(self.plugin.constraints(rec, self.spec), float)
        if self.n_con is None:
            self.n_con = len(G)
        return F, G, rec

    def exp(self, theta):
        rec = self.plugin.evaluate(repair(self.plugin, theta), self.spec, self.fidelity)
        F = np.asarray(self.plugin.objectives(rec, self.spec), float)
        G = np.asarray(self.plugin.constraints(rec, self.spec), float)
        return F, G, rec, bool(reached(rec, self.fidelity))


# =============================================================================
# 4. The loop
# =============================================================================


class PSL:
    def __init__(self, plugin, spec, seed: int, cfg: PSLConfig | None = None,
                 state: PSLState | None = None, fidelity: int = 1, log=None):
        self.cfg = cfg or PSLConfig()
        if self.cfg.threads:
            torch.set_num_threads(int(self.cfg.threads))
        self.dev = torch_device(self.cfg.device)
        self.log = log if log is not None else (lambda *_a, **_k: None)
        self.p = _Task(plugin, spec, fidelity)
        self.fidelity = int(fidelity)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if state is not None:
            if state.plugin_id != plugin.plugin_id or state.dim != self.p.dim:
                raise ValueError(
                    f"el estado heredado es de {state.plugin_id!r} ({state.dim}-D), "
                    f"no de {plugin.plugin_id!r} ({self.p.dim}-D)")
            if state.dev != self.dev:
                # a state trained on another device moves with the run
                state.dev = self.dev
                for net in (state.actor, state.actor_prev):
                    if net is not None:
                        net.to(self.dev)
        self.st = state if state is not None else PSLState(plugin, self.cfg, seed, self.dev)
        self.st.ensure_actor(self.p.spec_dim)
        # a carried actor is the INITIALISATION of the new spec, never its
        # anchor: the KL trust region restarts from the first update here
        self.st.actor_prev = None
        if state is not None:
            # re-widen the inherited cloud: the prior says WHERE, not how sure
            with torch.no_grad():
                b = self.st.actor.logstd.bias
                b.copy_(torch.maximum(b, torch.full_like(b, -1.0)))
        self.lo, self.hi = self.p.bounds[:, 0], self.p.bounds[:, 1]
        self.s_vec = self.p.s_vec
        self.use_torch_cheap = bool(self.p.has_torch and self.cfg.exact_cheap)
        # verified data of THIS spec
        self.theta, self.F, self.G, self.ok, self.recs = [], [], [], [], []
        self.cheapF_feas: list = []
        self.cands: list = []          # (thn, F_exact, v) of this spec's L0 candidates
        self.honesty, self.batch_log, self.history = [], [], []
        self.n_cheap = 0

    # --- helpers -----------------------------------------------------------
    def to_phys(self, thn):
        return self.lo + np.asarray(thn, float) * (self.hi - self.lo)

    def to_n(self, th):
        return (np.asarray(th, float) - self.lo) / (self.hi - self.lo)

    def _lhs(self, n):
        return lhs(n, self.p.dim, self.rng)

    def _cheap_np(self, thn):
        self.n_cheap += 1
        try:
            F, G, _rec = self.p.cheap(self.to_phys(thn))
            if np.all(np.isfinite(F)) and np.all(np.isfinite(G)):
                return F, G, True
        except EVALUATION_ERRORS:
            pass
        return None, None, False

    def _exp_np(self, thn):
        try:
            F, G, rec, ok = self.p.exp(self.to_phys(thn))
            if np.all(np.isfinite(F)) and np.all(np.isfinite(G)):
                return F, G, rec, ok
        except EVALUATION_ERRORS:
            pass
        return None, None, None, False

    def _obj_transform(self, F):
        """log for objectives that are strictly positive in the data."""
        F = np.asarray(F, float)
        if self.st.obj_log is None:
            self.st.obj_log = np.all(F > 0, axis=0) if F.ndim == 2 else (F > 0)
        out = F.copy()
        out[..., self.st.obj_log] = np.log(np.maximum(F[..., self.st.obj_log], 1e-12))
        return out

    def _obj_untransform_t(self, Ft):
        mask = torch.as_tensor(self.st.obj_log, device=self.dev)
        return torch.where(mask, torch.exp(Ft), Ft)

    def _n_con(self):
        if self.p.n_con is None:
            # probe until one design builds so the NaN rows have a width
            for thn in [np.full(self.p.dim, 0.5)] + list(self._lhs(32)):
                F, G, build = self._cheap_np(thn)
                self._add_cheap(thn, F, G, build)
                if build:
                    break
        return self.p.n_con or 1

    # --- data ----------------------------------------------------------------
    def _add_cheap(self, thn, F, G, build, s=None):
        s = self.s_vec if s is None else s
        n_con = self.p.n_con if self.p.n_con is not None else 1
        nF = F if build else np.full(self.p.n_obj, np.nan)
        nG = G if build else np.full(n_con, np.nan)
        self.st.pools.cheap_X.append(np.concatenate([thn, s]).astype(np.float32))
        self.st.pools.cheap_Y.append((nF, nG, 0.0 if build else 1.0))

    def _add_exp(self, thn, F_cheap, F_exp, ok):
        if not ok:
            return
        f0 = max(abs(F_cheap[0]), 1e-9)
        r = F_exp[0] / f0 - 1.0
        x = np.concatenate([thn, self.s_vec, self._obj_transform(F_cheap)]).astype(np.float32)
        self.st.pools.exp_X.append(x)
        self.st.pools.exp_r.append([r])
        self.st.pools.exp_spec.append(self.s_vec.copy())

    def _record(self, thn, F_true, G, ok, rec):
        self.theta.append(self.to_phys(thn)); self.F.append(F_true)
        self.G.append(G); self.ok.append(ok); self.recs.append(rec)

    # --- critics ---------------------------------------------------------------
    def _fit_critics(self, cheap_only: bool = False):
        cfg, P = self.cfg, self.st.pools
        if not self.use_torch_cheap:
            X_all = np.array(P.cheap_X)
            # every row of THIS spec, plus a bounded sample of the other specs:
            # the critic is conditioned on the spec, but with two or three specs
            # of data it cannot interpolate, so the current one must dominate
            cur = np.all(np.isclose(X_all[:, self.p.dim:], self.s_vec), axis=1)
            idx_cur = np.where(cur)[0][-8000:]
            idx_old = np.where(~cur)[0]
            if len(idx_old) > 2000:
                idx_old = self.rng.choice(idx_old, 2000, replace=False)
            sel = np.concatenate([idx_old, idx_cur]).astype(int)
            X = X_all[sel]
            rows = [P.cheap_Y[i] for i in sel]
            build = np.array([r[2] for r in rows])
            good = build < 0.5
            n_con = self._n_con()
            Fs = np.array([r[0] for r in rows])
            Gs = np.array([r[1] if len(r[1]) == n_con else np.full(n_con, np.nan) for r in rows])
            # g is learnt in asinh scale: the barrier needs accuracy near zero,
            # not on a wall that is sixty budgets away.
            Ft = self._obj_transform(Fs[good])
            Y = np.concatenate([Ft, np.arcsinh(Gs[good])], 1)
            self.st.cheap_critic = AnchoredEnsemble(
                X.shape[1], Y.shape[1], k=cfg.k_ensemble, hidden=(96, 96),
                prior_scale=0.3, seed=self.seed, device=self.dev,
                compile_nets=cfg.compile)
            self.st.cheap_critic.fit(X[good], Y, epochs=30 if len(Y) > 1500 else 100,
                                     batch=512, calib_frac=0.15)
            if (build > 0.5).any():
                self.st.build_critic = AnchoredEnsemble(
                    X.shape[1], 1, k=3, hidden=(64, 64), prior_scale=0.2,
                    seed=self.seed + 7, device=self.dev, compile_nets=cfg.compile)
                self.st.build_critic.fit(X, build[:, None], epochs=30, batch=512, calib_frac=0.0)
            else:
                self.st.build_critic = None
        if cheap_only:
            return
        if len(P.exp_r) >= 12:
            X = np.array(P.exp_X); Y = np.array(P.exp_r)
            self.st.res_critic = AnchoredEnsemble(
                X.shape[1], 1, k=cfg.k_ensemble, hidden=(64, 64), prior_scale=0.5,
                seed=self.seed + 3, device=self.dev, compile_nets=cfg.compile)
            self.st.res_critic.fit(X, Y, epochs=300, batch=64, calib_frac=0.2)

    def _model(self, thn_t, s_t, pess=True):
        """Pessimistic objectives and predicted g for a batch of normalised thetas."""
        cfg = self.cfg
        if self.use_torch_cheap:
            lo = torch.as_tensor(self.lo, dtype=torch.float32, device=self.dev)
            hi = torch.as_tensor(self.hi, dtype=torch.float32, device=self.dev)
            F0, G = self.p.torch_cheap(lo + thn_t * (hi - lo), s_t)
            sdF = torch.zeros_like(F0)
        else:
            mu, sd = self.st.cheap_critic.predict(torch.cat([thn_t, s_t], -1))
            F0 = self._obj_untransform_t(mu[:, :self.p.n_obj])
            sdF = sd[:, :self.p.n_obj] * F0.abs().clamp_min(1e-9) \
                * torch.as_tensor(self.st.obj_log, device=self.dev)
            G = mu[:, self.p.n_obj:]                        # asinh scale, monotone in g
            if self.st.build_critic is not None:
                bmu, _ = self.st.build_critic.predict(torch.cat([thn_t, s_t], -1))
                G = torch.cat([G, (bmu - 0.5) * 2.0], -1)  # unbuildable region as a constraint
        F = F0
        sd_r = torch.zeros(thn_t.shape[0], 1, device=self.dev)
        mu_r = torch.zeros(thn_t.shape[0], 1, device=self.dev)
        if self.st.res_critic is not None:
            mask = torch.as_tensor(self.st.obj_log, device=self.dev)
            x = torch.cat([thn_t, s_t, torch.where(mask, torch.log(F0.abs().clamp_min(1e-12)), F0)], -1)
            mu_r, sd_r = self.st.res_critic.predict(x)
            corr = 1.0 + mu_r[:, 0] + (cfg.kappa * sd_r[:, 0] if pess else 0.0)
            F = torch.cat([(F0[:, 0] * corr)[:, None], F0[:, 1:]], -1)
        if pess:
            F = F + cfg.kappa * sdF
        return F, G, sd_r[:, 0], mu_r[:, 0]

    def _transform_t(self, F):
        mask = torch.as_tensor(self.st.obj_log, device=self.dev)
        return torch.where(mask, torch.log(F.abs().clamp_min(1e-12)), F)

    # --- actor ---------------------------------------------------------------
    def _scales(self):
        """Ideal and span for Tchebycheff, in TRANSFORMED objective space,
        from every feasible design whose objectives are known exactly."""
        rows = [f for f, g in zip(self.F, self.G) if np.max(g) <= 1e-9]
        rows += list(self.cheapF_feas[-3000:])
        Fv = np.array(rows) if rows else np.zeros((0, self.p.n_obj))
        if len(Fv) < 3:
            Fv = np.array(self.F) if self.F else np.ones((1, self.p.n_obj))
        Ft = self._obj_transform(Fv)
        z = Ft.min(0); nad = np.percentile(Ft, 90, axis=0)
        span = np.maximum(nad - z, 1e-3)
        return (torch.as_tensor(z, dtype=torch.float32, device=self.dev),
                torch.as_tensor(span, dtype=torch.float32, device=self.dev))

    def _elites(self, z, span):
        """Exact-L0 elites of this spec, with the preference each one answers.
        Feasible and non-dominated if there are enough of them; otherwise
        the least-violating candidates (feasibility-first CEM)."""
        if not self.cands:
            return None
        cands = self.cands[-6000:]
        feas = [c for c in cands if c[2] <= 1e-9]
        zc, sc = z.cpu().numpy(), span.cpu().numpy()
        if len(feas) >= 5:
            F = np.array([c[1] for c in feas])
            Fn = (self._obj_transform(F) - zc) / sc
            nd = np.ones(len(F), bool)
            for i in range(len(F)):
                if nd[i]:
                    dom = np.all(F <= F[i], 1) & np.any(F < F[i], 1)
                    nd[i] = not dom.any()
            idx = np.where(nd)[0]
            if len(idx) > self.cfg.n_elite:
                idx = self.rng.choice(idx, self.cfg.n_elite, replace=False)
            th = np.array([feas[i][0] for i in idx]); Fn = Fn[idx]
        else:
            order = np.argsort([c[2] for c in cands])[: self.cfg.n_elite]
            th = np.array([cands[i][0] for i in order])
            F = np.array([cands[i][1] for i in order])
            Fn = (self._obj_transform(F) - zc) / sc
        lam = 1.0 / np.maximum(Fn - Fn.min(0) + 0.05, 0.05)
        lam = lam / lam.sum(1, keepdims=True)
        th = np.clip(th, 0.02, 0.98)          # |logit| <= 3.9, inside the actor's reach
        pre = np.log(th / (1.0 - th))
        return (torch.as_tensor(pre, dtype=torch.float32, device=self.dev),
                torch.as_tensor(lam, dtype=torch.float32, device=self.dev))

    def _train_actor(self, steps):
        cfg, A = self.cfg, self.st.actor
        prev = self.st.actor_prev
        opt = torch.optim.Adam(A.parameters(), lr=cfg.actor_lr)
        z, span = self._scales()
        elites = self._elites(z, span) if cfg.w_elite > 0 else None
        s_cur = torch.as_tensor(self.s_vec, device=self.dev)
        replay = [torch.as_tensor(s, device=self.dev) for s in self.st.prev_specs]
        B = cfg.actor_batch
        dirichlet = torch.distributions.Dirichlet(torch.ones(self.p.n_obj))
        for _it in range(steps):
            n_rep = int(cfg.replay_frac * B) if replay else 0
            s = s_cur.expand(B, -1).clone()
            if n_rep:
                pick = self.rng.integers(0, len(replay), n_rep)
                s[:n_rep] = torch.stack([replay[i] for i in pick])
            lam = dirichlet.sample((B,)).to(self.dev)
            eps = torch.randn(B, self.p.dim, device=self.dev)
            thn, m, ls = A.sample(s, lam, eps)
            F, G, _, _ = self._model(thn, s, pess=True)
            Fn = (self._transform_t(F) - z) / span
            tch = (lam * Fn).max(-1).values + 0.05 * (lam * Fn).sum(-1)
            # linear barrier: a squared one drives the cloud into a point
            pen = torch.relu(G).sum(-1)
            ent = ls.mean(-1)
            loss = tch.mean() + cfg.w_pen * pen.mean() - cfg.w_ent * ent.mean()
            if elites is not None:
                # CEM anchor: the exact elites must be likely under the actor
                pre_e, lam_e = elites
                s_e = s_cur.expand(pre_e.shape[0], -1)
                m_e, ls_e = A.params(s_e, lam_e)
                nll = (0.5 * ((pre_e - m_e) / torch.exp(ls_e)) ** 2 + ls_e).sum(-1)
                loss = loss + cfg.w_elite * nll.mean() / self.p.dim
            if prev is not None:
                with torch.no_grad():
                    m2, ls2 = prev.params(s, lam)
                loss = loss + cfg.w_kl * Actor.kl(m, ls, m2, ls2).mean()
            if cfg.w_stein > 0:
                cur = thn[n_rep:]
                nb = cur.shape[0]
                d2 = torch.cdist(cur, cur).pow(2)
                h = d2.detach().median().clamp_min(1e-3)
                # mean pairwise kernel, diagonal excluded: O(1), not O(B)
                rep = (torch.exp(-d2 / h).sum() - nb) / max(nb * (nb - 1), 1)
                loss = loss + cfg.w_stein * rep
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(A.parameters(), 5.0)
            opt.step()
        self.st.actor_prev = copy.deepcopy(A)

    # --- batch selection -----------------------------------------------------
    def _hv(self, F, ref):
        return hypervolume(np.asarray(F, float), ref) if len(F) else 0.0

    def _propose(self, batch_size):
        cfg, A, N = self.cfg, self.st.actor, self.cfg.n_candidates
        n_con = self._n_con()
        with torch.no_grad():
            s = torch.as_tensor(self.s_vec, device=self.dev).expand(N, -1)
            lam = torch.distributions.Dirichlet(torch.ones(self.p.n_obj)).sample((N,)).to(self.dev)
            eps = torch.randn(N, self.p.dim, device=self.dev)
            temp = torch.where(torch.arange(N, device=self.dev) < int(0.8 * N), 1.0, 2.0)[:, None]
            thn, _, _ = A.sample(s, lam, eps, temp=temp)
            thn = thn.cpu().numpy().astype(float)
        # exact L0 on every candidate: g is never predicted where it can be computed
        Fc, Gc, feas, vio = [], [], [], []
        for t in thn:
            F, G, build = self._cheap_np(t)
            self._add_cheap(t, F, G, build)
            if not build:
                Fc.append(np.full(self.p.n_obj, np.nan)); Gc.append(np.full(n_con, np.nan))
                feas.append(False); vio.append(np.inf); continue
            v = float(np.sum(np.maximum(G, 0)))
            Fc.append(F); Gc.append(G); feas.append(v <= 1e-9); vio.append(v)
            self.cands.append((np.asarray(t, float), np.asarray(F, float), v))
            if v <= 1e-9:
                self.cheapF_feas.append(np.asarray(F, float))
        Fc, Gc, feas, vio = np.array(Fc), np.array(Gc), np.array(feas), np.array(vio)
        n_feas = int(feas.sum())
        idx_b = np.where(np.isfinite(vio))[0]
        with torch.no_grad():
            t_t = torch.as_tensor(thn[idx_b], dtype=torch.float32, device=self.dev)
            s_t = torch.as_tensor(self.s_vec, device=self.dev).expand(len(idx_b), -1)
            if len(idx_b) and (self.use_torch_cheap or self.st.cheap_critic is not None):
                _, _, sd_r, mu_r = self._model(t_t, s_t, pess=True)
                sd_r, mu_r = sd_r.cpu().numpy(), mu_r.cpu().numpy()
            else:
                sd_r, mu_r = np.zeros(len(idx_b)), np.zeros(len(idx_b))
            # selection uses the EXACT L0 of every candidate plus the predicted
            # residual (pessimistic); the L0 surrogate only ever lives inside the
            # actor's gradient
            Fp = Fc[idx_b].copy()
            if len(idx_b):
                Fp[:, 0] = Fp[:, 0] * (1.0 + mu_r + cfg.kappa * sd_r)
        Fpred = np.full((N, self.p.n_obj), np.nan); Fpred[idx_b] = Fp
        SD = np.zeros(N); SD[idx_b] = sd_r
        MU = np.zeros(N); MU[idx_b] = mu_r
        # support gate: distance to the nearest verified design (normalised)
        if self.theta:
            Tn = np.array([self.to_n(t) for t in self.theta])
            dmin = np.min(np.linalg.norm(thn[:, None, :] - Tn[None], axis=2), axis=1)
            gate_thr = np.percentile(dmin[feas], 90) if n_feas > 5 else np.inf
        else:
            dmin = np.zeros(N); gate_thr = np.inf
        rejected = dmin > gate_thr

        chosen: list[int] = []

        def far_enough(i):
            return all(np.linalg.norm(thn[i] - thn[j]) >= cfg.min_dist for j in chosen)

        # 1. exploit: greedy hypervolume gain over the verified feasible front
        ref = self._ref_for_selection()
        Fv = np.array([f for f, g in zip(self.F, self.G) if np.max(g) <= 1e-9]) \
            if self.F else np.zeros((0, self.p.n_obj))
        base_hv = self._hv(Fv, ref)
        pool = [i for i in np.where(feas & ~rejected)[0]]
        cur = list(Fv)
        k_exp = int(round(cfg.exploit_frac * batch_size))
        for _ in range(k_exp):
            best, best_gain = None, -1.0
            for i in pool:
                if i in chosen or not far_enough(i):
                    continue
                gain = self._hv(cur + [Fpred[i]], ref) - base_hv
                if gain > best_gain:
                    best, best_gain = i, gain
            if best is None:
                break
            chosen.append(best); cur.append(Fpred[best]); base_hv = self._hv(cur, ref)
        # 2. explore: widest residual band among the feasible, plus what the gate rejected
        k_explore = int(round(cfg.explore_frac * batch_size))
        cand_rej = [i for i in np.where(feas & rejected)[0]]
        self.rng.shuffle(cand_rej)
        for i in cand_rej[: max(1, k_explore // 2)]:
            if len(chosen) < k_exp + k_explore and far_enough(i):
                chosen.append(i)
        order = np.argsort(-SD)
        for i in order:
            if len(chosen) >= k_exp + k_explore:
                break
            if feas[i] and i not in chosen and far_enough(i):
                chosen.append(i)
        # 3. frontier: |max g| nearest zero, the family's FRONTIER pool
        gmax = np.array([np.max(g) if np.all(np.isfinite(g)) else np.inf for g in Gc])
        near = [i for i in np.argsort(np.abs(gmax)) if np.isfinite(gmax[i])]
        for i in near:
            if len(chosen) >= batch_size:
                break
            if i not in chosen and far_enough(i):
                chosen.append(i)
        for i in np.where(feas)[0]:
            if len(chosen) >= batch_size:
                break
            if i not in chosen and far_enough(i):
                chosen.append(i)
        # starvation guard: a collapsed cloud must not waste the budget. Relax the
        # dedup radius, then fill with fresh LHS points (free exploration).
        radius = cfg.min_dist
        while len(chosen) < batch_size and radius > 1e-4:
            radius *= 0.5
            for i in np.argsort(np.abs(gmax)):
                if len(chosen) >= batch_size:
                    break
                if np.isfinite(gmax[i]) and i not in chosen and \
                        all(np.linalg.norm(thn[i] - thn[j]) >= radius for j in chosen):
                    chosen.append(i)
        extra = []
        if len(chosen) < batch_size:
            for t in self._lhs(batch_size - len(chosen)):
                F, G, build = self._cheap_np(t)
                self._add_cheap(t, F, G, build)
                if build:
                    extra.append((t, F, 0.0, 0.0))
        chosen = chosen[:batch_size]
        self.batch_log.append((len(chosen), int(np.sum(feas[chosen])) if chosen else 0, n_feas, N))
        return [(thn[i], Fc[i], MU[i], SD[i]) for i in chosen] + extra

    def _ref_for_selection(self):
        Fall = np.array(self.F) if self.F else np.ones((1, self.p.n_obj))
        return np.maximum(np.percentile(Fall, 95, axis=0) * 1.1, 1e-9)

    # --- the run -------------------------------------------------------------
    def run(self, n_init: int, rounds: int, batch_size: int) -> dict:
        cfg = self.cfg
        self._n_con()
        # free L0 seed for the cheap critic (the family's NSGA-II spends
        # thousands of exact L0 evaluations per round; this is the same money)
        if not self.use_torch_cheap and cfg.n_cheap_init > 0:
            for thn in self._lhs(cfg.n_cheap_init):
                Fc, Gc, build = self._cheap_np(thn)
                self._add_cheap(thn, Fc, Gc, build)
                if build:
                    v = float(np.sum(np.maximum(Gc, 0)))
                    self.cands.append((np.asarray(thn, float), np.asarray(Fc, float), v))
                    if v <= 1e-9:
                        self.cheapF_feas.append(np.asarray(Fc, float))
        self.log("sample", f"initial Latin hypercube: {n_init} physical "
                 f"evaluations at {fidelity_label(self.fidelity)} (device {self.dev}, "
                 f"jit {self.st.jit})")
        if n_init > 0:
            for thn in self._lhs(n_init):
                Fc, Gc, build = self._cheap_np(thn)
                self._add_cheap(thn, Fc, Gc, build)
                if not build:
                    continue
                Fe, Ge, rec, ok = self._exp_np(thn)
                if Fe is None:
                    continue
                self._record(thn, Fe if ok else Fc, Ge, ok, rec)
                self._add_exp(thn, Fc, Fe, ok)
        first = True
        for rd in range(rounds):
            self.log("round", f"{rd + 1}/{rounds}")
            self._fit_critics()
            self._train_actor(cfg.actor_steps + (150 if first else 0))
            first = False
            batch = self._propose(batch_size)
            # cheap inner loop (surrogate-L0 path only): the candidates just
            # evaluated at the exact L0 sharpen the cheap critic and the actor
            # moves again. No expensive evaluation happens here.
            inner = 0
            while (not self.use_torch_cheap and inner < cfg.max_inner
                   and (inner < cfg.inner_fixed
                        or self.batch_log[-1][2] < cfg.feas_target * self.batch_log[-1][3])):
                inner += 1
                self._fit_critics(cheap_only=True)
                self._train_actor(200)
                batch = self._propose(batch_size)
            n_before = len(self.theta)
            for thn, Fc, mu_r, sd in batch:
                Fe, Ge, rec, ok = self._exp_np(thn)
                if Fe is None:
                    continue
                self._record(thn, Fe if ok else Fc, Ge, ok, rec)
                self._add_exp(thn, Fc, Fe, ok)
                if ok and self.st.res_critic is not None:
                    r_true = Fe[0] / max(abs(Fc[0]), 1e-9) - 1.0
                    self.honesty.append((abs(r_true - float(mu_r)), float(sd), float(r_true)))
            self.history.append({"round": rd + 1, "inner": inner,
                                 "candidates": self.batch_log[-1][3],
                                 "candidates_feasible": self.batch_log[-1][2],
                                 "spent": len(self.theta) - n_before,
                                 "verified": len(self.theta),
                                 "n_cheap": self.n_cheap,
                                 "surrogate": dict(self.st.res_critic.metrics)
                                 if self.st.res_critic else {}})
            self.log("acquire", f"{len(self.theta) - n_before} designs to the expensive "
                     f"fidelity after {inner} cheap inner iterations")
        # the actor is the prior of the next spec
        self.st.prev_specs.append(self.s_vec.copy())
        plugin, spec = self.p.plugin, self.p.spec
        individuals = []
        for th, f, g, rec in zip(self.theta, self.F, self.G, self.recs):
            b = np.asarray(plugin.objective_bands(rec, spec), float)
            individuals.append(Individual(
                theta=np.asarray(rec.get("theta", th), float), rec=rec,
                f=np.asarray(f, float), b=b, g=np.asarray(g, float),
                v=plugin_violation(plugin, g), label=plugin_label(plugin, rec),
                cell=plugin_cell(plugin, rec)))
        return {
            "individuals": individuals,
            "theta": np.array(self.theta), "F": np.array(self.F), "G": np.array(self.G),
            "ok": np.array(self.ok, bool), "n_exp": len(self.theta),
            "order": np.arange(len(self.theta)), "n_init": n_init, "batch_size": batch_size,
            "n_cheap": int(self.n_cheap),
            "honesty": np.array(self.honesty) if self.honesty else np.zeros((0, 3)),
            "batch_log": self.batch_log, "history": self.history,
            "surrogate_gate": dict(self.st.res_critic.metrics) if self.st.res_critic else {},
        }
