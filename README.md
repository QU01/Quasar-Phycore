# QUASAR phycore — the design engine of the Phy family, written once

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License: Apache 2.0">
  <img src="https://img.shields.io/badge/python-%E2%89%A53.10-blue.svg" alt="Python >= 3.10">
  <img src="https://img.shields.io/badge/core-NumPy%20only-brightgreen.svg" alt="NumPy-only core">
  <img src="https://img.shields.io/badge/tests-52-brightgreen.svg" alt="52 tests">
  <img src="https://img.shields.io/badge/engine%20revision-2-blue.svg" alt="ENGINE_REVISION 2">
</p>

> **One engine, many physics.** Every QUASAR Phy model (Phy-CC
> compressors, Phy-AC, Phy-AT and Phy-CT turbines, Phy-HX heat
> exchangers, Phy-Wing wings, Phy-RP rockets) used to carry its own copy
> of the same design loop: a residual ensemble with split conformal
> bands, NSGA-II on Deb's constrained dominance, an
> exploit / explore / frontier acquisition batch, and a Pareto front made
> only of evaluations that really happened. The family's architecture
> audit counted that loop copied **eight times**. `phycore` is that loop
> written once — plus a newer, learned search — behind a single **plugin
> contract**, so a model chooses its search with one argument and keeps
> its physics to itself.
>
> **Phy-Prop** ([QUASAR Phy-Prop](https://github.com/QU01), propellers,
> rotors and small wind turbines) is the first product built on it end
> to end: everything Phy-Prop searches, gates, calibrates and reports
> comes from here, and the parity tests that prove `phycore` reproduces
> the family's original optimiser **bit for bit** live in both
> repositories.

---

## What it gives a model

```
Phy-1   search="nsga2"   ensemble on the L1 − L0 residual, split conformal bands,
                         NSGA-II on the PESSIMISTIC merit, exploit/explore/frontier
                         acquisition, a VERIFIED front (the Phy-HX loop, verbatim:
                         the parity test reproduces optimizer_hx bit for bit)

Phy-2   search="psl"     an actor π(θ | spec, λ) with pathwise gradients, an anchored
                         residual critic and a cheap-L0 critic, Tchebycheff scalarisation,
                         CEM anchoring to exact-L0 elites, memory across specs
                         (the version that won Phy-Bench, ported to the same contract)
```

Both searches return the same `DesignResult`, so a product's report, its
release gate and its benchmarks read either one without knowing which
ran.

Around the search, the pieces every Phy product needs and used to
re-implement:

| Package | What it holds |
|---|---|
| `phycore.plugin` | the `PhyPlugin` Protocol, the typed errors (`InfeasibleDesign`, `ContractError`), `repair`, `violation`, and `validate_plugin` — the list of contract breaches, before the engine trusts a plugin |
| `phycore.engine` | `mlp`, `ensemble` (bootstrap + split conformal, keeps its calibration split), `dominance` (Deb, pessimistic merit, `Individual`), `nsga2`, `acquisition`, `hypervolume` (exact, 2-D and n-D, NumPy), `sampling`, `designer` (the Phy-1 loop and the entry points), `psl` (Phy-2, torch) |
| `phycore.uq` | `conformal` (quantile, isotonic, **Mondrian** with a per-cell gate), `bands` (declared bands with marks, a table with **no defaults**), `discrepancy` (the AR1 discrepancy GP of the expensive ladder: $y_{L2} = \rho\,y_{L1} + \delta(x)$ with ρ marginalised, exact LOO, a **do-not-calibrate** verdict) |
| `phycore.contracts` | `schema` (common validators of the geometry contracts: keys, numbers, monotone z, polygons of equal count, a versioned `ContractSchema`) and `marks` (`[V]` `[S]` `[M]` `[?]`, and the prior width each mark implies) |
| `phycore.vvuq` | `campaign` (`Criterion`, `Score`, subordination, **BLOCKED by silence**, a release `gate` that fails with a named cause, `report`) |
| `phycore.product` | `run_meta` (`ENGINE_REVISION` + `model_revision` + a SHA-256 fingerprint of the constants: the cache key of every run) |
| `phycore.backend` | `get_backend("auto")` → NumPy / `jax.numpy` (float64 on) / CuPy as one `xp`; `jit`, `grad`, `torch_device`, `compile_module`, `device_info`, and `torch_from_jax` — a JAX-written L0 turned into a torch-differentiable function by its VJP, which is what Phy-2 needs as `torch_cheap` |
| `phycore.adapters` | `hx.HXPlugin` (Phy-HX behind the contract, unchanged) and `bench.ProblemPlugin` (the Phy-Bench problems) |

Three rules hold everywhere: the **core stays pure NumPy** (torch only
inside `engine.psl`, and only when asked for); **a plugin never imports
another plugin**; and **the front only contains evaluations that
happened** — a surrogate's prediction is never a point on it.

---

## Install

```bash
pip install "phycore @ git+https://github.com/QU01/Quasar-Phycore.git"            # the engine, NumPy only
pip install "phycore[neural] @ git+https://github.com/QU01/Quasar-Phycore.git"    # + torch, for search="psl" (Phy-2)
```

From a checkout:

```bash
git clone https://github.com/QU01/Quasar-Phycore.git phycore
pip install -e phycore              # or:  pip install -e "phycore[neural]"
```

Optional accelerators, none required: **torch** (Phy-2, CUDA when it
sees one), **JAX** (a differentiable L0 in the plugins, float64), **CuPy**
(an `xp` on the GPU), **numba** (`backend.jit` fallback). Ask the machine
what it can do:

```bash
python -m phycore
```

```json
{"phycore": "0.1.0", "engine_revision": 2,
 "compute": {"numpy": "2.4.6", "torch": "2.9.0+cu129", "torch_cuda": true,
             "torch_device_name": "NVIDIA GeForce RTX 3070 Ti", "torch_compile": true,
             "jax": "0.11.0", "jax_gpu": false, "cupy": "14.1.1"}}
```

Requires Python ≥ 3.10.

---

## Use

```python
from phycore import design, design_sequence, validate_plugin
from phycore.adapters.hx import HXPlugin            # Phy-HX, unchanged behind the contract

hx = HXPlugin()                                     # a plate-fin recuperator family
spec = hx.ds.recuperator_spec()
assert validate_plugin(hx, spec, fidelity=1) == []  # the contract, checked before trusting the plugin

r1 = design(hx, spec, fidelity=1, search="nsga2")   # Phy-1: ensemble + NSGA-II
r2 = design(hx, spec, fidelity=1, search="psl")     # Phy-2: actor + critics, cold start
rs = design_sequence(hx, [spec_a, spec_b, spec_c],  # Phy-2 with memory: the actor, critics and
                     search="psl")                  # data of one spec are the prior of the next

r1.front          # the VERIFIED front: real evaluations only, non-dominated on the pessimistic merit
r1.population     # every real evaluation, as Individuals (theta, rec, f, b, g, v, feasible)
r1.history        # one row per round: what was spent, what was verified, the surrogate's gate
r1.cell_gate      # the Mondrian gate per (family, mode) cell
r2.honesty        # Phy-2: |truth − critic| against the critic's own sd, for every proposal it spent
r2.state          # Phy-2: actor + critics + data — the prior of the next spec
```

`design(plugin, spec, fidelity=0, n_init=60, rounds=3, batch_size=12,
pop=40, generations=20, seed=SEED, log=None, on_generation=None,
search="nsga2", psl_config=None, state=None)`. The budget is
`n_init + rounds × batch_size` **expensive** evaluations; the inner
search never calls the expensive layer directly.

### The fidelity ladder, and what the surrogate learns

`fidelity=0` means the plugin's cheap level **is** the truth: the residual
is zero by construction, so the search runs on the exact physics and no
surrogate is built — the engine says so in its log rather than fitting
a model of nothing. `fidelity ≥ 1` raises the **expensive** evaluations
(the initial Latin hypercube and every acquisition batch) to that level
and trains the ensemble on the **residual** between the expensive level
and the one below, over the plugin's `embedding`. Only records whose
expensive truth *exists* train it: a level that did not converge keeps
the lower level's answer, labelled, and `residual_matrix` skips it — a
missing truth is not a zero.

The surrogate has to **earn** the search: its split-conformal coverage
must reach the nominal level minus 5 points, or the round spends twice
its batch on more samples instead of on a model that has not proved
itself (`QUALITY GATE FAILED`, in the log). Every objective is judged at
the **pessimistic** end of its declared band, and the front is filtered
on that merit.

### Phy-2 in one paragraph

`search="psl"` trains an actor $\pi(\theta \mid s, \lambda)$ on the spec
vector $s$ and a preference $\lambda$, scores its proposals with a cheap
critic (the plugin's exact L0 when it offers a differentiable one via
`torch_cheap`, a learned one otherwise) and a residual critic anchored on
the expensive truths, scalarises with Tchebycheff, keeps the actor near
the exact-L0 **elites** (a CEM step) and spends its batch on the
proposals the critics rate best — then records, for each proposal it
spent, how far the truth fell from what the critic said (`honesty`).
With `design_sequence` the trained state seeds the next spec, which is
what buys the 40–60 % of the expensive budget Phy-Bench measured.
`PSLConfig` holds the knobs (actor steps, candidates, ensemble size,
`device="auto"` → CUDA when torch sees one, `compile=True` →
`torch.compile` with a guard that falls back to eager and says so).

---

## Writing a plugin

A plugin is an object with five attributes — `plugin_id`,
`model_revision`, `bounds` (d × 2), `integer_dims`, `objective_names` —
and eight calls:

| Call | What it is |
|---|---|
| `spec_vec(spec)` | the specification as a vector in ≈ [0, 1]ᵏ; it conditions the learned front |
| `evaluate(theta, spec, fidelity)` | **the physics**; returns a record carrying `theta` and `fidelity_used`; raises `InfeasibleDesign` / `ContractError` / `ValueError` when there is no design |
| `objectives(rec, spec)` | all **minimised** |
| `objective_bands(rec, spec)` | absolute declared half-widths, one per objective (zero for what is geometry) |
| `constraints(rec, spec)` | g ≤ 0, continuous, **exact** — never predicted |
| `embedding(rec)` | what the residual surrogate learns over |
| `residual(rec, spec)` / `apply_residual(rec, spec, r)` | the inverse pair between the expensive level and the one below; `apply_residual` marks the record `corrected_by_surrogate` |

Optional: `cell(rec)` (the Mondrian cell), `label(rec)`, `violation(g)`,
`constraint_names(spec)`, `residual_dim`, and `torch_cheap(theta_t,
spec_t)` — a differentiable cheap model for Phy-2 (`backend.torch_from_jax`
builds it from a JAX L0). `validate_plugin(plugin, spec)` returns the list
of breaches; an empty list is the plugin's birth certificate.

The contract is what lets a product keep its physics private and still
get the whole engine: Phy-Prop's `plugin_prop.py` is ~1 000 lines of
propeller physics and **zero** lines of search.

---

## Structure

```
phycore/
├── __init__.py            version, ENGINE_REVISION, the public names
├── __main__.py            python -m phycore: what the machine can accelerate
├── plugin.py              PhyPlugin Protocol, errors, repair / violation, validate_plugin
├── backend.py             xp / jit / grad / torch_device / compile_module / torch_from_jax / device_info
├── engine/
│   ├── designer.py        the Phy-1 loop, design(), design_sequence(), DesignResult, verified_front
│   ├── psl.py             Phy-2: PSLConfig, actor, critics, honesty (torch)
│   ├── ensemble.py        bootstrap ensemble + split conformal (keeps its calibration split)
│   ├── nsga2.py           NSGA-II on Deb's constrained dominance, pessimistic merit
│   ├── acquisition.py     exploit / explore / frontier batch
│   ├── dominance.py       Individual, pessimistic(), the front
│   ├── hypervolume.py     exact hypervolume (2-D, n-D), reference_point, hv_curve
│   ├── sampling.py        Latin hypercube
│   └── mlp.py             the small NumPy network of the ensemble
├── uq/
│   ├── conformal.py       quantile, isotonic, reliability, Mondrian with a per-cell gate
│   ├── bands.py           DeclaredBand, BandTable (no defaults), BandNotDeclared
│   └── discrepancy.py     DiscrepancyGP (AR1, Matérn 5/2 ARD, marginal ρ, exact LOO, verdict)
├── contracts/
│   ├── schema.py          check_keys, num, monotone_z, check_polygon, ContractSchema, validate_or_raise
│   └── marks.py           [V] [S] [M] [?], parse / count / strip, prior_width
├── vvuq/campaign.py       Criterion, Score, Campaign, tally, gate, report
├── product/run_meta.py    constants_fingerprint, theta_digest, spec_digest, cache_key, run_meta
└── adapters/              hx.py (Phy-HX), bench.py (Phy-Bench)
tests/                     52 tests
```

---

## Tests

```bash
python -m pytest -q          # 52 tests
```

NumPy is enough for the core; torch runs Phy-2 and the CUDA tests
(skipped without a GPU); JAX, numba and CuPy only where used. The parity
test reproduces the family's original `optimizer_hx` **bit for bit**,
and the tests that need the sibling checkouts `Phy-HX` and `Phy-Bench`
look for them next to this repository and skip, by name, when they are
not there.

---

## What is not here yet

From the architecture plan: the equation-oriented differentiable L0 of
Phy-AT, Sobolev training, hierarchical calibration with NumPyro, the
asynchronous L3 with cost-aware acquisition, and projected refinement.
The contract already leaves room for them: `torch_cheap` /
`backend.grad` for the derivatives, `cell` for the per-cell gate,
`marks.prior_width` for the priors by mark. What *did* land since the
plan — the AR1 discrepancy GP (`uq.discrepancy`), fixed-shape batching
for the differentiable L0, and the critic's honesty record — landed
because Phy-Prop needed it first.

---

## License

Apache 2.0 (`LICENSE`), like the rest of the QUASAR Phy family.
