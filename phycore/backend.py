"""Array backend, JIT and device selection - optional, never required.

The family's invariant was "pure NumPy in the core". Phy-2 relaxes it to
"Array API with JAX optional": a plugin writes its L0 against ``xp`` and
gets NumPy on a laptop, JAX (with ``jit`` and ``grad``) or CuPy on a GPU
without touching the physics. Nothing here imports torch, jax, cupy or
numba at module load; each is looked up on first use and its absence is
reported, not raised, so a CPU-only machine runs everything except the
acceleration it does not have.

Entry points::

    xp, name = get_backend("auto")        # "numpy" | "jax" | "cupy"
    f_fast   = jit(f)                     # jax.jit / numba.njit / identity
    dev      = torch_device("auto")       # "cuda" if torch sees one
    compile_module(nn_module)             # torch.compile, guarded
    torch_from_jax(fn)                    # differentiable bridge for PSL
    device_info()                         # what this machine can do
"""

from __future__ import annotations

import functools
import importlib
import os
from typing import Any, Callable

import numpy as np

_CACHE: dict = {}
_COMPILE_STATE: dict = {"ok": False, "error": None}


def _try_import(name: str):
    if name in _CACHE:
        return _CACHE[name]
    try:
        mod = importlib.import_module(name)
    except Exception:                                        # noqa: BLE001
        mod = None
    _CACHE[name] = mod
    return mod


# =============================================================================
# 1. Array backend
# =============================================================================


def get_backend(name: str = "auto") -> tuple:
    """Return ``(xp, backend_name)``.

    ``auto`` honours ``PHYCORE_BACKEND`` and otherwise prefers JAX, then
    CuPy when a CUDA device is visible, then NumPy. ``jax`` enables
    float64 so a plugin's L0 does not silently lose eight digits.
    """
    want = (os.environ.get("PHYCORE_BACKEND", "") or name).lower()
    if want in ("", "auto"):
        jax = _try_import("jax")
        if jax is not None:
            want = "jax"
        elif cuda_available("cupy"):
            want = "cupy"
        else:
            want = "numpy"
    if want == "numpy":
        return np, "numpy"
    if want == "jax":
        jax = _try_import("jax")
        if jax is None:
            return np, "numpy"
        try:
            jax.config.update("jax_enable_x64", True)
        except Exception:                                    # noqa: BLE001
            pass
        return importlib.import_module("jax.numpy"), "jax"
    if want == "cupy":
        cp = _try_import("cupy")
        if cp is None:
            return np, "numpy"
        return cp, "cupy"
    raise ValueError(f"backend desconocido: {name!r} (numpy | jax | cupy | auto)")


def to_numpy(x) -> np.ndarray:
    """Whatever array it is, a NumPy array on the host."""
    if isinstance(x, np.ndarray):
        return x
    for attr in ("get", "cpu"):                              # cupy, torch
        if hasattr(x, attr):
            try:
                y = getattr(x, attr)()
                if hasattr(y, "detach"):
                    y = y.detach()
                return np.asarray(y.numpy() if hasattr(y, "numpy") else y)
            except Exception:                                # noqa: BLE001
                pass
    return np.asarray(x)


# =============================================================================
# 2. JIT
# =============================================================================


def jit(fn: Callable | None = None, *, backend: str = "auto",
        static_argnums=(), nopython: bool = True) -> Callable:
    """Compile ``fn`` with whatever compiler the backend has.

    * JAX backend  -> ``jax.jit`` (the function must be written on xp);
    * NumPy backend -> ``numba.njit`` when numba is installed and the
      function survives it, otherwise the function itself;
    * CuPy backend -> the function itself (CuPy fuses at the kernel level).

    A failed compilation is not an error: the plain function is returned
    and ``fn.__phycore_jit__`` says which path was taken, so a report can
    state it.
    """
    def deco(f: Callable) -> Callable:
        _, name = get_backend(backend)
        if name == "jax":
            jax = _try_import("jax")
            try:
                g = jax.jit(f, static_argnums=static_argnums)
                g.__phycore_jit__ = "jax.jit"
                return g
            except Exception:                                # noqa: BLE001
                pass
        elif name == "numpy":
            nb = _try_import("numba")
            if nb is not None:
                try:
                    g = nb.njit(f, cache=False) if nopython else nb.jit(f)
                    g.__phycore_jit__ = "numba.njit"
                    return g
                except Exception:                            # noqa: BLE001
                    pass
        f.__phycore_jit__ = "none"
        return f
    return deco if fn is None else deco(fn)


def grad(fn: Callable, argnums=0) -> Callable | None:
    """``jax.grad`` when the backend is JAX; ``None`` otherwise (the caller
    falls back to the learned critic, section 7.5 of the report)."""
    _, name = get_backend("auto")
    if name != "jax":
        return None
    return _try_import("jax").grad(fn, argnums=argnums)


# =============================================================================
# 3. Devices
# =============================================================================


def cuda_available(lib: str = "torch") -> bool:
    mod = _try_import(lib)
    if mod is None:
        return False
    try:
        if lib == "torch":
            return bool(mod.cuda.is_available())
        if lib == "cupy":
            return int(mod.cuda.runtime.getDeviceCount()) > 0
        if lib == "jax":
            return any(d.platform == "gpu" for d in mod.devices())
    except Exception:                                        # noqa: BLE001
        return False
    return False


def torch_device(want: str = "auto") -> str:
    """``"cuda"`` when asked for and available, else ``"cpu"``. ``auto``
    honours ``PHYCORE_DEVICE``."""
    want = (os.environ.get("PHYCORE_DEVICE", "") or want).lower()
    if want in ("", "auto"):
        return "cuda" if cuda_available("torch") else "cpu"
    if want.startswith("cuda") and not cuda_available("torch"):
        return "cpu"
    return want


def compile_module(module: Any, mode: str | None = None, sample=None) -> Any:
    """``torch.compile`` guarded: on a machine without a working inductor
    (no Triton on Windows CUDA, no C++ toolchain) it returns the module
    unchanged and marks it, rather than crashing a design run for the
    sake of a speed-up. ``torch.compile`` is lazy, so when ``sample`` is
    given the compiled module is exercised once here, where a failure
    can still be caught."""
    torch = _try_import("torch")
    if torch is None or not hasattr(torch, "compile"):
        return module
    if _COMPILE_STATE["error"] is not None:
        # already failed once on this machine: do not pay the attempt again
        return module
    try:
        out = torch.compile(module, mode=mode) if mode else torch.compile(module)
        if sample is not None:
            with torch.no_grad():
                out(sample)
        out.__phycore_jit__ = "torch.compile"
        _COMPILE_STATE["ok"] = True
        return out
    except Exception as exc:                                 # noqa: BLE001
        # e.g. TritonMissing on Windows CUDA, "cl is not found" on CPU
        _COMPILE_STATE["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
        try:
            module.__phycore_jit__ = "none"
        except Exception:                                    # noqa: BLE001
            pass
        return module


def compile_status() -> dict:
    """Whether ``torch.compile`` has worked on this machine in this
    process, and why not when it has not."""
    return dict(_COMPILE_STATE)


def device_info() -> dict:
    """What this machine can accelerate. Goes into run metadata."""
    torch = _try_import("torch")
    jax = _try_import("jax")
    info = {
        "numpy": np.__version__,
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": cuda_available("torch"),
        "torch_device_name": None,
        "torch_compile": bool(torch is not None and hasattr(torch, "compile")),
        "jax": getattr(jax, "__version__", None),
        "jax_gpu": cuda_available("jax"),
        "cupy": getattr(_try_import("cupy"), "__version__", None),
        "numba": getattr(_try_import("numba"), "__version__", None),
        "backend": get_backend("auto")[1],
        "torch_compile_status": compile_status(),
    }
    if info["torch_cuda"]:
        try:
            info["torch_device_name"] = torch.cuda.get_device_name(0)
        except Exception:                                    # noqa: BLE001
            pass
    return info


# =============================================================================
# 4. JAX -> torch bridge for a differentiable L0
# =============================================================================


def torch_from_jax(fn: Callable, n_outputs: int = 2) -> Callable:
    """Wrap a JAX function ``fn(theta, s) -> (F, G)`` so torch can
    back-propagate through it.

    Forward runs the JAX function and keeps its VJP; backward feeds the
    cotangents into that VJP. Data crosses via the host (NumPy), which is
    the robust path on every platform; for the batch sizes the actor uses
    (a few hundred rows) the copy is not the bottleneck, the physics is.
    The gradient only flows to ``theta``; the spec is a constant.
    """
    torch = _try_import("torch")
    jax = _try_import("jax")
    if torch is None or jax is None:
        raise ImportError("torch_from_jax necesita torch y jax instalados")
    jnp = importlib.import_module("jax.numpy")

    class _Bridge(torch.autograd.Function):
        @staticmethod
        def forward(ctx, theta_t, s_t):
            th = jnp.asarray(theta_t.detach().cpu().numpy())
            s = jnp.asarray(s_t.detach().cpu().numpy())
            outs, vjp = jax.vjp(lambda t: fn(t, s), th)
            ctx.vjp = vjp
            ctx.dev, ctx.dtype = theta_t.device, theta_t.dtype
            # torch hands back cotangents in ITS dtype (float32 in Phy-2); the
            # JAX function may run in float64: the VJP wants them matched
            ctx.out_dtypes = [o.dtype for o in outs]
            return tuple(torch.as_tensor(np.array(o), device=theta_t.device,
                                         dtype=theta_t.dtype) for o in outs)

        @staticmethod
        def backward(ctx, *cot):
            ct = tuple(jnp.asarray(c.detach().cpu().numpy(), dtype=d)
                       for c, d in zip(cot, ctx.out_dtypes))
            (g_theta,) = ctx.vjp(ct)
            return (torch.as_tensor(np.array(g_theta), device=ctx.dev,
                                    dtype=ctx.dtype), None)

    @functools.wraps(fn)
    def wrapped(theta_t, s_t):
        if not (torch.is_grad_enabled() and theta_t.requires_grad):
            # nothing will call backward: run the function alone. Linearising
            # it (jax.vjp) costs two to three forward passes and keeps every
            # residual alive for a backward that never comes (PSL's candidate
            # scoring runs under no_grad). Same function; the values can differ
            # from the linearised program's primal by XLA's compilation only
            # (measured <= 1e-8 relative on Phy-Prop's iterative L0).
            outs = fn(jnp.asarray(theta_t.detach().cpu().numpy()),
                      jnp.asarray(s_t.detach().cpu().numpy()))
            return tuple(torch.as_tensor(np.array(o), device=theta_t.device,
                                         dtype=theta_t.dtype) for o in outs)
        return _Bridge.apply(theta_t, s_t)
    return wrapped
