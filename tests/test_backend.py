import numpy as np
import pytest

from phycore import backend as be


def test_get_backend_numpy_explicit():
    xp, name = be.get_backend("numpy")
    assert xp is np and name == "numpy"
    with pytest.raises(ValueError):
        be.get_backend("fortran")


def test_jit_falls_back_gracefully():
    @be.jit(backend="numpy")
    def f(x):
        return x * 2.0
    assert float(f(np.float64(2.0))) == 4.0
    assert getattr(f, "__phycore_jit__", "none") in ("none", "numba.njit")


def test_torch_device_and_info():
    d = be.torch_device("cpu")
    assert d == "cpu"
    assert be.torch_device("cuda") in ("cuda", "cpu")
    info = be.device_info()
    assert isinstance(info["torch_cuda"], bool) and info["backend"] in ("numpy", "jax", "cupy")


def test_to_numpy_torch():
    torch = pytest.importorskip("torch")
    t = torch.arange(3.0, requires_grad=True)
    assert be.to_numpy(t).tolist() == [0.0, 1.0, 2.0]


def test_compile_module_guarded():
    torch = pytest.importorskip("torch")
    net = torch.nn.Linear(2, 1)
    out = be.compile_module(net, sample=torch.zeros(3, 2))
    y = out(torch.zeros(3, 2))
    assert y.shape == (3, 1)


def test_torch_from_jax_bridge():
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    def f(t, s):
        return jnp.stack([(t ** 2).sum(-1) + s[..., 0], t[..., 0]], -1), (t - 0.5)

    g = be.torch_from_jax(f)
    th = torch.tensor([[0.3, 0.4]], requires_grad=True)
    s = torch.tensor([[1.0]])
    F, G = g(th, s)
    F[:, 0].sum().backward()
    assert torch.allclose(th.grad, 2 * th.detach())
