import jax
from jax import numpy as jnp
import optax
import optinspect
import pytest
import re


def test_print_update(capsys: pytest.CaptureFixture, skip_if_traced: bool):
    x = jnp.array(4.0)
    value_and_grad = jax.value_and_grad(jnp.square)
    value, grad = value_and_grad(x)

    optim = optax.chain(
        optinspect.print_update(
            "1: updates={updates}, value={value}", skip_if_traced=skip_if_traced
        ),
        optax.sgd(1e-3),
        optinspect.print_update(
            "2: updates={updates}, value={value}", skip_if_traced=skip_if_traced
        ),
    )

    # Apply a non-traced update.
    state = optim.init(x)
    optim.update(grad, state, x, value=value)
    out, _ = capsys.readouterr()
    assert re.search(r"1: updates=8.0\d*, value=16.0", out)
    assert re.search(r"2: updates=-0.0080\d*, value=16.0", out)

    # Apply a traced update.
    jax.jit(optim.update)(grad, state, x, value=value)
    out, _ = capsys.readouterr()
    if skip_if_traced:
        assert not out
    else:
        assert "1: updates=Traced<ShapedArray(float32[], weak_type=True)>" in out
        assert "2: updates=Traced<ShapedArray(float32[], weak_type=True)>" in out


def test_print_wrapped(capsys: pytest.CaptureFixture, skip_if_traced: bool) -> None:
    x = jnp.array(4.0)
    value_and_grad = jax.value_and_grad(jnp.square)
    value, grad = value_and_grad(x)
    optim = optinspect.print_wrapped(
        optax.adam(0.1),
        "after: {state.inner}",
        skip_if_traced=skip_if_traced,
    )

    # Apply a non-traced update.
    state = optim.init(x)
    optim.update(grad, state, x, value=value)
    out, _ = capsys.readouterr()
    assert "after: (ScaleByAdamState(count=Array(1, dtype=int32)" in out

    # Apply a traced update.
    jax.jit(optim.update)(grad, state, x, value=value)
    out, _ = capsys.readouterr()
    if skip_if_traced:
        assert not out
    else:
        assert "after: (ScaleByAdamState(count=Traced<ShapedArray(int32[])>" in out
