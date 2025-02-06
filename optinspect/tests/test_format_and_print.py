import jax
from jax import numpy as jnp
import optax
import optinspect
import pytest
import re


@pytest.mark.parametrize("skip_if_traced", [True, False])
def test_print_on_update(capsys: pytest.CaptureFixture, skip_if_traced: bool):
    x = jnp.array(4.0)
    value_and_grad = jax.value_and_grad(jnp.square)
    value, grad = value_and_grad(x)

    optim = optax.chain(
        optinspect.print_on_update(
            "1: updates={updates}, value={value}", skip_if_traced=skip_if_traced
        ),
        optax.sgd(1e-3),
        optinspect.print_on_update(
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
        assert (
            "1: updates=Traced<ShapedArray(float32[], weak_type=True)>with"
            "<DynamicJaxprTrace>, value=Traced<ShapedArray(float32[], weak_type=True)>"
            "with<DynamicJaxprTrace>"
        ) in out
        assert (
            "2: updates=Traced<ShapedArray(float32[], weak_type=True)>with"
            "<DynamicJaxprTrace>, value=Traced<ShapedArray(float32[], weak_type=True)>"
            "with<DynamicJaxprTrace>"
        ) in out


@pytest.mark.parametrize("skip_if_traced", [False, True])
def test_print_before_after_update(
    capsys: pytest.CaptureFixture, skip_if_traced: bool
) -> None:
    x = jnp.array(4.0)
    value_and_grad = jax.value_and_grad(jnp.square)
    value, grad = value_and_grad(x)
    optim = optinspect.print_before_after_update(
        optax.adam(0.1),
        before_format="before: {state}",
        after_format="after: {state}",
        skip_if_traced=skip_if_traced,
    )

    # Apply a non-traced update.
    state = optim.init(x)
    optim.update(grad, state, x, value=value)
    out, _ = capsys.readouterr()
    assert "before: (ScaleByAdamState(count=Array(0, dtype=int32)" in out
    assert "after: (ScaleByAdamState(count=Array(1, dtype=int32)" in out

    # Apply a traced update.
    jax.jit(optim.update)(grad, state, x, value=value)
    out, _ = capsys.readouterr()
    if skip_if_traced:
        assert not out
    else:
        assert "before: (ScaleByAdamState(count=Traced<ShapedArray(int32[])>" in out
        assert "after: (ScaleByAdamState(count=Traced<ShapedArray(int32[])>" in out
