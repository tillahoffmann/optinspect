import jax
from jax import numpy as jnp
import optax
import optinspect
import os
from unittest import mock
import pytest


def test_is_traced() -> None:
    x = {"a": 3.0, "b": (2.0, 1.0)}
    assert not optinspect.is_traced(x)
    assert jax.jit(optinspect.is_traced)(x)


def test_before_after_update_missing() -> None:
    with pytest.raises(ValueError, match="At least one of"):
        optinspect.before_after_update(None)


def test_before_after_update(skip_if_traced: bool) -> None:
    x = jnp.array(4.0)
    value_and_grad = jax.value_and_grad(jnp.square)
    _, grad = value_and_grad(x)

    before: list[optax.ScaleByAdamState] = []
    after: list[optax.ScaleByAdamState] = []
    optim = optinspect.before_after_update(
        optax.scale_by_adam(0.1),
        lambda state, **kwargs: before.append(state),
        lambda state, **kwargs: after.append(state),
        skip_if_traced=skip_if_traced,
    )
    state = optim.init(x)
    optim.update(grad, state, x)
    assert before[0].count == 0
    assert after[0].count == 1

    if skip_if_traced:
        jax.jit(optim.update)(grad, state, x)
        assert len(before) == 1 and len(after) == 1
    else:
        # This fails because the before and after append functions have side effects.
        with pytest.raises(Exception, match="Leaked trace"), jax.check_tracer_leaks():
            jax.jit(optim.update)(grad, state, x)


def test_maybe_skip_if_traced() -> None:
    with pytest.raises(ValueError, match="does not have a parameter"):
        optinspect.util.maybe_skip_if_traced(lambda: None)
    with pytest.raises(ValueError, match="must be keyword-only"):
        optinspect.util.maybe_skip_if_traced(lambda skip_if_traced: None)
    with pytest.raises(ValueError, match="must not have a default value"):
        optinspect.util.maybe_skip_if_traced(lambda *, skip_if_traced="foo": None)

    func = optinspect.util.maybe_skip_if_traced(
        lambda *, skip_if_traced: skip_if_traced
    )
    assert (
        "INSPECT_IF_TRACED" not in os.environ
    ), "Environment variable `INSPECT_IF_TRACED` must not be set for tests."

    assert func(skip_if_traced=False) is False
    assert func(skip_if_traced=True) is True
    assert func() is True

    with mock.patch.dict(os.environ, {"INSPECT_IF_TRACED": "true"}):
        assert func(skip_if_traced=False) is False
        assert func(skip_if_traced=True) is True
        assert func() is False
