import jax
from jax import numpy as jnp
import optax
import optinspect
import os
import pytest
import re
from typing import Optional
from unittest import mock


def test_is_traced() -> None:
    x = {"a": 3.0, "b": (2.0, 1.0)}
    assert not optinspect.is_traced(x)
    assert jax.jit(optinspect.is_traced)(x)


def test_wrapped(value_and_grad_and_params, skip_if_traced: bool) -> None:
    value_and_grad, params = value_and_grad_and_params
    _, grad = value_and_grad(params)

    update: list[optax.ScaleByAdamState] = []
    optim = optinspect.inspect_wrapped(
        optax.scale_by_adam(0.1),
        lambda updates, state, *args, **kwargs: update.append(state.inner),
        skip_if_traced=skip_if_traced,
    )
    state = optim.init(params)
    optim.update(grad, state, params)
    assert update[0].count == 1

    if skip_if_traced:
        jax.jit(optim.update)(grad, state)
        assert len(update) == 1
    else:
        # This fails because the `append` function has side effects.
        with pytest.raises(Exception, match="Leaked trace"), jax.check_tracer_leaks():
            jax.jit(optim.update)(grad, state)


@pytest.mark.parametrize(
    "jit, skip_if_traced, env, should_skip",
    [
        (False, False, False, False),
        (False, True, False, False),
        (False, None, False, False),
        (False, False, True, False),
        (False, True, True, False),
        (False, None, True, False),
        (True, False, False, False),
        (True, True, False, True),
        (True, None, False, True),
        (True, False, True, False),
        (True, True, True, True),
        (True, None, True, False),
    ],
)
def test_get_skip(
    jit: bool, skip_if_traced: Optional[bool], env: bool, should_skip: bool
) -> None:
    assert (
        "INSPECT_IF_TRACED" not in os.environ
    ), "Environment variable `INSPECT_IF_TRACED` must not be set for tests."

    x = 3

    def update(updates, state, params=None, **extra_args):
        if optinspect.inspect._get_skip(updates, skip_if_traced):
            return updates, state
        else:
            return -updates, -state

    if jit:
        update = jax.jit(update)

    if env:
        with mock.patch.dict(os.environ, {"INSPECT_IF_TRACED": "true"}):
            result = update(x, x)
    else:
        result = update(x, x)

    if should_skip:
        assert result == (x, x)
    else:
        assert result == (-x, -x)


def test_frepr() -> None:
    a = 3
    actual = optinspect.frepr(lambda x: a + x)
    assert re.match(
        r"<function optinspect\.tests\.test_util\.test_frepr\.<locals>\.<lambda>\(x\), "
        r"file '.*?optinspect/tests/test_util\.py', line \d+, 1 free var>",
        actual,
    )


def test_invalid_key_func() -> None:
    with pytest.raises(ValueError, match="must be a string, integer, or callable"):
        optinspect.util.make_key_func(1.3)
