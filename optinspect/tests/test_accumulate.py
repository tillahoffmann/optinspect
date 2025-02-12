import jax
from jax import numpy as jnp
import optax
import optinspect
import pytest
from typing import Any, Callable


@pytest.mark.parametrize(
    "accumulate_func, expected",
    [
        (optinspect.accumulate_cumulative_average(), 4.5),
        (optinspect.accumulate_cumulative_average(period=4), 8.5),
        (
            optinspect.accumulate_most_recent(),
            9,
        ),
        (optinspect.accumulate_cumulative_average(0), 4.5),
    ],
)
@pytest.mark.parametrize("scan", [False, True], ids=["loop", "scan"])
def test_accumulate_update(
    accumulate_func: Callable, expected: Any, skip_if_traced: bool, scan: bool
) -> None:
    n = 10
    x = 4
    optim = optinspect.accumulate_update(
        "tag", accumulate_func, skip_if_traced=skip_if_traced
    )
    state = optim.init(x)

    if scan:
        (updated, state), _ = jax.lax.scan(
            lambda carry, grad: (optim.update(grad, carry[1], x), None),
            (x, state),
            jnp.arange(n),
        )
        if skip_if_traced:
            assert jnp.allclose(state.value, x)
            assert state.count == 0
        else:
            assert jnp.allclose(state.value, expected)
            assert state.count == n
    else:
        for grad in jnp.arange(n):
            updated, state = optim.update(grad, state, x)
            assert jnp.allclose(updated, grad), "Gradient must be unchanged."
        assert state.count == n
        assert jnp.allclose(state.value, expected)

    # Get the traced values.
    if not skip_if_traced or not scan:
        assert jnp.allclose(optinspect.get_accumulated_values(state)["tag"], expected)

    # Check that resetting works.
    state = optinspect.reset_accumulate_count(state)
    assert state.count == 0


def test_accumulate_wrapped() -> None:
    optim = optinspect.accumulate_wrapped(
        optax.adam(0.1),
        "mu",
        optinspect.accumulate_cumulative_average(),
        lambda _, state, *args, **kwargs: state[0].mu,
    )
    state = optim.init(3.0)

    expected = 0.0
    n = 5
    for grad in jnp.arange(n):
        _, state = optim.update(grad, state)
        expected += state.inner[0].mu

    expected /= n
    actual = optinspect.get_accumulated_values(state, "mu")
    assert jnp.allclose(actual, expected)  # type: ignore
