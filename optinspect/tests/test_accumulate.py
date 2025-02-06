import jax
from jax import numpy as jnp
import optinspect
import pytest
from typing import Any, Callable


@pytest.mark.parametrize(
    "accumulate_func, expected",
    [
        (optinspect.accumulate_cumulative_average(), 4.5),
        (optinspect.accumulate_cumulative_average(period=4), 8.5),
        (optinspect.accumulate_most_recent(lambda **kwargs: kwargs["state"].count), 9),
    ],
)
@pytest.mark.parametrize("scan", [False, True], ids=["loop", "scan"])
def test_accumulate(
    accumulate_func: Callable, expected: Any, skip_if_traced: bool, scan: bool
) -> None:
    n = 10
    x = 4
    optim = optinspect.accumulate_on_update(
        accumulate_func, skip_if_traced=skip_if_traced
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


def test_accumulate_invalid_key() -> None:
    with pytest.raises(ValueError, match="must be a string or callable"):
        optinspect.accumulate_cumulative_average(key=9)
