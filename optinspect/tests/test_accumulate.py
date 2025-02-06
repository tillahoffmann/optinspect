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
def test_accumulate(accumulate_func: Callable, expected: Any) -> None:
    x = jnp.array(4.0)
    optim = optinspect.accumulate_on_update(accumulate_func)
    state = optim.init(x)

    n = 10
    for grad in jnp.arange(n):
        updated, state = optim.update(grad, state, x)
        assert jnp.allclose(updated, grad), "Gradient must be unchanged."

    assert state.count == n
    assert jnp.allclose(state.value, expected)


def test_accumulate_invalid_key() -> None:
    with pytest.raises(ValueError, match="must be a string or callable"):
        optinspect.accumulate_cumulative_average(key=9)
