import jax
from jax import numpy as jnp
import optax
import pytest
from typing import Callable


@pytest.fixture(params=[False, True], ids=["do_if_traced", "skip_if_traced"])
def skip_if_traced(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(
    params=[
        (jnp.square, jnp.array(4.0)),
    ]
)
def value_and_grad_and_params(
    request: pytest.FixtureRequest,
) -> tuple[Callable, optax.Params]:
    loss, params = request.param
    return jax.value_and_grad(loss), params
