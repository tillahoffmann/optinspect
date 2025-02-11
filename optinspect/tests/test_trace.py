from jax import numpy as jnp
import optax
import optinspect
import pytest
from typing import Callable


def test_trace(value_and_grad_and_params: tuple[Callable, optax.Params]) -> None:
    value_and_grad, params = value_and_grad_and_params

    optim = optax.chain(
        optinspect.trace_update("step 1"),
        optax.scale_by_adam(),
        optinspect.trace_update("step 2", key="value"),
        optax.scale_by_learning_rate(0.1),
        optinspect.trace_update("step 3"),
    )

    state = optim.init(params)
    for _ in range(5):
        # Update state.
        value, grad = value_and_grad(params)
        updates, state = optim.update(grad, state, params, value=value)

        # Validate trace.
        trace = optinspect.get_traced_values(state)
        assert jnp.allclose(trace["step 2"], value)

        # Update parameters.
        params = optax.apply_updates(params, updates)

    # Check we can get information directly from the state.
    assert jnp.allclose(optinspect.get_traced_values(state, "step 2"), value)


def test_trace_duplicate_key() -> None:
    optim = optax.chain(
        optinspect.trace_update("step 1"),
        optax.scale_by_adam(),
        optinspect.trace_update("step 1"),
    )
    state = optim.init(4.0)
    state = optim.update(3.0, state)
    with pytest.raises(ValueError, match="Duplicate tag `step 1`."):
        optinspect.get_traced_values(state)


def test_trace_wrapped(
    value_and_grad_and_params: tuple[Callable, optax.Params],
) -> None:
    value_and_grad, params = value_and_grad_and_params
    optim = optinspect.trace_wrapped(
        optax.adam(0.1), "mu", key=lambda _, state, *args, **kwargs: state[0].mu
    )
    state = optim.init(params)
    value, grad = value_and_grad(params)
    _, state = optim.update(grad, state, params, value=value)
    trace = optinspect.get_traced_values(state)
    assert jnp.allclose(trace["mu"], state.inner[0].mu)
