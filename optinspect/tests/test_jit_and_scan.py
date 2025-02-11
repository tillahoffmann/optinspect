import functools
import jax
from jax import numpy as jnp
import optax
import optinspect
import pytest


mark_transformations = pytest.mark.parametrize(
    "transformation, wrapped",
    [
        (functools.partial(optinspect.print_update, format="value: {value}"), False),
        (
            functools.partial(
                optinspect.print_wrapped, format="value: {state.inner.nu}"
            ),
            True,
        ),
        (
            functools.partial(
                optinspect.accumulate_update,
                "tag",
                optinspect.accumulate_cumulative_average(),
            ),
            False,
        ),
        (functools.partial(optinspect.trace_update, "grad"), False),
        (
            functools.partial(
                optinspect.trace_wrapped,
                tag="nu",
                key=lambda _, state, *args, **kwargs: state.nu,
            ),
            True,
        ),
    ],
)


@pytest.mark.parametrize("jit", [False, True], ids=["not-jitted", "jitted"])
@mark_transformations
def test_jit(
    jit: bool, value_and_grad_and_params, transformation, wrapped, skip_if_traced: bool
) -> None:
    value_and_grad, params = value_and_grad_and_params
    if wrapped:
        optim = optax.chain(
            transformation(optax.scale_by_adam(), skip_if_traced=skip_if_traced),
            optax.scale_by_learning_rate(0.01),
        )
    else:
        optim = optax.chain(
            optax.scale_by_adam(),
            transformation(skip_if_traced=skip_if_traced),
            optax.scale_by_learning_rate(0.01),
        )

    update = jax.jit(optim.update) if jit else optim.update

    state = optim.init(params)
    for _ in range(7):
        value, grad = value_and_grad(params)
        updates, state = update(grad, state, params, value=value)
        params = optax.apply_updates(params, updates)


@mark_transformations
@pytest.mark.parametrize("jit", [False, True], ids=["not-jitted", "jitted"])
def test_with_and_without_identical(
    value_and_grad_and_params, transformation, wrapped, jit
) -> None:
    value_and_grad, params = value_and_grad_and_params
    base_optim = optax.scale_by_adam()
    base_lr = optax.scale_by_learning_rate(0.01)
    if wrapped:
        optim = optax.chain(
            transformation(base_optim),
            base_lr,
        )
    else:
        optim = optax.chain(
            base_optim,
            transformation(),
            base_lr,
        )

    def _run(optim, params):
        state = optim.init(params)
        for _ in range(7):
            value, grad = value_and_grad(params)
            updates, state = optim.update(grad, state, params, value=value)
            params = optax.apply_updates(params, updates)
        return params

    if jit:
        _run = jax.jit(_run, static_argnames=["optim"])

    params1 = _run(optim, params)
    params2 = _run(optax.chain(base_optim, base_lr), params)
    assert jnp.allclose(params1, params2)


@mark_transformations
def test_scan(
    value_and_grad_and_params, transformation, wrapped, skip_if_traced: bool
) -> None:
    value_and_grad, params = value_and_grad_and_params
    base_optim = optax.scale_by_adam()
    base_lr = optax.scale_by_learning_rate(0.01)
    if wrapped:
        optim = optax.chain(
            transformation(base_optim, skip_if_traced=skip_if_traced),
            base_lr,
        )
    else:
        optim = optax.chain(
            base_optim,
            transformation(skip_if_traced=skip_if_traced),
            base_lr,
        )

    state = optim.init(params)

    def _body(carry, _):
        params, state = carry
        value, grad = value_and_grad(params)
        updates, state = optim.update(grad, state, params, value=value)
        params = optax.apply_updates(params, updates)
        return (params, state), _

    params, state = jax.lax.scan(_body, (params, state), jnp.arange(10))
