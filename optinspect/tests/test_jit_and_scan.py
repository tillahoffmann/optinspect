import functools
import jax
import optax
import optinspect
import pytest


@pytest.mark.parametrize("jit", [False, True], ids=["not-jitted", "jitted"])
@pytest.mark.parametrize("skip_if_traced", [False, True], ids=["skip", "do"])
@pytest.mark.parametrize(
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
                optinspect.accumulate_update, optinspect.accumulate_cumulative_average()
            ),
            False,
        ),
        (functools.partial(optinspect.trace_update, "grad"), False),
        (
            functools.partial(
                optinspect.trace_wrapped,
                name="nu",
                key=lambda _, state, *args, **kwargs: state.nu,
            ),
            True,
        ),
    ],
)
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
