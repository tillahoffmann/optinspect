"""
The :mod:`~optinspect.inspect` module implements the foundational gradient
transforations :func:`.inspect_update` to inspect updates and :func:`.inspect_wrapped`
to inspect the state of a wrapped gradient transformation. Both functions leave updates
unchanged to guarantee that inspecting the gradient transformations does not change
their behavior. By default, the inspection gradient transformations are skipped when the
code is jit-compiled to reduce overheads---your instrumented code will run just as fast
when jit-compiled.
"""

import optax
import os
from typing import Any, NamedTuple, Optional
from .util import is_traced


def _get_skip(value: Any, skip_if_traced: Optional[bool]) -> bool:
    if not is_traced(value):
        return False
    if skip_if_traced is None:
        skip_if_traced = "INSPECT_IF_TRACED" not in os.environ
    return skip_if_traced


def inspect_update(
    update: optax.TransformUpdateExtraArgsFn,
    init: Optional[optax.TransformInitFn] = None,
    *,
    skip_if_traced: bool = None,
) -> optax.GradientTransformationExtraArgs:
    """
    Call a function and leave updates unchanged.

    Args:
        func: Function with the same signature as
            :meth:`~optax.GradientTransformationExtraArgs.update`, returning the updated
            inspection state. If no value is returned, it is replaced by an
            :class:`~optax.EmptyState`.
        init: Function with the same signature as
            :meth:`~optax.GradientTransformationExtraArgs.init` or :code:`None` to
            initialize with an :class:`~optax.EmptyState`.
        skip_if_traced: Skip the :code:`update` function if the :code:`updates`
            argument is traced.

    Returns:
        Gradient transformation.

    Example:
        >>> import jax
        >>> from jax import numpy as jnp
        >>> import optinspect
        >>>
        >>> optim = optinspect.inspect_update(
        ...     lambda *args, **kwargs: print(f"args: {args}, kwargs: {kwargs}")
        ... )
        >>> params = 3.0
        >>> value_and_grad = jax.value_and_grad(jnp.square)
        >>> state = optim.init(params)
        >>> value, grad = value_and_grad(params)
        >>> updates, state = optim.update(grad, state, params, value=value)
        args: (Array(6., dtype=float32, weak_type=True), EmptyState(), 3.0),
        kwargs: {'value': Array(9., dtype=float32, weak_type=True)}
    """

    _init = init or (lambda _: optax.EmptyState())

    def _update(
        updates: optax.Updates,
        state: optax.EmptyState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> tuple[optax.Updates, optax.OptState]:
        if not _get_skip(updates, skip_if_traced):
            state = update(updates, state, params, **extra_args)
            if state is None:
                state = optax.EmptyState()
        return updates, state

    return optax.GradientTransformationExtraArgs(_init, _update)


class WrappedState(NamedTuple):
    """
    State wrapping another gradient transformation.
    """

    inner: optax.OptState
    """State of the wrapped optimizer."""
    outer: optax.OptState
    """Additional state information."""


def inspect_wrapped(
    inner: optax.GradientTransformationExtraArgs,
    update: optax.TransformUpdateExtraArgsFn,
    init: Optional[optax.TransformInitFn] = None,
    *,
    skip_if_traced: bool = None,
) -> optax.GradientTransformationExtraArgs:
    """
    Call a function and leave the updates unchanged.

    Args:
        update: Function with the same signature as
            :meth:`~optax.GradientTransformationExtraArgs.update` receiving a
            :class:`.WrappedState` after the wrapped transformation has been applied. It
            must return the updated :code:`outer` state. If no value is returned, it is
            replaced by an :class:`~optax.EmptyState`.
        init: Function with the same signature as
            :meth:`~optax.GradientTransformationExtraArgs.init` or :code:`None` to
            initialize with a :class:`.WrappedState` whose :code:`outer` state is
            :class:`~optax.EmptyState`.
        skip_if_traced: Skip the :code:`update` function if the :code:`updates`
            argument is traced.

    Returns:
        Gradient transformation.

    Example:
        >>> import jax
        >>> from jax import numpy as jnp
        >>> import optax
        >>> import optinspect
        >>>
        >>> optim = optinspect.inspect_wrapped(
        ...     optax.adam(0.1),
        ...     lambda _, state, *args, **kwargs: print(state.inner[0])
        ... )
        >>> params = 3.0
        >>> value_and_grad = jax.value_and_grad(jnp.square)
        >>> state = optim.init(params)
        >>> value, grad = value_and_grad(params)
        >>> updates, state = optim.update(grad, state, params, value=value)
        ScaleByAdamState(count=Array(1, dtype=int32),
                         mu=Array(0.6, dtype=float32, weak_type=True),
                         nu=Array(0.036, dtype=float32, weak_type=True))
    """
    inner = optax.with_extra_args_support(inner)
    _init = init or (
        lambda params: WrappedState(inner.init(params), optax.EmptyState())
    )

    def _update(
        updates: optax.Updates,
        state: WrappedState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> tuple[optax.Updates, optax.OptState]:
        updates, inner_state = inner.update(updates, state.inner, params, **extra_args)
        state = WrappedState(inner_state, state.outer)
        if not _get_skip(updates, skip_if_traced):
            outer_state = update(updates, state, params, **extra_args)
            if outer_state is None:
                outer_state = optax.EmptyState()
            state = WrappedState(inner_state, outer_state)
        return updates, state

    return optax.GradientTransformationExtraArgs(_init, _update)
