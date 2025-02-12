"""
The :mod:`~optinspect.trace` module implements gradient transformations
:func:`.trace_update` to trace updates, :func:`.trace_wrapped` to trace the state of a
wrapped gradient transformation, and :func:`.get_traced_values` to extract traced values
from the optimizer state.
"""

import optax
from typing import Any, Callable, NamedTuple, Optional, Union
from .inspect import inspect_update, inspect_wrapped, WrappedState
from .tag import _update_tagged_state, get_tagged_values
from .util import make_key_func


@_update_tagged_state
class TraceState(NamedTuple):
    """
    State for tracing values.
    """

    tag_: dict[str, None]
    """
    Unique tag of the traced value as a dictionary with a single key because strings
    are not valid jax types (cf. https://github.com/jax-ml/jax/issues/3045).
    """
    value: optax.Params
    """Accumulated value."""


def trace_update(
    tag: str,
    key: Union[str, int, Callable] = "updates",
    init: Any = None,
    *,
    skip_if_traced: Optional[bool] = None,
) -> optax.GradientTransformationExtraArgs:
    """
    Trace a gradient update.

    Args:
        tag: Tag for the traced state.
        key: Quantity to trace. If a callable with the same signature as
            :meth:`~optax.GradientTransformationExtraArgs.update`, trace the
            returned value. If a string, trace arguments by name. If an integer,
            trace arguments by their position.
        init: Initial traced value, defaults to :code:`params` passed to :code:`init`.
        skip_if_traced: Skip if the state passed to :code`update` is traced.

    Returns:
        Gradient transformation.

    Example:
        >>> import jax
        >>> from jax import numpy as jnp
        >>> import optinspect
        >>>
        >>> optim = optinspect.trace_update(
        ...     "updates_and_value",
        ...     lambda updates, *args, value, **kwargs: {
        ...         "updates": updates, "value": value
        ...     }
        ... )
        >>> params = 3.0
        >>> value_and_grad = jax.value_and_grad(jnp.square)
        >>> state = optim.init(params)
        >>> value, grad = value_and_grad(params)
        >>> updates, state = optim.update(grad, state, params, value=value)
        >>> optinspect.get_traced_values(state)
        {'updates_and_value': {'updates': Array(6., ...), 'value': Array(9., ...)}}
    """
    key_func = make_key_func(key)

    def _init(params: optax.Params) -> TraceState:
        return TraceState({tag: None}, params if init is None else init)

    def _update(
        updates: optax.Updates,
        state: optax.EmptyState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> tuple[optax.Updates, optax.OptState]:
        return TraceState({tag: None}, key_func(updates, state, params, **extra_args))

    return inspect_update(_update, _init, skip_if_traced=skip_if_traced)


def trace_wrapped(
    inner: optax.GradientTransformation,
    tag: str,
    key: Union[str, int, Callable] = "updates",
    *,
    skip_if_traced: Optional[bool] = None,
) -> optax.GradientTransformationExtraArgs:
    """
    Trace the state of a wrapped gradient transformation after an update.

    Args:
        inner: Gradient transformation to wrap.
        tag: Tag for the traced state.
        key: Quantity to trace. If a callable with the same signature as
            :meth:`~optax.GradientTransformationExtraArgs.update`, trace the
            returned value. The second :code:`state` argument is the state of the
            wrapped transformation. If a string, trace arguments by name. If an integer,
            trace arguments by their position.
        skip_if_traced: Skip if the state passed to :code`update` is traced.

    Returns:
        Gradient transformation.

    Example:
        >>> import jax
        >>> from jax import numpy as jnp
        >>> import optax
        >>> import optinspect
        >>>
        >>> optim = optinspect.trace_wrapped(
        ...     optax.adam(0.1),
        ...     "second_moment",
        ...     lambda _, state, *args, **kwargs: state[0].nu,
        ... )
        >>> params = 3.0
        >>> value_and_grad = jax.value_and_grad(jnp.square)
        >>> state = optim.init(params)
        >>> value, grad = value_and_grad(params)
        >>> updates, state = optim.update(grad, state, params, value=value)
        >>> optinspect.get_traced_values(state)
        {'second_moment': Array(0.036, ...)}
    """
    key_func = make_key_func(key)

    def _init(params: optax.Params, inner_state: optax.OptState) -> TraceState:
        value = key_func(None, inner_state, params)
        return TraceState({tag: None}, value)

    def _update(
        updates: optax.Updates,
        state: WrappedState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> Any:
        return TraceState(
            {tag: None}, key_func(updates, state.inner, params, **extra_args)
        )

    return inspect_wrapped(inner, _update, _init, skip_if_traced=skip_if_traced)


def get_traced_values(
    state: optax.OptState, tag: Optional[Any] = None
) -> dict[str, Any]:
    """
    Extract traced values from an optimizer state.

    Args:
        state: Optimizer state.
        tag: Tag of the state to extract. If specified, return only the requested traced
            value.

    Returns:
        Dictionary mapping tag names to traced values.
    """
    return get_tagged_values(state, tag=tag, tuple_name="TraceState")
