"""
The :mod:`~optinspect.trace` module implements gradient transformations
:func:`.trace_update` to trace updates, :func:`.trace_wrapped` to trace the state of a
wrapped gradient transformation, and :func:`.get_trace` to extract traced values from
the optimizer state.
"""

import itertools
import optax
from typing import Any, Callable, NamedTuple, Optional, Union
from .inspect import inspect_update, inspect_wrapped, WrappedState
from .util import make_key_func


class TraceState(NamedTuple):
    """
    State for tracing values.
    """

    name: dict[str, None]
    """
    Unique name of the traced value as a dictionary with a single key because strings
    are not valid jax types (cf. https://github.com/jax-ml/jax/issues/3045).
    """
    value: Any
    """Traced value."""


def trace_update(
    name: str,
    key: Union[str, Callable] = "updates",
    init: Any = None,
    *,
    skip_if_traced: bool = None,
) -> optax.GradientTransformationExtraArgs:
    """
    Trace a gradient update.

    Args:
        name: Name of the traced state.
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
        >>> optinspect.get_trace(state)
        {'updates_and_value': {'updates': Array(6., dtype=float32, weak_type=True),
                               'value': Array(9., dtype=float32, weak_type=True)}}
    """
    key_func = make_key_func(key)

    def _init(params: optax.Params) -> TraceState:
        return TraceState({name: None}, params if init is None else init)

    def _update(
        updates: optax.Updates,
        state: optax.EmptyState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> tuple[optax.Updates, optax.OptState]:
        return updates, TraceState(
            {name: None}, key_func(updates, state, params, **extra_args)
        )

    return inspect_update(_update, _init, skip_if_traced=skip_if_traced)


def trace_wrapped(
    inner: optax.GradientTransformation,
    name: str,
    key: Union[str, Callable] = "updates",
    *,
    skip_if_traced: bool = None,
) -> optax.GradientTransformationExtraArgs:
    """
    Trace the state of a wrapped gradient transformation after an update.

    Args:
        inner: Gradient transformation to wrap.
        name: Name of the traced state.
        key: Quantity to trace. If a callable with the same signature as
            :meth:`~optax.GradientTransformationExtraArgs.update`, trace the
            returned value. If a string, trace arguments by name. If an integer,
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
        >>> optinspect.get_trace(state)
        {'second_moment': Array(0.036, dtype=float32, weak_type=True)}
    """
    inner = optax.with_extra_args_support(inner)
    key_func = make_key_func(key)

    def _init(params: optax.Params) -> WrappedState:
        inner_state = inner.init(params)
        value = key_func(None, inner_state, params)
        return WrappedState(inner_state, TraceState({name: None}, value))

    def _update(
        updates: optax.Updates,
        state: WrappedState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> Any:
        return TraceState(
            {name: None}, key_func(updates, state.inner, params, **extra_args)
        )

    return inspect_wrapped(inner, _update, _init, skip_if_traced=skip_if_traced)


def get_trace(state: optax.OptState, name: Optional[Any] = None) -> dict[str, Any]:
    """
    Extract traced values from an optimizer state.

    Args:
        state: Optimizer state.
        name: Name of the state to extract. If specified, return only the requested
            traced value.

    Returns:
        Dictionary mapping names to traced values.
    """
    all_with_path = itertools.chain(
        optax.tree_utils.tree_get_all_with_path(state, "TraceState"),
        optax.tree_utils.tree_get_all_with_path(state, "WrapperTraceState"),
    )
    trace = {}
    state: TraceState
    for _, state in all_with_path:
        (current_name,) = state.name
        if name is not None and name == current_name:
            return state.value
        if current_name in trace:
            raise ValueError(f"Duplicate name `{current_name}` in trace.")
        trace[current_name] = state.value
    return trace
