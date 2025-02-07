import itertools
import optax
from typing import Any, Callable, NamedTuple, Optional, Union
from .util import (
    inspect_update,
    inspect_wrapped,
    make_key_func,
    WrappedState,
)


class TraceState(NamedTuple):
    """
    State for tracing values.

    Attributes:
        name: Unique name of the traced value.
        value: Traced value.
    """

    name: str
    value: Any


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
    """
    key_func = make_key_func(key)

    def _init(params: optax.Params) -> TraceState:
        return TraceState(name, params if init is None else init)

    def _update(
        updates: optax.Updates,
        state: optax.EmptyState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> tuple[optax.Updates, optax.OptState]:
        return updates, TraceState(name, key_func(updates, state, params, **extra_args))

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
    """
    inner = optax.with_extra_args_support(inner)
    key_func = make_key_func(key)

    def _init(params: optax.Params) -> WrappedState:
        inner_state = inner.init(params)
        value = key_func(state=inner_state, params=params)
        return WrappedState(inner_state, TraceState(name, value))

    def _update(
        updates: optax.Updates,
        state: WrappedState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> Any:
        return TraceState(name, key_func(state.inner))

    return inspect_wrapped(inner, _update, _init, skip_if_traced=skip_if_traced)


def get_trace(state: optax.OptState) -> dict[str, Any]:
    """
    Extract traced values from an optimizer state.

    Args:
        state: Optimizer state.

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
        if state.name in trace:
            raise ValueError(f"Duplicate name `{state.name}` in trace.")
        trace[state.name] = state.value
    return trace
