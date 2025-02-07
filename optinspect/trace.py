import itertools
import optax
from typing import Any, Callable, NamedTuple, Optional, Union
from .util import is_traced, make_key_func, maybe_skip_if_traced


class TraceState(NamedTuple):
    name: str
    value: Any


@maybe_skip_if_traced
def trace_on_update(
    name: str,
    key: Union[str, Callable] = "updates",
    init: Any = None,
    *,
    skip_if_traced: bool,
) -> optax.GradientTransformationExtraArgs:
    """
    Trace a gradient update.

    Args:
        name: Name of the traced state.
        key: Quantity to accumulate. If a string, accumulate a keyword argument. If a
            callable with the same signature as
            :meth:`~optax.GradientTransformationExtraArgs.update` (although only
            accepting keyword arguments), accumulate the returned value.
        init: Initial traced value, defaults to :code:`params` passed to :code:`init`.
        skip_if_traced: Skip value tracing if the state passed to :code`update` is
            traced.

    Returns:
        Gradient transformation.
    """
    key_func = make_key_func(key)

    def init_func(params: optax.Params) -> TraceState:
        return TraceState(name, params if init is None else init)

    def update_func(
        updates: optax.Updates,
        state: optax.EmptyState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> tuple[optax.Updates, optax.OptState]:
        skip = skip_if_traced and is_traced(state)
        if not skip:
            state = TraceState(
                name,
                key_func(updates=updates, params=params, **extra_args),
            )
        return updates, state

    return optax.GradientTransformationExtraArgs(init_func, update_func)


class WrapperTraceState(NamedTuple):
    inner_state: optax.OptState
    name: str
    value: Any


@maybe_skip_if_traced
def trace_wrapped(
    inner: optax.GradientTransformation,
    name: str,
    key: Union[str, Callable] = "updates",
    *,
    skip_if_traced: bool,
) -> optax.GradientTransformationExtraArgs:
    """
    Trace the state of a wrapped gradient transformation.

    Args:
        inner: Gradient transformation to wrap.
        name: Name of the traced state.
        key: Quantity to accumulate. If a string, accumulate a keyword argument. If a
            callable with the same signature as
            :meth:`~optax.GradientTransformationExtraArgs.update` (although only
            accepting keyword arguments), accumulate the returned value.
        skip_if_traced: Skip value tracing if the state passed to :code`update` is
            traced.

    Returns:
        Gradient transformation.
    """
    inner = optax.with_extra_args_support(inner)
    key_func = make_key_func(key)

    def init(params: optax.Params) -> WrapperTraceState:
        inner_state = inner.init(params)
        value = key_func(state=inner_state, params=params)
        return WrapperTraceState(inner_state, name, value)

    def update(
        updates: optax.Updates,
        state: WrapperTraceState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> tuple[optax.Updates, WrapperTraceState]:
        updates, inner_state = inner.update(
            updates, state.inner_state, params, **extra_args
        )
        skip = skip_if_traced and is_traced(updates)
        value = state.value if skip else key_func(inner_state)
        return updates, WrapperTraceState(inner_state, name, value)

    return optax.GradientTransformationExtraArgs(init, update)


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
