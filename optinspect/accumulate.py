import jax
import optax
from typing import Any, Callable, NamedTuple, Optional, Union
from .util import is_traced


def _make_key_func(key: Union[str, int, Callable]) -> Callable:
    if isinstance(key, str):
        return lambda **kwargs: kwargs[key]
    elif callable(key):
        return key
    raise ValueError(f"`key` must be a string or callable but got {key}.")


def accumulate_cumulative_average(
    key: Union[str, Callable] = "updates",
    period: Optional[int] = None,
) -> Callable:
    """
    Accumulate the cumulative average.

    Args:
        key: Quantity to accumulate. If a string, accumulate a keyword argument. If a
            callable with the same signature as
            :meth:`~optax.GradientTransformationExtraArgs.update` (although only
            accepting keyword arguments), accumulate the returned value.
        period: Period after which the cumulative average resets.

    Returns:
        Accumulator function.
    """
    key_func = _make_key_func(key)

    def _accumulate(
        count: int,
        value: Any,
        updates: optax.Updates,
        state: optax.OptState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> Any:
        step = count % period if period else count
        return jax.tree.map(
            lambda old, new: (step * old + new) / (step + 1),
            value,
            key_func(updates=updates, state=state, params=params, **extra_args),
        )

    return _accumulate


def accumulate_most_recent(key: Union[str, Callable] = "updates") -> Callable:
    """
    Accumulate the most recent value.

    Args:
        key: Quantity to accumulate. If a string, accumulate a keyword argument. If a
            callable with the same signature as
            :meth:`~optax.GradientTransformationExtraArgs.update` (although only
            accepting keyword arguments), accumulate the returned value.

    Returns:
        Accumulator function.
    """
    key_func = _make_key_func(key)

    def _accumulate(
        count: int,
        value: Any,
        updates: optax.Updates,
        state: optax.OptState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> Any:
        return key_func(updates=updates, state=state, params=params, **extra_args)

    return _accumulate


class AccumulateOnUpdateState(NamedTuple):
    count: int
    value: Any


def accumulate_on_update(
    func: Callable, init: Any = None, *, skip_if_traced: bool = True
) -> optax.GradientTransformationExtraArgs:
    """
    Accumulate updates, parameters, or extra arguments without channging updates.

    Args:
        func: Accumulator function, receiving the iteration number as its first
            argument, previous accumulated value as its second argument, and
            :code:`updates`, :code:`state`, :code:`params`, and unpacked
            :code:`**extra_args` as keyword arguments.
        init: Initial state of the accumulator, defaults to the argument passed to
            :code:`init`.
        skip_if_traced: Skip accumulation if any of the arguments passed to
            :code`update` are traced.
    """

    # We don't use `optinspect.util.on_update` because we need to keep track of the
    # accumulator state.
    def init_func(params: optax.Params) -> AccumulateOnUpdateState:
        return AccumulateOnUpdateState(0, params if init is None else init)

    def update_func(
        updates: optax.Updates,
        state: AccumulateOnUpdateState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> tuple[optax.Updates, AccumulateOnUpdateState]:
        skip = skip_if_traced and is_traced(state)
        if not skip:
            value = func(
                state.count,
                state.value,
                updates=updates,
                state=state,
                params=params,
                **extra_args,
            )
            state = AccumulateOnUpdateState(state.count + 1, value)
        return updates, state

    return optax.GradientTransformationExtraArgs(init_func, update_func)
