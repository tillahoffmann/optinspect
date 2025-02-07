import jax
import optax
from typing import Any, Callable, NamedTuple, Optional, Union
from .util import inspect_update, make_key_func


def accumulate_cumulative_average(
    key: Union[str, Callable] = "updates",
    period: Optional[int] = None,
) -> Callable:
    """
    Accumulate the cumulative average.

    Args:
        key: Quantity to accumulate. If a callable with the same signature as
            :meth:`~optax.GradientTransformationExtraArgs.update`, accumulate the
            returned value. If a string, accumulate arguments by name. If an integer,
            accumulate arguments by their position.
        period: Period after which the cumulative average resets.

    Returns:
        Accumulator function.
    """
    key_func = make_key_func(key)

    def _accumulate(
        updates: optax.Updates,
        state: AccumulateState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> Any:
        step = state.count % period if period else state.count
        return jax.tree.map(
            lambda old, new: (step * old + new) / (step + 1),
            state.value,
            key_func(updates, state, params, **extra_args),
        )

    return _accumulate


def accumulate_most_recent(key: Union[str, Callable] = "updates") -> Callable:
    """
    Accumulate the most recent value.

    Args:
        key: Quantity to accumulate. If a callable with the same signature as
            :meth:`~optax.GradientTransformationExtraArgs.update`, accumulate the
            returned value. If a string, accumulate arguments by name. If an integer,
            accumulate arguments by their position.

    Returns:
        Accumulator function.
    """
    key_func = make_key_func(key)

    def _accumulate(
        updates: optax.Updates,
        state: AccumulateState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> optax.Params:
        return key_func(updates, state, params, **extra_args)

    return _accumulate


class AccumulateState(NamedTuple):
    """
    State for accumulating values.

    Attributes:
        count: Iteration number.
        value: Accumulated value.
    """

    count: int
    value: optax.Params


def accumulate_update(
    accumulate: optax.GradientTransformationExtraArgs,
    init: Optional[optax.TransformInitFn] = None,
    *,
    skip_if_traced: bool = None,
) -> optax.GradientTransformationExtraArgs:
    """
    Accumulate updates, parameters, or extra arguments.

    Args:
        accumulate: Accumulation function with the same signature as
            :meth:`~optax.GradientTransformationExtraArgs.update`, returning the updated
            accumulated value.
        init: Callable to initialize the :cls:`.AccumulateState` or :code:`None` to
            initialize with the parameters passed to the :code:`init` function of the
            transformation.
        skip_if_traced: Skip accumulation if the :code:`updates` argument is traced.

    Returns:
        Gradient transformation.

    .. seealso::

        :func:`.accumulate_cumulative_average`, :func:`.accumulate_most_recent`.
    """

    _init = init or (lambda params: AccumulateState(0, params))

    def _update(
        updates: optax.Updates,
        state: AccumulateState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> AccumulateState:
        return AccumulateState(
            state.count + 1, accumulate(updates, state, params, **extra_args)
        )

    return inspect_update(_update, _init, skip_if_traced=skip_if_traced)
