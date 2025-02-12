"""
The :mod:`~optinspect.accumulate` module implements gradient transformations
:func:`.accumulate_update` to accumulate updates and :func:`.accumulate_wrapped` to
accumulate the state of a wrapped gradient transformations.
"""

import jax
import optax
from typing import Any, Callable, NamedTuple, Optional, Union
from .inspect import inspect_update, inspect_wrapped, WrappedState
from .tag import _update_tagged_state, get_tagged_values
from .util import make_key_func


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
        state: Union[AccumulateState, WrappedState],
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> Any:
        if isinstance(state, WrappedState):
            _, outer_state = state
        else:
            outer_state = state
        step = outer_state.count % period if period else outer_state.count
        return jax.tree.map(
            lambda old, new: (step * old + new) / (step + 1),
            outer_state.value,
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


@_update_tagged_state
class AccumulateState(NamedTuple):
    """
    State for accumulating values.
    """

    tag_: dict[str, None]
    """
    Unique tag of the traced value as a dictionary with a single key because strings
    are not valid jax types (cf. https://github.com/jax-ml/jax/issues/3045).
    """
    count: int
    """Iteration number."""
    value: optax.Params
    """Accumulated value."""


def accumulate_update(
    tag: str,
    accumulate: optax.GradientTransformationExtraArgs,
    init: Optional[optax.TransformInitFn] = None,
    *,
    skip_if_traced: bool = None,
) -> optax.GradientTransformationExtraArgs:
    """
    Accumulate updates, parameters, or extra arguments.

    Args:
        tag: Tag for the accumulated value.
        accumulate: Accumulation function with the same signature as
            :meth:`~optax.GradientTransformationExtraArgs.update`, returning the updated
            accumulated value.
        init: Callable to initialize the :class:`.AccumulateState` or :code:`None` to
            initialize with the parameters passed to the :code:`init` function of the
            transformation.
        skip_if_traced: Skip accumulation if the :code:`updates` argument is traced.

    Returns:
        Gradient transformation.

    Example:
        >>> import jax
        >>> from jax import numpy as jnp
        >>> import optinspect
        >>>
        >>> optim = optinspect.accumulate_update(
        ...     "accumulated_updates",
        ...     optinspect.accumulate_cumulative_average("updates")
        ... )
        >>> params = 3.0
        >>> value_and_grad = jax.value_and_grad(jnp.square)
        >>> state = optim.init(params)
        >>> value, grad = value_and_grad(params)
        >>> grad
        Array(6., ...)
        >>> updates, state = optim.update(grad, state, params, value=value)
        >>> state
        AccumulateState(tag='accumulated_updates', count=1, value=Array(6., ...))
        >>> params = 4.0
        >>> value, grad = value_and_grad(params)
        >>> grad
        Array(8., ...)
        >>> updates, state = optim.update(grad, state, params, value=value)
        >>> state
        AccumulateState(tag='accumulated_updates', count=2, value=Array(7., ...))

    .. seealso::

        :func:`.accumulate_cumulative_average`, :func:`.accumulate_most_recent`.
    """

    _init = init or (lambda params: AccumulateState({tag: None}, 0, params))

    def _update(
        updates: optax.Updates,
        state: AccumulateState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> AccumulateState:
        return AccumulateState(
            {tag: None},
            state.count + 1,
            accumulate(updates, state, params, **extra_args),
        )

    return inspect_update(_update, _init, skip_if_traced=skip_if_traced)


def reset_accumulate_count(state: optax.OptState) -> optax.OptState:
    """
    Reset the :code:`count` variable of all :class:`AccumulateState`\\ s.

    Args:
        state: State to update.

    Returns:
        State with the :code:`count` variable of :class:`AccumulateState`\\ s reset.

    Example:
        >>> import jax
        >>> from jax import numpy as jnp
        >>> import optinspect
        >>>
        >>> optim = optinspect.accumulate_update(
        ...     "accumulated_updates",
        ...     optinspect.accumulate_cumulative_average("updates")
        ... )
        >>> params = 3.0
        >>> value_and_grad = jax.value_and_grad(jnp.square)
        >>> state = optim.init(params)
        >>> value, grad = value_and_grad(params)
        >>> updates, state = optim.update(grad, state, params, value=value)
        >>> updates, state = optim.update(grad, state, params, value=value)
        >>> state
        AccumulateState(tag='accumulated_updates', count=2, value=Array(6., ...))
        >>> state = optinspect.reset_accumulate_count(state)
        >>> state
        AccumulateState(tag='accumulated_updates', count=0, value=Array(6., ...))
    """
    return optax.tree_utils.tree_set(
        state,
        lambda path, _: any(
            isinstance(part, optax.tree_utils.NamedTupleKey)
            and part.tuple_name == AccumulateState.__name__
            for part in path
        ),
        count=0,
    )


def get_accumulated_values(
    state: optax.OptState, tag: Optional[Any] = None
) -> dict[str, Any]:
    """
    Extract accumulated values from an optimizer state.

    Args:
        state: Optimizer state.
        tag: Tag of the state to extract. If specified, return only the requested
            accumulated value.

    Returns:
        Dictionary mapping tag names to accumulated values.
    """
    return get_tagged_values(state, tag=tag, tuple_name="AccumulateState")


def accumulate_wrapped(
    inner: optax.GradientTransformation,
    tag: str,
    accumulate: optax.GradientTransformationExtraArgs,
    init: Optional[optax.TransformInitFn] = None,
    *,
    skip_if_traced: bool = None,
) -> optax.GradientTransformationExtraArgs:
    """
    Accumulate the state of a wrapped gradient transformation after an update.

    Args:
        inner: Gradient transformation to wrap.
        tag: Tag for the accumulated value.
        accumulate: Accumulation function with the same signature as
            :meth:`~optax.GradientTransformationExtraArgs.update`, returning the updated
            accumulated value.
        init: Callable to initialize the :class:`.AccumulateState` or :code:`None` to
            initialize with the parameters passed to the :code:`init` function of the
            transformation.
        skip_if_traced: Skip accumulation if the :code:`updates` argument is traced.

    Returns:
        Gradient transformation.

    Example:
        >>> import jax
        >>> from jax import numpy as jnp
        >>> import optinspect
        >>>
        >>> optim = optinspect.accumulate_wrapped(
        ...     inner=optax.adam(1e-3),
        ...     tag="second_moment",
        ...     accumulate=optinspect.accumulate_cumulative_average(
        ...         lambda _, state, *args, **kwargs: state.inner[0].nu
        ...     )
        ... )
        >>> params = 3.0
        >>> value_and_grad = jax.value_and_grad(jnp.square)
        >>> state = optim.init(params)
        >>> value, grad = value_and_grad(params)
        >>> grad
        Array(6., ...)
        >>> updates, state = optim.update(grad, state, params, value=value)
        >>> state
        WrappedState(inner=(ScaleByAdamState(count=Array(1, ...),
                                             mu=Array(0.6, ...),
                                             nu=Array(0.036, ...)), EmptyState()),
                     outer=AccumulateState(tag='second_moment',
                                           count=1,
                                           value=Array(0.036, ...)))

    .. seealso::

        :func:`.accumulate_cumulative_average`, :func:`.accumulate_most_recent`.
    """

    def _init(params: optax.Params) -> WrappedState:
        inner_state = inner.init(params)
        outer_state = AccumulateState({tag: None}, 0, params)
        value = accumulate(None, WrappedState(inner_state, outer_state), params)
        outer_state = AccumulateState({tag: None}, 0, value)
        return WrappedState(inner_state, AccumulateState({tag: None}, 0, value))

    def _update(
        updates: optax.Updates,
        state: WrappedState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> AccumulateState:
        return AccumulateState(
            {tag: None},
            state.outer.count + 1,
            accumulate(updates, state, params, **extra_args),
        )

    return inspect_wrapped(inner, _update, _init, skip_if_traced=skip_if_traced)
