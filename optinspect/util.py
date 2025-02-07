import functools
import inspect
import jax
from jax.core import Tracer
import optax
import os
from typing import Any, Callable, NamedTuple, Optional, Union


def make_key_func(key: Union[str, int, Callable]) -> Callable:
    """
    Create a function to extract a value from the arguments of a
    :cls:`~optax.TransformUpdateExtraArgsFn` function.

    Args:
        key: :cls:`str` to retrieve an argument by name, :cls:`int` to retrieve a
            positional argument, or a callable which is returned unchanged.

    Returns:
        to extract a value from the arguments of a
        :cls:`~optax.TransformUpdateExtraArgsFn` function.
    """
    if isinstance(key, Callable):
        return key
    elif isinstance(key, int):
        return lambda *args, **_: args[key]
    elif isinstance(key, str):

        def _key_func(
            updates: optax.Updates,
            state: optax.OptState,
            params: Optional[optax.Params] = None,
            **extra_args: Any,
        ) -> Any:
            extra_args.update(
                {
                    "updates": updates,
                    "state": state,
                    "params": params,
                }
            )
            return extra_args[key]

        return _key_func

    raise ValueError(f"`key` must be a string, integer, or callable, but got `{key}`.")


def maybe_skip_update_if_traced(
    update: optax.TransformUpdateExtraArgsFn = None, *, skip_if_traced: bool
) -> optax.TransformUpdateExtraArgsFn:
    """
    Skip an update function if the first :code:`updates` argument is traced depending on
    :code:`skip_if_traced`.

    Args:
        update: Update function to wrap.
        skip_if_traced: Indicates whether to skip the update if traced. If :code:`None`,
            the update is skipped unless the environment variable
            :code:`INSPECT_IF_TRACED` is set.

    Returns:
        Update function that is skipped if the first :code:`updates` argument is traced.
    """
    if update is None:
        return functools.partial(
            maybe_skip_update_if_traced, skip_if_traced=skip_if_traced
        )

    @functools.wraps(update)
    def _wrapped_update(
        updates: optax.Updates,
        state: optax.EmptyState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> tuple[optax.Updates, optax.OptState]:
        nonlocal skip_if_traced
        if skip_if_traced is None:
            skip_if_traced = "INSPECT_IF_TRACED" not in os.environ
        if skip_if_traced and is_traced(updates):
            return updates, state
        else:
            return update(updates, state, params, **extra_args)

    return _wrapped_update


def is_traced(*args: Any) -> bool:
    """
    Check if any of the arguments are traced.

    Args:
        *args: Sequence of arguments.

    Returns:
        :code:`True` if any of the arguments are traced, otherwise :code:`False`.
    """
    leaves, _ = jax.tree.flatten(args)
    return any(isinstance(leaf, Tracer) for leaf in leaves)


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
            :cls:`~optax.EmptyState`.
        init: Function with the same signature as
            :meth:`~optax.GradientTransformationExtraArgs.init` or :code:`None` to
            initialize with an :cls:`~optax.EmptyState`.
        skip_if_traced: Skip the :code:`update` function if the :code:`updates`
            argument is traced.

    Returns:
        Gradient transformation.
    """

    _init = init or (lambda _: optax.EmptyState())

    @maybe_skip_update_if_traced(skip_if_traced=skip_if_traced)
    def _update(
        updates: optax.Updates,
        state: optax.EmptyState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> tuple[optax.Updates, optax.OptState]:
        state = update(updates, state, params, **extra_args)
        if state is None:
            state = optax.EmptyState()
        return updates, state

    return optax.GradientTransformationExtraArgs(_init, _update)


class WrappedState(NamedTuple):
    """
    State wrapping another gradient transformation.

    Attributes:
        inner: State of the wrapped optimizer.
        outer: Additional state information.
    """

    inner: optax.OptState
    outer: optax.OptState


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
            :cls:`.WrappedState` after the wrapped transformation has been applied. It
            must return the updated :code:`outer` state. If no value is returned, it is
            replaced by an :cls:`~optax.EmptyState`.
        init: Function with the same signature as
            :meth:`~optax.GradientTransformationExtraArgs.init` or :code:`None` to
            initialize with a :cls:`.WrappedState` whose :code:`outer` state is
            :cls:`~optax.EmptyState`.
        skip_if_traced: Skip the :code:`update` function if the :code:`updates`
            argument is traced.

    Returns:
        Gradient transformation.
    """

    _init = init or (
        lambda params: WrappedState(inner.init(params), optax.EmptyState())
    )

    @maybe_skip_update_if_traced(skip_if_traced=skip_if_traced)
    def _update(
        updates: optax.Updates,
        state: WrappedState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> tuple[optax.Updates, optax.OptState]:
        updates, inner_state = inner.update(updates, state.inner, params, **extra_args)
        state = WrappedState(inner_state, state.outer)
        outer_state = update(updates, state, params, **extra_args)
        if outer_state is None:
            outer_state = optax.EmptyState()
        return updates, WrappedState(inner_state, outer_state)

    return optax.GradientTransformationExtraArgs(_init, _update)


def frepr(func: Callable) -> str:
    """
    Represent a function, including signature, closures, and declaration.
    """
    signature = inspect.signature(func)
    code = func.__code__
    n_freevars = len(code.co_freevars)
    if inspect.isfunction(func):
        kind = "function"
    else:
        ValueError(func)  # pragma: no cover
    parts = [
        f"{kind} {func.__module__}.{func.__qualname__}{signature}",
        f"file '{code.co_filename}'",
        f"line {code.co_firstlineno}",
        f"{n_freevars} free {'var' if n_freevars == 1 else 'vars'}",
    ]
    return f"<{', '.join(parts)}>"
