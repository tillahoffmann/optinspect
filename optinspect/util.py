import functools
import inspect
import jax
from jax.core import Tracer
import optax
import os
from typing import Any, Callable, Optional


def get_skip_if_traced(skip_if_traced: Optional[bool]) -> bool:
    """
    Determine if a transformation should be skipped if traced.

    .. warning::

        Whether a transformation is skipped if traced is determined when the transform
        is `created`, not on a case-by-case basis when it is `applied`.

    Args:
        skip_if_traced: Whether to skip the transformation if traced. If :code:`None`,
            skip unless the :code:`INSPECT_IF_TRACED` environment variable is set.

    Returns:
        :code:`skip_if_traced` if the input is not :code:`None`, otherwise whether the
        environment variable :code:`INSPECT_IF_TRACED` is set.
    """
    return (
        "INSPECT_IF_TRACED" in os.environ if skip_if_traced is None else skip_if_traced
    )


def maybe_skip_if_traced(func: Callable) -> Callable:
    """
    Wrap a function with keyword-only argument :code:`skip_if_traced` and normalize it
    based on its value and the environment variable :code:`INSPECT_IF_TRACED` begin set.
    """

    signature = inspect.signature(func)
    parameter = signature.parameters.get("skip_if_traced")
    if (
        parameter is None
        or parameter.kind is not parameter.KEYWORD_ONLY
        or parameter.default is not parameter.empty
    ):
        raise ValueError(
            f"Function `{func}` does not have a keyword-only argument `skip_if_traced` "
            "without default values."
        )

    @functools.wraps(func)
    def _wrapped(*args, skip_if_traced: Optional[bool] = None, **kwargs):
        skip_if_traced = get_skip_if_traced(skip_if_traced)
        return func(*args, skip_if_traced=skip_if_traced, **kwargs)

    return _wrapped


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


@maybe_skip_if_traced
def on_update(
    func: Callable, *, skip_if_traced: bool
) -> optax.GradientTransformationExtraArgs:
    """
    Call a function and leave the updates unchanged.

    Args:
        func: Function receiving keyword arguments :code:`updates`, :code:`params`, and
            unpacked :code:`**extra_args`.
        skip_if_traced: Skip printing if any of the arguments passed to :code`update`
            are traced.

    Returns:
        Gradient transformation that calls :code:`func` and leaves updates unchanged.
    """

    def init(params: optax.Params) -> optax.EmptyState:
        return optax.EmptyState()

    def update(
        updates: optax.Updates,
        state: optax.EmptyState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> tuple[optax.Updates, optax.OptState]:
        skip = skip_if_traced and is_traced(updates, params, extra_args)
        if not skip:
            func(updates=updates, params=params, **extra_args)
        return updates, state

    return optax.GradientTransformationExtraArgs(init, update)


@maybe_skip_if_traced
def before_after_update(
    inner: optax.GradientTransformationExtraArgs,
    before: Optional[Callable] = None,
    after: Optional[Callable] = None,
    *,
    skip_if_traced: bool,
) -> optax.GradientTransformationExtraArgs:
    """
    Call functions before and after applying a transformation.

    Args:
        inner: Gradient transformation to wrap.
        before: Function to call before applying :code:`inner`, receiving
            :code:`updates`, :code:`state`, :code:`params`, and unpacked
            :code:`**extra_args` as keyword arguments.
        after: Function to call after applying :code:`inner`, receiving
            :code:`updates`, :code:`state`, :code:`params`, and unpacked
            :code:`**extra_args` as keyword arguments.
        skip_if_traced: Skip execution of :code:`before` and :code:`after` if the state
            of :code:`inner` is traced.

    Returns:
        Gradient transformation equivalent to :code:`inner`.
    """
    if before is None and after is None:
        raise ValueError("At least one of `before` or `after` must be specified.")

    def init(params: optax.Params) -> optax.OptState:
        return inner.init(params)

    def update(
        updates: optax.Updates,
        state: optax.OptState,
        params: Optional[optax.Params] = None,
        **extra_args: Any,
    ) -> tuple[optax.Updates, optax.OptState]:
        skip = skip_if_traced and is_traced(state)
        if not skip and before:
            before(updates=updates, state=state, params=params, **extra_args)
        updates, state = inner.update(updates, state, params, **extra_args)
        if not skip and after:
            after(updates=updates, state=state, params=params, **extra_args)
        return updates, state

    return optax.GradientTransformationExtraArgs(init, update)
