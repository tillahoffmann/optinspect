import functools
import inspect
import jax
from jax.core import Tracer
import optax
import os
from typing import Any, Callable, Optional, Union


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
