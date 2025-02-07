import functools
import optax
from typing import Any
from .util import inspect_wrapped, inspect_update


def _format_and_print(format: str, *args: Any, **kwargs: Any) -> None:
    # Augment the keyword arguments with the values from positional arguments.
    kwargs.update(dict(zip(["updates", "state", "params"], args)))
    return print(format.format(*args, **kwargs))


def print_update(
    format: str, *, skip_if_traced: bool = None
) -> optax.GradientTransformationExtraArgs:
    """
    Print updates, parameters, or extra arguments.

    Args:
        format: Format string, receiving the same positional and keyword arguments as
            :func:`~.optax.GradientTransformationExtraArgs`.
        skip_if_traced: Skip if the :code:`updates` argument is traced.

    Returns:
        Gradient transformation.
    """
    return inspect_update(
        functools.partial(_format_and_print, format), skip_if_traced=skip_if_traced
    )


def print_wrapped(
    inner: optax.GradientTransformation,
    format: str,
    *,
    skip_if_traced: bool = None,
) -> optax.GradientTransformationExtraArgs:
    """
    Print state information after an update.

    Args:
        inner: Transformation whose state to inspect.
        format: Format string, receiving the same positional and keyword arguments as
            :func:`~.optax.GradientTransformationExtraArgs`.
        skip_if_traced: Skip if the :code:`updates` argument is traced.

    Returns:
        Gradient transform.
    """
    return inspect_wrapped(
        inner,
        functools.partial(_format_and_print, format),
        skip_if_traced=skip_if_traced,
    )
