"""
The :mod:`~optinspect.format_and_print` module implements gradient transformations
:func:`.print_update` to print updates and :func:`.print_wrapped` to print the state of
a wrapped gradient transformation for quick debugging.
"""

import functools
import optax
from typing import Any
from .inspect import inspect_wrapped, inspect_update


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

    Example:
        >>> import jax
        >>> from jax import numpy as jnp
        >>> import optinspect
        >>>
        >>> optim = optinspect.print_update("params: {params}")
        >>> params = 3.0
        >>> value_and_grad = jax.value_and_grad(jnp.square)
        >>> state = optim.init(params)
        >>> value, grad = value_and_grad(params)
        >>> updates, state = optim.update(grad, state, params, value=value)
        params: 3.0
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

    Example:
        >>> import jax
        >>> from jax import numpy as jnp
        >>> import optax
        >>> import optinspect
        >>>
        >>> optim = optinspect.print_wrapped(
        ...     optax.adam(0.1), "second moment: {state.inner[0].nu:.3f}"
        ... )
        >>> params = 3.0
        >>> value_and_grad = jax.value_and_grad(jnp.square)
        >>> state = optim.init(params)
        >>> value, grad = value_and_grad(params)
        >>> updates, state = optim.update(grad, state, params, value=value)
        second moment: 0.036
    """
    return inspect_wrapped(
        inner,
        functools.partial(_format_and_print, format),
        skip_if_traced=skip_if_traced,
    )
