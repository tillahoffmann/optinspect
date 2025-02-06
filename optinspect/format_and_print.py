import functools
import optax
from typing import Any, Optional
from .util import before_after_update, on_update


def _format_and_print(format: str, **kwargs: Any) -> None:
    return print(format.format(**kwargs))


def print_on_update(
    format: str, *, skip_if_traced: bool = True
) -> optax.GradientTransformationExtraArgs:
    """
    Print updates, parameters, or extra arguments without changing updates.

    Args:
        format: Format string receiving keyword arguments :code:`updates`,
            :code:`params`, and unpacked :code:`**extra_args`.
        skip_if_traced: Skip printing if any of the arguments passed to :code`update`
            are traced.

    Returns:
        Gradient transformation that prints and leaves updates unchanged.
    """
    return on_update(
        functools.partial(_format_and_print, format), skip_if_traced=skip_if_traced
    )


def print_before_after_update(
    inner: optax.GradientTransformation,
    before_format: Optional[str] = None,
    after_format: Optional[str] = None,
    *,
    skip_if_traced: bool = True,
) -> optax.GradientTransformationExtraArgs:
    """
    Print state information before and/or after updates.

    Args:
        inner: Transformation whose state to monitor.
        before_format: Format string to use before updates, receiving keyword arguments
            :code:`state`, :code:`updates`, :code:`params`, and unpacked
            :code:`**extra_args`.
        after_format: Format string to use after updates, receiving keyword arguments
            :code:`state`, :code:`updates`, :code:`params`, and unpacked
            :code:`**extra_args`.
        skip_if_traced: Skip printing if the state is traced.

    Returns:
        Gradient transform that prints state information before and/or after updates and
        leaves updates unchanged.
    """
    return before_after_update(
        inner,
        functools.partial(_format_and_print, before_format) if before_format else None,
        functools.partial(_format_and_print, after_format) if after_format else None,
        skip_if_traced=skip_if_traced,
    )
