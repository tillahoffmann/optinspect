import inspect
import jax
from jax.core import Tracer
import optax
from typing import Any, Callable, Optional, Union


def make_key_func(key: Union[str, int, Callable]) -> Callable:
    """
    Create a function to extract a value from the arguments of a
    :class:`~optax.TransformUpdateExtraArgsFn` function.

    Args:
        key: :class:`str` to retrieve an argument by name, :class:`int` to retrieve a
            positional argument, or a callable which is returned unchanged.

    Returns:
        Function to extract a value from the arguments of a
        :class:`~optax.TransformUpdateExtraArgsFn` function.
    """
    if callable(key):
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


def is_traced(*args: Any) -> bool:
    """
    Check if any of the arguments are traced.

    Args:
        *args: Sequence of arguments.

    Returns:
        :code:`True` if any of the arguments are traced, otherwise :code:`False`.

    Example:
        >>> import jax
        >>> import optinspect
        >>>
        >>> optinspect.is_traced({"a": 3})
        False
        >>> jax.jit(optinspect.is_traced)({"a": 3})
        Array(True, ...)
    """
    leaves, _ = jax.tree.flatten(args)
    return any(isinstance(leaf, Tracer) for leaf in leaves)


def func_repr(func: Callable, freevars: bool = True) -> str:
    """
    Represent a function, including signature, closures, and declaration.

    Args:
        func: Function to represent.
        freevars: Include values of free variables from the closure.

    Returns:
        Verbose representation of :code:`func`.

    Example:
        >>> import optax
        >>> import optinspect
        >>>
        >>> optinspect.func_repr(optax.scale_by_adam(0.02).update)
        "<function ...update_fn(updates, state, params=None),
          file '.../transform.py', line ...,
          free vars: {'b1': 0.02, 'b2': 0.999, 'eps': 1e-08, ...}>"
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
    ]
    if freevars and func.__closure__:
        values = {
            key: cell.cell_contents
            for key, cell in zip(code.co_freevars, func.__closure__)
        }
        parts.append(f"free {'var' if n_freevars == 1 else 'vars'}: {values}")
    else:
        parts.append(f"{n_freevars} free {'var' if n_freevars == 1 else 'vars'}")
    return f"<{', '.join(parts)}>"
