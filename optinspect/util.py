import functools
import inspect
import jax
from jax.core import Tracer
from jax import numpy as jnp
import optax
from typing import Any, Callable, Hashable, Iterable, Optional, Sequence, Union


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


@functools.singledispatch
def _get(obj: Any, key: Any) -> Any:
    """
    Get the value at :code:`key` from :code:`obj`. The default implementation handles
    :class:`jax.tree_util.GetAttrKey` and otherwise tries to index by the key. Other
    types can be handled by registering a new implementation using
    :code:`_get.register`.
    """
    if isinstance(key, jax.tree_util.GetAttrKey):
        return getattr(obj, key.name)
    try:
        return obj[key]
    except Exception as ex:  # pragma: no cover
        raise KeyError(f"Cannot index object `{obj}` by key `{key}`.") from ex


@functools.singledispatch
def _set(obj: Any, key: Any, value: Any) -> Any:  # pragma: no cover
    raise NotImplementedError(f"`_set` is not implemented for object `{obj}`.")


@_set.register
def _set_dict(obj: dict, key: Hashable, value: Any) -> dict:
    obj = obj.copy()
    obj[key] = value
    return obj


@_set.register(list)
@_set.register(tuple)
def _set_sequence(
    obj: Union[list, tuple], key: Union[int, str, jax.tree_util.GetAttrKey], value: Any
) -> Sequence:
    # Handle named tuples separately.
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        if isinstance(key, jax.tree_util.GetAttrKey):
            key = key.name
        elif isinstance(key, int):
            key = obj._fields[key]
        else:  # pragma: no cover
            raise KeyError(f"Key `{key}` is not a valid key for `{obj}`.")
        return obj._replace(**{key: value})  # type: ignore[attr-defined]

    # Handle general tuples and lists.
    assert isinstance(key, int), f"Sequence indices must be integers, got `{key}`."
    cls = obj.__class__
    obj = list(obj)
    obj[key] = value
    return obj if cls is list else cls(obj)


@_set.register
def _set_ndarray(obj: jnp.ndarray, key: Any, value: Any) -> jnp.ndarray:
    return obj.at[key].set(value)


def tree_get(tree: Any, path: Union[Iterable, Any]) -> Any:
    """
    Get the value at :code:`path` in :code:`tree`. This function does not flatten the
    tree and can act on non-leaf nodes.

    Args:
        tree: Tree to get a value from.
        path: Path in the tree. If not iterable, the path is assumed to be a key and is
            replaced by :code:`(path,)`.

    Returns:
        Value at :code:`path` in :code:`tree`.

    Example:
        >>> from jax import numpy as jnp
        >>> import optinspect
        >>>
        >>> tree = {"a": [3, 4, jnp.arange(2, 5)], "b": "hello world"}
        >>> tree
        {'a': [3, 4, Array([2, 3, 4], ...)], 'b': 'hello world'}
        >>> optinspect.tree_get(tree, ["a", -1, slice(2)])
        Array([2, 3], ...)
    """
    if not isinstance(path, Iterable) or isinstance(path, str):
        path = (path,)
    key, *path = path
    obj = _get(tree, key)
    if path:
        return tree_get(obj, path)
    return obj


def tree_set(tree: Any, path: Union[Iterable, Any], value: Any) -> Any:
    """
    Set the :code:`value` at :code:`path` in :code:`tree`. This function does not
    flatten the tree and can act on non-leaf nodes.

    Args:
        tree: Tree to set a value in.
        path: Path in the tree. If not iterable, the path is assumed to be a key and is
            replaced by :code:`(path,)`.
        value: Value to set.

    Returns:
        Tree updated out of place.

    Example:
        >>> from jax import numpy as jnp
        >>> import optinspect
        >>>
        >>> tree = {"a": [3, 4, jnp.arange(2, 5)], "b": "hello world"}
        >>> tree
        {'a': [3, 4, Array([2, 3, 4], ...)], 'b': 'hello world'}
        >>> optinspect.tree_set(tree, ["a", -1, 1], 99)
        {'a': [3, 4, Array([ 2, 99,  4], dtype=int32)], 'b': 'hello world'}
    """
    if not isinstance(path, Iterable) or isinstance(path, str):
        path = (path,)
    key, *path = path
    if path:
        value = tree_set(_get(tree, key), path, value)
    return _set(tree, key, value)


def with_tree_get(func: Callable, path: Union[Iterable, Any]) -> Callable:
    """
    Create a function :code:`transformed(*args, **kwargs)` that extracts the subtree of
    :code:`func(*args, **kwargs)` at :code:`path`.

    Args:
        func: Function to transform.
        path: Path in the returned tree at which to extract values. If not iterable, the
            path is assumed to be a key and is replaced by :code:`(path,)`.

    Returns:
        Transformed function.

    Example:
        >>> import optinspect
        >>>
        >>> def func(x):
        ...     return {"a": [3, 4], "b": [19 * x, 2]}
        >>>
        >>> optinspect.with_tree_get(func, ["b", 0])(2)
        38
    """

    @functools.wraps(func)
    def _inner(*args, **kwargs):
        tree = func(*args, **kwargs)
        return tree_get(tree, path)

    return _inner


def with_tree_set(func: Callable, x: Any, path: Union[Iterable, Any]) -> Callable:
    """
    Create a function :code:`transformed(y, *args, **kwargs)` that evaluates
    :code:`func(x, *args, **kwargs)`, where the value at :code:`path` in :code:`x` is
    replaced by :code:`y`.

    This transformation is useful for evaluating gradients, Jacobians, and Hessians with
    respect to the subtree of :code:`x` at :code:`path` evaluated at :code:`y`.

    Args:
        func: Function to transform.
        x: Tree in which to replace the value at :code:`path`.
        path: Path in :code:`x` at which to replace values. If not iterable, the path is
            assumed to be a key and is replaced by :code:`(path,)`.

    Returns:
        Transformed function.

    Example:
        >>> import jax
        >>> from jax import numpy as jnp
        >>> import optinspect
        >>>
        >>> def func(params, x):
        ...     return (x - params["loc"]) ** 2 / (2 * params["scale"] ** 2)
        >>>
        >>> params = {"loc": 1.0, "scale": 2.0}
        >>> x = 3.0
        >>> func(params, x)
        0.5
        >>> jax.grad(func)(params, x)
        {'loc': Array(-0.5, ...), 'scale': Array(-0.5, ...)}
        >>>
        >>> func_at_loc = optinspect.with_tree_set(func, params, "loc")
        >>> func_at_loc(1.0, x)
        0.5
        >>> func_at_loc(3.0, x)
        0.0
        >>> jax.grad(func_at_loc)(1.0, x)
        Array(-0.5, ...)
        >>> jax.grad(func_at_loc)(3.0, x)
        Array(-0., ...)
    """

    @functools.wraps(func)
    def _inner(y, *args, **kwargs) -> Any:
        replaced_x = tree_set(x, path, y)
        # breakpoint()
        return func(replaced_x, *args, **kwargs)

    return _inner
