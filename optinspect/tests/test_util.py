import jax
from jax import numpy as jnp
import optax
import optinspect
import os
import pytest
import re
from typing import Any, NamedTuple, Optional
from unittest import mock


def test_is_traced() -> None:
    x = {"a": 3.0, "b": (2.0, 1.0)}
    assert not optinspect.is_traced(x)
    assert jax.jit(optinspect.is_traced)(x)


def test_wrapped(value_and_grad_and_params, skip_if_traced: bool) -> None:
    value_and_grad, params = value_and_grad_and_params
    _, grad = value_and_grad(params)

    update: list[optax.ScaleByAdamState] = []
    optim = optinspect.inspect_wrapped(
        optax.scale_by_adam(0.1),
        lambda updates, state, *args, **kwargs: update.append(state.inner),
        skip_if_traced=skip_if_traced,
    )
    state = optim.init(params)
    optim.update(grad, state, params)
    assert update[0].count == 1

    if skip_if_traced:
        jax.jit(optim.update)(grad, state)
        assert len(update) == 1
    else:
        # This fails because the `append` function has side effects.
        with pytest.raises(Exception, match="Leaked trace"), jax.check_tracer_leaks():
            jax.jit(optim.update)(grad, state)


@pytest.mark.parametrize(
    "jit, skip_if_traced, env, should_skip",
    [
        (False, False, False, False),
        (False, True, False, False),
        (False, None, False, False),
        (False, False, True, False),
        (False, True, True, False),
        (False, None, True, False),
        (True, False, False, False),
        (True, True, False, True),
        (True, None, False, True),
        (True, False, True, False),
        (True, True, True, True),
        (True, None, True, False),
    ],
)
def test_get_skip(
    jit: bool, skip_if_traced: Optional[bool], env: bool, should_skip: bool
) -> None:
    assert (
        "INSPECT_IF_TRACED" not in os.environ
    ), "Environment variable `INSPECT_IF_TRACED` must not be set for tests."

    x = 3

    def update(updates, state, params=None, **extra_args):
        if optinspect.inspect._get_skip(updates, skip_if_traced):
            return updates, state
        else:
            return -updates, -state

    if jit:
        update = jax.jit(update)

    if env:
        with mock.patch.dict(os.environ, {"INSPECT_IF_TRACED": "true"}):
            result = update(x, x)
    else:
        result = update(x, x)

    if should_skip:
        assert result == (x, x)
    else:
        assert result == (-x, -x)


def test_frepr() -> None:
    a = 3
    actual = optinspect.func_repr(lambda x: a + x, freevars=False)
    assert re.match(
        r"<function optinspect\.tests\.test_util\.test_frepr\.<locals>\.<lambda>\(x\), "
        r"file '.*?optinspect/tests/test_util\.py', line \d+, 1 free var>",
        actual,
    )
    actual = optinspect.func_repr(lambda x: a + x)
    assert re.match(
        r"<function optinspect\.tests\.test_util\.test_frepr\.<locals>\.<lambda>\(x\), "
        r"file '.*?optinspect/tests/test_util\.py', line \d+, free var: {'a': 3}>",
        actual,
    )


def test_invalid_key_func() -> None:
    with pytest.raises(ValueError, match="must be a string, integer, or callable"):
        optinspect.util.make_key_func(1.3)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "key, expected",
    [
        ("state", "arg2"),
        (2, "arg3"),
        (lambda *, extra: 2 * extra, "kwargkwarg"),
        (lambda *, state: state.split("g"), ["ar", "2"]),
        (lambda **kwargs: kwargs["updates"] + kwargs["params"], "arg1arg3"),
        (lambda *, params, **kwargs: kwargs["updates"] + params, "arg1arg3"),
    ],
)
def test_make_key_func(key, expected) -> None:
    args = ("arg1", "arg2", "arg3")
    kwargs = {"extra": "kwarg"}
    key_func = optinspect.util.make_key_func(key)
    actual = key_func(*args, **kwargs)
    assert actual == expected


def test_tree_get_set() -> None:
    class TestTuple(NamedTuple):
        c: Any
        d: Any

    tree: Any = {"a": TestTuple(3, jnp.arange(2, 5)), "b": "hello world"}
    key = ["a", jax.tree_util.GetAttrKey("d"), slice(2)]
    assert jnp.allclose(optinspect.tree_get(tree, key), 2 + jnp.arange(2))

    new = optinspect.tree_set(tree, key, 17)
    assert jnp.allclose(new["a"].d, jnp.array([17, 17, 4]))

    # Try named tuple access by index, and getattrkey.
    tree = TestTuple(3, 4)
    assert optinspect.tree_get(tree, 1) == 4
    assert optinspect.tree_get(tree, [jax.tree_util.GetAttrKey("d")]) == 4

    assert optinspect.tree_set(tree, 1, 9) == TestTuple(3, 9)
    assert optinspect.tree_set(tree, [jax.tree_util.GetAttrKey("d")], 7) == TestTuple(
        3, 7
    )
