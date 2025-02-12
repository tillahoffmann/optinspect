import optax
from typing import Any, NamedTuple, Optional, Type, Union


def _state_repr(self: NamedTuple) -> str:
    args = {"tag": self.tag}
    args.update({key: value for key, value in zip(self._fields, self) if key != "tag_"})
    args = ", ".join(f"{key}={repr(value)}" for key, value in args.items())
    return f"{self.__class__.__name__}({args})"


def _update_tagged_state(cls: Type[NamedTuple]) -> Type[NamedTuple]:
    """
    Update a "tagged" state with first element named :code:`tag_` which comprises a tag
    as :code:`{tag: None}`. The update gives reasonable :code:`repr` and adds a
    :code:`tag` property which unwraps the tag from the dictionary.
    """
    assert cls._fields[0] == "tag_", f"The first field of {cls} must be `tag_`."
    cls.__repr__ = _state_repr
    cls.tag = property(lambda self: next(iter(self.tag_)))
    return cls


def get_tagged_values(
    state: optax.OptState, *, tag: Optional[Any] = None, tuple_name: str
) -> Union[dict[str, Any], Any]:
    """
    Extract tagged values from an optimizer state.

    Args:
        state: Optimizer state.
        tag: Tag of the state to extract. If specified, return only the requested tagged
            value.
        tuple_name: Name of the tuple type to extract.

    Returns:
        Dictionary mapping tag names to values.
    """
    all_with_path = optax.tree_utils.tree_get_all_with_path(state, tuple_name)
    trace = {}
    for _, state in all_with_path:
        if tag is not None and tag == state.tag:
            return state.value
        if state.tag in trace:
            raise ValueError(f"Duplicate tag `{state.tag}`.")
        trace[state.tag] = state.value
    return trace
