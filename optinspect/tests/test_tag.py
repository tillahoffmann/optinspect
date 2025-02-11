from optinspect.tag import _update_tagged_state
from typing import NamedTuple


@_update_tagged_state
class _TestState(NamedTuple):
    tag_: dict[str, None]
    a: int
    b: float


def test_tagged_state_repr() -> None:
    state = _TestState({"test_tag": None}, 3, 4.0)
    assert repr(state) == "_TestState(tag='test_tag', a=3, b=4.0)"
