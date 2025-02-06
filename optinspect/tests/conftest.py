import pytest


@pytest.fixture(params=[False, True], ids=["do_if_traced", "skip_if_traced"])
def skip_if_traced(request: pytest.FixtureRequest) -> bool:
    return request.param
