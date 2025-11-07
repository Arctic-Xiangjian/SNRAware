from types import ModuleType

import snraware


def test_basics() -> None:
    assert type(snraware) is ModuleType
