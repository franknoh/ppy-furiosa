"""Mapping syntax must round-trip as compiler data and reject source injection."""

import pytest
from ppy_compiler.ir import parse_type

from ppy_furiosa.mapping import Const, Mapping, Symbol


def test_mapping_matches_baseline_and_roundtrips() -> None:
    hidden = Symbol("H")
    mapping = Mapping((hidden // 16, hidden % 16))
    assert mapping.rust() == "m![H / 16, H % 16]"
    assert Mapping.from_ir(parse_type(str(mapping.to_ir()))) == mapping
    assert hash(mapping) == hash(Mapping((hidden // 16, hidden % 16)))


def test_padding_and_extent_are_distinct() -> None:
    rows = (Symbol("L") % 60).with_extent(4).pad(60)
    assert Mapping((rows, Symbol("H") // 16)).rust() == "m![L % 60 = 4 # 60, H / 16]"
    assert Mapping((Const(1).pad(256),)).rust() == "m![1 # 256]"


@pytest.mark.parametrize("name", ["H]; panic!()", "", "3x", "type", "a::b"])
def test_rejects_invalid_symbols(name: str) -> None:
    with pytest.raises(ValueError, match="symbol"):
        Symbol(name)


def test_rejects_zero_divisor_and_truncating_padding() -> None:
    with pytest.raises(ValueError, match="positive"):
        _ = Symbol("H") // 0
    with pytest.raises(ValueError, match="padding"):
        Const(4).pad(2)
