from pathlib import Path
import struct
p = Path(__file__).resolve().parent
assert (p / "local0.bin").read_bytes() == (p / "local1.bin").read_bytes()
for suffix in ("q", "v", "scene", "actor", "depth", "validity", "metadata"):
    a = (p / f"world-single.{suffix}").read_bytes()
    b = (p / f"world-split.{suffix}").read_bytes()
    assert a and a == b, suffix
single = (p / "world-single.status").read_bytes()
split = (p / "world-split.status").read_bytes()
assert len(single) == len(split) == 4 * 48
for i in range(4):
    a = list(struct.unpack_from("<12I", single, i * 48))
    b = list(struct.unpack_from("<12I", split, i * 48))
    assert a[0] == b[0] == 0 and a[2] == i // 2 and b[2] == 0
    a[2] = b[2] = 0
    assert a == b, (i, a, b)
assert "tracker frame identity or time reset at submission boundary" in (p / "baseline-ctest.log").read_text()
assert "100% tests passed out of 22" in (p / "final-ctest2.log").read_text()
print("verified local-slot and world chunk comparisons; statuses differ only in local controlStep")
