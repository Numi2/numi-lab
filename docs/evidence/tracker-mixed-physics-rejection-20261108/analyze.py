from pathlib import Path
import struct

raw = Path(__file__).with_name("two_verified.actor").read_bytes()
assert len(raw) == 2 * 2 * 3 * 34 * 4
values = struct.unpack(f"<{len(raw) // 4}f", raw)
for step in range(2):
    for environment in range(2):
        print(f"step={step} environment={environment}")
        for history in range(3):
            index = (step * 2 + environment) * 102 + history * 34
            print(f"  slot={history}: {values[index:index + 7]}")
