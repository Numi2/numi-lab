from pathlib import Path
import re

root = Path(__file__).resolve().parent
steps, environments, phases, snapshot_bytes, env_bytes = 30, 2, 3, 2048, 96

def read(name, ext):
    return (root / f"{name}.{ext}").read_bytes()

def codes(name):
    log = (root / f"{name}.log").read_text()
    found = [(int(i), int(c)) for i, c in re.findall(r"contact\[(\d+)\] code=(\d+)", log)]
    assert [i for i, _ in found] == list(range(steps * environments))
    return [code for _, code in found]

def history(data, step, phase, env):
    start = (step * phases + phase) * snapshot_bytes + env * env_bytes
    return data[start:start + env_bytes]

original_codes = codes("trial")
assert codes("control") == original_codes == codes("repair") == codes("repair_repeat")
assert [(t, e) for t in range(steps) for e in range(environments)
        if original_codes[t * environments + e]] == [(13, 0), (14, 0), (26, 0), (27, 0), (27, 1)]
for ext in ("q", "v", "scene", "actor", "status"):
    assert read("trial", ext) == read("control", ext), f"capture changed {ext}"
    assert read("trial", ext) == read("repair", ext), f"prototype changed {ext}"
    assert read("repair", ext) == read("repair_repeat", ext), f"repeat changed {ext}"
assert read("repair", "history") == read("repair_repeat", "history")
original, repaired = read("trial", "history"), read("repair", "history")
assert len(original) == len(repaired) == steps * phases * snapshot_bytes
for t in range(steps):
    for e in range(environments):
        before, observed, after = [history(repaired, t, x, e) for x in range(phases)]
        failed = original_codes[t * environments + e] != 0
        assert after == (before if failed else observed), (t, e, failed)
for t in (13, 26):
    assert original_codes[2*(t-1):2*(t-1)+2] == [0, 0]
    assert original_codes[2*t:2*t+2] == [5, 0]
    for e in range(environments):
        before, observed, after = [history(original, t, x, e) for x in range(phases)]
        assert before == history(original, t-1, 2, e)
        assert before != observed and observed == after
    assert history(repaired,t,0,0) != history(repaired,t,1,0)
    assert history(repaired,t,0,1) != history(repaired,t,1,1)
print("PASS: control equivalence; 30-step statuses; mixed failures at 13 and 26")
print("PASS: prototype restored 5/5 failed histories and retained 55/55 successful histories")
print("PASS: repair repeat is byte-identical; scope is object-track history only")
