from pathlib import Path
import re

root = Path(__file__).resolve().parent
baseline = root.parent / "tracker-physical-boundary-20261108"

def check_history(name, stride, env_bytes):
    log = (root / f"{name}.log").read_text()
    codes = [(int(i), int(x)) for i,x in re.findall(r"contact\[(\d+)\] code=(\d+)",log)]
    assert [i for i,_ in codes] == list(range(60))
    codes = [x for _,x in codes]
    assert [(t,e) for t in range(30) for e in range(2) if codes[2*t+e]] == [(13,0),(14,0),(26,0),(27,0),(27,1)]
    data = (root / f"{name}.history").read_bytes()
    assert len(data)==30*3*stride
    def get(t,p,e):
        start=(3*t+p)*stride+e*env_bytes
        return data[start:start+env_bytes]
    for t in range(30):
        for e in range(2):
            before,observed,after=[get(t,p,e) for p in range(3)]
            assert after == (before if codes[2*t+e] else observed), (name,t,e)
    for t in (13,26):
        assert codes[2*(t-1):2*(t-1)+2] == [0,0] and codes[2*t:2*t+2] == [5,0]
        for e in range(2):
            assert get(t,0,e) == get(t-1,2,e)
            assert get(t,0,e) != get(t,1,e)
    print(f"{name}: 5 failed histories restored, 55 accepted retained")

check_history("final_candidate",2048,96)
check_history("two_bindings",4096,192)
for ext in ("q","v","scene","actor","status"):
    old=(baseline/f"trial.{ext}").read_bytes()
    assert all((root/f"{arm}.{ext}").read_bytes()==old for arm in ("final_candidate","production","integration")), ext
    assert (root/f"hook_control.{ext}").read_bytes()==(root/f"hook_retry.{ext}").read_bytes(), ext
assert (root/"legacy-control.bin").read_bytes()==(root/"legacy-retry.bin").read_bytes()
print("PASS: physical arrays match baseline on main, instrumented and integrated builds")
print("PASS: callback abort retry and legacy callback retry match their controls")
