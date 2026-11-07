"""Address arithmetic oracle; does not simulate Metal memory access."""
U32 = 1 << 32

def intended(count, raw_slot, action, delay):
    if not (count > 0 and raw_slot >= 0 and 0 <= action < count and 0 <= delay <= raw_slot):
        raise ValueError("invalid source coordinate")
    return (raw_slot - delay)*count + action

def metal_index(count, action, delay):
    # Both operation.source.{y,z} and program.counts0.x are uint32.
    return (action - delay * count) % U32

def observed_relative(count, raw_slot, action, delay):
    # Subscript is applied to pointer at raw_slot*count. Adding the wrapped
    # uint offset is pointer arithmetic, not a second uint32 wrap.
    return raw_slot*count + metal_index(count, action, delay)

for count in (1, 12, 29):
    for raw_slot in (1, 2, 5):
        for delay in range(raw_slot+1):
            for action in {0,count-1}:
                expected=intended(count,raw_slot,action,delay)
                actual=observed_relative(count,raw_slot,action,delay)
                assert (actual==expected) == (delay==0)
                if delay:
                    assert actual-expected==U32
for bad in ((0,1,0,0),(29,1,-1,0),(29,1,29,0),(29,1,0,2)):
    try: intended(*bad)
    except ValueError: pass
    else: raise AssertionError(bad)
print("CPU oracle: zero delay resolves; positive-delay uint32 offsets differ by 2^32 elements; invalid/boundary cases rejected")
print("G1 count=29, raw slot=1, ankle action=5, delay=1: intended relative element", intended(29,1,5,1), ", unsigned subscript", metal_index(29,5,1), ", effective relative element", observed_relative(29,1,5,1))
