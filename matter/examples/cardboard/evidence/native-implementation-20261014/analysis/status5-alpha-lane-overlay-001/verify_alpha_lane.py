#!/usr/bin/env python3
"""Source-bound contract test for the status-5 diagnostic alpha lane."""
from pathlib import Path
import hashlib, re
BASE = Path('/Users/home/cardboard-implementation-20261014/snapshots/source-status5-vtccd-diagnostics-125')
CAND = Path(__file__).resolve().parent/'candidate'
EXPECTED = {
 'matter/src/runtime.mm': '1719f14310d643ca96c520eb47b3028f8259f85e3da0d263460921b06d94ac74',
 'matter/src/metal/fem.metalinc': '70c2f396045f9abdcfdc32f5714e6e3a400f5566e5a7824353e11b307de52b7d',
}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
for rel, expected in EXPECTED.items():
    assert sha(BASE/rel)==expected, f'frozen base changed: {rel}'
runtime=(CAND/'matter/src/runtime.mm').read_text()
metal=(CAND/'matter/src/metal/fem.metalinc').read_text()
# The diagnostic buffer keeps its ABI index/type but is explicitly the private trial metrics.
assert 'id<MTLBuffer> trialMetrics)' in runtime
assert '[encoder setBuffer:trialMetrics != nil ? trialMetrics : state.dummy' in runtime
assert '[encoder setBytes:&hasTrialMetrics length:sizeof(hasTrialMetrics) atIndex:18u]' in runtime
assert 'const std::uint32_t hasTrialMetrics = trialMetrics != nil ? 1u : 0u;' in runtime
assert 'device const float4* trialMetrics [[buffer(12)]]' in metal
assert 'constant uint& hasTrialMetrics [[buffer(18)]]' in metal
assert 'record.scalars.y = trialMetrics[environment].z;' in metal
assert 'lineSearch[environment].x' not in metal
private_stages=(30,31,40,41)
for stage in private_stages:
    m=re.search(rf'captureStatus5\({stage}u,.*?\);',runtime,re.S)
    assert m, f'missing stage {stage}'
    assert 'state.femArmijoMetrics' in m.group(0), f'stage {stage} has wrong alpha lane'
    assert 'state.femLineSearch' not in m.group(0), f'stage {stage} binds legacy line-search scratch'
for stage in (10,20,50,60):
    m=re.search(rf'captureStatus5\({stage}u,.*?\);',runtime,re.S)
    assert m and re.search(r',\s*nil\);$',m.group(0)), f'global stage {stage} must mark alpha unavailable'
print('PASS: frozen base hashes, four private call bindings, four global unavailable bindings, kernel reads Armijo metrics.z')
