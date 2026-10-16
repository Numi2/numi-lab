from pathlib import Path
import hashlib
import json
import shutil

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
S = B / 'snapshots/source-shared-contact-budget-184'
P = B / 'snapshots/source-admission-encoder-182-003'
O = B / 'analysis/source184-shared-contact-budget-001'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert not S.exists() and not O.exists()
O.mkdir()
frozen = json.loads((P / 'frozen-source.json').read_text())
changed = [
    'matter/CMakeLists.txt', 'matter/src/runtime.mm',
    'matter/tools/cardboard_shipping_memory.hpp',
    'matter/tools/cardboard_shipping_memory_check.cpp',
]
added = [
    'matter/include/numi/matter/detail/full_contact_armijo_scratch_budget.hpp',
    'matter/tools/cardboard_full_contact_armijo_scratch_budget_check.cpp',
]
assert sorted(name for name, digest in frozen['source_sha256'].items()
              if sha(R / name) != digest) == sorted(changed)
for name in added:
    assert name not in frozen['source_sha256'] and (R / name).is_file()
shutil.copytree(P, S)
for name in changed + added:
    target = S / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(R / name, target)
    frozen['source_sha256'][name] = sha(target)
frozen['base_snapshot'] = str(P)
frozen['scope'] = ('Source182003 plus shared exact full-contact private scratch byte inventory '
    'used by runtime and shipping estimator. Includes per-candidate status allocations and '
    'typed sentinels. No Metal/operator/tolerance changes. No native qualification implied.')
(S / 'frozen-source.json').write_text(json.dumps(frozen, indent=2) + '\n')
assert len(frozen['source_sha256']) == 807
for name, digest in frozen['source_sha256'].items():
    assert sha(S / name) == digest
(O / 'source-binding.json').write_text(json.dumps({
    'schema': 'numi.cardboard.source184-shared-budget.v1',
    'parent_frozen_sha256': sha(P / 'frozen-source.json'),
    'frozen_sha256': sha(S / 'frozen-source.json'),
    'source_files_verified': 807,
    'changed': {name: sha(S / name) for name in changed},
    'added': {name: sha(S / name) for name in added},
    'gpu_dispatch': False,
}, indent=2) + '\n')
(O / 'prepare.py').write_bytes(Path(__file__).read_bytes())
print((O / 'source-binding.json').read_text())
