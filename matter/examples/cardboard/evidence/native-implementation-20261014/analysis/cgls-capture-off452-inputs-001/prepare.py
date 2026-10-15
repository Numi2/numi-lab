from pathlib import Path
import hashlib, json, shutil

B = Path('/Users/home/cardboard-implementation-20261014')
old = B / 'solver/bounded-cgls64-exact-state-450'
new = B / 'solver/bounded-cgls64-capture-off-parity-452'
receipt_path = old / 'receipt.json'
receipt = json.loads(receipt_path.read_text())
assert receipt['execution_started'] and receipt['source_inputs_unchanged']
assert not receipt['timed_out']
assert json.loads((B / 'analysis/native450-launch-guard-001/receipt.json').read_text())['space_abort'] is False
O = B / 'analysis/cgls-capture-off452-inputs-001'
O.mkdir(exist_ok=False)
alias = O / 'native450-observer-on-receipt.json'
shutil.copy2(receipt_path, alias)
assert hashlib.sha256(alias.read_bytes()).digest() == hashlib.sha256(receipt_path.read_bytes()).digest()
c = json.loads((B / 'native-450-command.json').read_text())
c = [x.replace(str(old), str(new)) for x in c]
for prefix in ['NM_FEM_BASE_CGLS_CAPTURE_NEWTON=', 'NM_FEM_BASE_CGLS_CAPTURE_PATH=']:
    indices = [i for i, value in enumerate(c) if value.startswith(prefix)]
    assert len(indices) == 1 and c[indices[0] - 1] == '--env'
    i = indices[0]
    del c[i-1:i+1]
c[c.index('--question') + 1] = 'Does disabling only the CGLS scratch capture preserve the entire attempted source142 CGLS1:48 coupon history from450?'
c[c.index('--prediction') + 1] = (
    'Expect exact numerical/accepted/rejected histories, physical files, ordinary '
    'Armijo and true-residual traces through terminal rollback, apart from paths, '
    'provenance, capture marker/file and timing. Only the two CGLS scratch capture '
    'environment keys are removed; the CGLS algorithm and all physical/numerical '
    'settings and other diagnostics remain identical. A parity pass is bounded '
    'observer evidence and does not complete the coupon or convergence study.')
i = c.index('--')
c[i:i] = ['--input-file', str(alias)]
external = [Path(c[i+1]) for i, x in enumerate(c) if x == '--input-file']
assert len({p.name for p in external}) == len(external)
assert all(p.is_file() for p in external)
(B / 'native-452-command.json').write_text(json.dumps(c, indent=2) + '\n')
(O / 'prepare.py').write_bytes(Path(__file__).read_bytes())
guard = (B / 'run_native450.py').read_text().replace('native-450-command.json', 'native-452-command.json').replace('native450-launch-guard-001', 'native452-launch-guard-001').replace('native450-runner.log', 'native452-runner.log')
(B / 'run_native452.py').write_text(guard)
print('452 observer-off pair prepared; no native execution')
