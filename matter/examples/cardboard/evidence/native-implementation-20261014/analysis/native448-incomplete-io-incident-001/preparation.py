from pathlib import Path
import datetime, hashlib, json, os

B = Path('/Users/home/cardboard-implementation-20261014')
old = B / 'solver/bounded-cgls64-exact-state-448'
new = B / 'solver/bounded-cgls64-exact-state-450'
incident = B / 'analysis/native448-incomplete-io-incident-001'
incident.mkdir(exist_ok=False)
files = {}
for p in sorted(old.rglob('*')):
    if p.is_file() and 'inputs' not in p.relative_to(old).parts:
        files[str(p.relative_to(old))] = {
            'bytes': p.stat().st_size,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
        }
report = {
    'schema': 'numi.cardboard.operational-incident.v1',
    'recorded_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'case': str(old), 'wrapper_observed_exit_code': 120,
    'observation_source': 'root exec session 58122 completion; disk-full observations retained in conversation and cleanup017',
    'receipt_present': (old / 'receipt.json').exists(),
    'cgls_capture_present': (old / 'base-cgls.bin').exists(),
    'verdict': 'incomplete_evidence_due_to_disk_exhaustion',
    'limitations': [
        'Do not infer whether the selected CGLS recurrence executed.',
        'CSV fgmres_iterations is not the Newton iteration index.',
        'Partial output cannot qualify production acceptance or derivative correctness.',
        'A fresh same-protocol case is required; the original case remains preserved.',
    ],
    'retained_noninput_files': files,
}
(incident / 'incident.json').write_text(json.dumps(report, indent=2) + '\n')
command = json.loads((B / 'native-448-command.json').read_text())
command = [x.replace(str(old), str(new)) for x in command]
command[command.index('--question') + 1] = (
    'Repeat incomplete-I/O attempt448 with identical source142/protocol: does native CGLS64 '
    'produce a verified descent candidate at exact primary Armijo failure1:48?')
command[command.index('--prediction') + 1] += (
    ' This is a fresh same-protocol rerun after448 lost evidence under disk exhaustion; '
    '448 is not treated as a numerical success or failure.')
separator = command.index('--')
command[separator:separator] = ['--input-file', str(incident / 'incident.json')]
(B / 'native-450-command.json').write_text(json.dumps(command, indent=2) + '\n')
(incident / 'preparation.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'prepared': str(B / 'native-450-command.json'),
                  'incident': str(incident / 'incident.json'),
                  'available_bytes': os.statvfs(B).f_bavail * os.statvfs(B).f_frsize}))
