from pathlib import Path
import subprocess,sys
G=Path('artifacts/crow-rl-git-recovery-quality-tolerances-20261108')
p=subprocess.run([sys.executable,'tools/crow_evidence_archive.py','omit',str(G/'omission.json')],capture_output=True)
(G/'omit.stdout').write_bytes(p.stdout);(G/'omit.stderr').write_bytes(p.stderr)
print(p.stdout.decode(),end='');print(p.stderr.decode(),end='',file=sys.stderr)
assert p.returncode==0
