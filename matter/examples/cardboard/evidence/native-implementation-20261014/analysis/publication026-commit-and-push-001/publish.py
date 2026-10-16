from pathlib import Path
import datetime
import hashlib
import json
import subprocess

R=Path('/Users/home/numi-cardboard-20261006')
B=Path('/Users/home/cardboard-implementation-20261014')
O=B/'analysis/publication026-commit-and-push-001'
O.mkdir(exist_ok=True)
def run(args):return subprocess.run(args,cwd=R,text=True,capture_output=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
head=run(['git','rev-parse','HEAD']).stdout.strip()
assert run(['git','branch','--show-current']).stdout.strip()=='main'
remote=run(['git','ls-remote','origin','refs/heads/main'])
assert remote.returncode==0 and remote.stdout.split()[0]==head
ap=B/'analysis/publication026-independent-byte-audit-001/report.json'
audit=json.loads(ap.read_text())
assert audit['verdict']=='PASS_BYTE_AND_PREFIX_AUDIT; physical qualification not evaluated'
for key in ['compressed_copies_and_restores_match','eligible_external_file_inventories_complete',
            'external_source_bytes_and_hashes_match','new_pathspec_tail_exact',
            'prior_pathspec_exact_byte_prefix','prior_prefix_exact_json_objects',
            'raw_repository_copies_match','six_axis_verdicts_preserved_unqualified',
            'staged_expected_paths_only']:
    assert audit[key] is True,key
assert not audit['selected_149_460_463_case_processes_active']
(O/'preflight.json').write_text(json.dumps({'head':head,'branch':'main',
    'remote_main':remote.stdout.strip(),'audit_sha256':sha(ap),
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
commit=run(['git','commit','-m','Add bounded native block right scaling and retain failed solves'])
(O/'commit.stdout').write_text(commit.stdout);(O/'commit.stderr').write_text(commit.stderr)
assert commit.returncode==0,commit.stdout+commit.stderr
after=run(['git','rev-parse','HEAD']).stdout.strip()
push=run(['git','push','origin','main'])
(O/'push.stdout').write_text(push.stdout);(O/'push.stderr').write_text(push.stderr)
assert push.returncode==0,push.stdout+push.stderr
published=run(['git','ls-remote','origin','refs/heads/main'])
assert published.returncode==0 and published.stdout.split()[0]==after
(O/'publish.py').write_bytes(Path(__file__).read_bytes())
(O/'receipt.json').write_text(json.dumps({'schema':'numi.cardboard.publication026_commit_push.v1',
    'parent':head,'commit':after,'remote_main':published.stdout.split()[0],
    'remote_verified':True,'commit_exit_code':commit.returncode,'push_exit_code':push.returncode,
    'audit_sha256':sha(ap),'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'qualification':'scoped native conditioning milestone; no complete coupon or downstream physical qualification'},indent=2)+'\n')
print(json.dumps({'commit':after,'remote_verified':True}))
