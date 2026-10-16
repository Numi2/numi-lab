from pathlib import Path
import datetime,hashlib,json,subprocess
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');R=Q/'checkout-publication032';O=Q/'analysis/publication033-commit-and-push-001'
assert not O.exists();O.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checked(args,**kw):
 p=subprocess.run(args,cwd=R,capture_output=True,**kw);assert p.returncode==0,(args,p.stderr.decode());return p.stdout
def emit(n,v):(O/n).write_text(json.dumps(v,indent=2)+'\n')
ap=Q/'analysis/publication033-independent-byte-audit-002/report.json';a=json.loads(ap.read_text())
assert a['verdict']=='PASS' and a['historical_records_unchanged']==1669 and not a['full_protocol_qualification']
# An independent agent audit is required in addition to root's strict byte verifier.
agentpath=Q/'analysis/publication033-independent-review-006/report.json'
agent=json.loads(agentpath.read_text())
assert agent['verdict']=='PASS' and agent['six_axes_open'] and not agent['active_native521_outputs_included'],agent
assert agent['summary_sha256']==a['summary_sha256'] and agent['staged_pathlist_sha256']==a['staged_pathlist_sha256']
parent=checked(['git','rev-parse','HEAD']).decode().strip();assert parent=='877cd6e69657c9a7b4e5032bcd475705bcc06c85'
assert checked(['git','branch','--show-current']).decode().strip()=='main'
assert checked(['git','ls-remote','origin','refs/heads/main']).decode().split()[0]==parent
pl=Q/'publication033-staged-paths.nul';assert sha(pl)==a['staged_pathlist_sha256'];expected=set(pl.read_bytes().split(b'\0'))-{b''};assert len(expected)==1632
assert not checked(['git','diff','--cached','--name-only','-z'])
frozen=Q/'snapshots/source-contact-damped-memory-198-001/frozen-source.json';assert sha(frozen)==a['source198_frozen_sha256']
sources=json.loads(frozen.read_text())['source_sha256']
for n,h in sources.items():assert sha(R/n)==h,n
assert checked(['git','diff','--check'])==b''
checked(['git','add','-f','--pathspec-from-file='+str(pl),'--pathspec-file-nul'])
actual=set(checked(['git','diff','--cached','--name-only','-z']).split(b'\0'))-{b''};assert actual==expected
assert not checked(['git','diff','--name-only','-z']) and not checked(['git','diff','--cached','--name-only','--diff-filter=D','-z'])
blobs=checked(['git','cat-file','--batch'],input=b''.join(b':'+p+b'\n' for p in sorted(expected)));offset=0
for p in sorted(expected):
 end=blobs.index(b'\n',offset);fields=blobs[offset:end].split();assert len(fields)==3 and fields[1]==b'blob';size=int(fields[2]);offset=end+1
 assert blobs[offset:offset+size]==(R/p.decode()).read_bytes(),p;offset+=size;assert blobs[offset:offset+1]==b'\n';offset+=1
assert offset==len(blobs)
pre={'parent':parent,'audit_sha256':sha(ap),'independent_agent_audit_sha256':sha(agentpath),'staged_paths':len(expected),'staged_pathlist_sha256':sha(pl),'source_hashes_verified':len(sources),'all_staged_blob_bytes_match':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};emit('preflight.json',pre)
for label,cmd in [('commit',['git','-c','gc.auto=0','commit','-m','Repair mixed-contact trials and preserve bounded damped-solver evidence']),('push',['git','-c','gc.auto=0','push','origin','main'])]:
 p=subprocess.run(cmd,cwd=R,capture_output=True);(O/(label+'.stdout')).write_bytes(p.stdout);(O/(label+'.stderr')).write_bytes(p.stderr);assert p.returncode==0,(label,p.stderr.decode())
head=checked(['git','rev-parse','HEAD']).decode().strip();remote=checked(['git','ls-remote','origin','refs/heads/main']).decode().split()[0];assert head==remote
assert checked(['git','status','--porcelain'])==b''
(O/'publish.py').write_bytes(Path(__file__).read_bytes())
receipt={'schema':'numi.cardboard.publication033-commit-push.v1',**pre,'commit':head,'remote_main':remote,'remote_verified':True,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'qualification':'Regional synthetic contact fixture reaches intended postCommit fault and rolls back; damped production-kernel checks pass for positive damping. All six full-protocol axes remain open. Active521 excluded.'}
emit('receipt.json',receipt);print(json.dumps(receipt,indent=2))
