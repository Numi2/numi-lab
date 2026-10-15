from pathlib import Path
import datetime,hashlib,json,subprocess
b=Path('/Users/home/cardboard-implementation-20261014');r=Path('/Users/home/numi-cardboard-20261006');o=b/'analysis/frozen-source095-restoration-audit-001';o.mkdir(exist_ok=False);s=b/'snapshots/source-contact-capture-and-resolved-material-sweep-095';m=json.loads((s/'frozen-source.json').read_text());h=lambda d:hashlib.sha256(d).hexdigest()
for rel,sha in m['source_sha256'].items():assert h((s/rel).read_bytes())==sha,rel
verified=[]
for revision in ['d328c1150173b95c30d19ec4cf33262a6311baee',subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip()]:
 raw=subprocess.check_output(['git','cat-file','--batch'],cwd=r,input=''.join(revision+':'+p+'\n' for p in m['source_sha256']).encode());offset=0
 for rel,sha in m['source_sha256'].items():
  end=raw.index(b'\n',offset);head=raw[offset:end].split();assert len(head)==3 and head[1]==b'blob',rel;n=int(head[2]);payload=raw[end+1:end+1+n];assert len(payload)==n and h(payload)==sha,rel;assert raw[end+1+n:end+2+n]==b'\n';offset=end+2+n
 assert offset==len(raw);verified.append(revision)
record=dict(schema='numi.cardboard.restored-source-audit.v1',observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_manifest_sha256=h((s/'frozen-source.json').read_bytes()),source_bindings=len(m['source_sha256']),all_current_snapshot_bytes_match=True,all_published_commit_bytes_match=verified,incident='Owner diagnostic overlay had a symlink and briefly edited two lines of frozen095 mixed_fem.metalinc; owner reported exact restoration. This independent audit verifies all774 current bytes against both original manifest and published commits. Exact incident diff/timing is retained separately by owner.',execution_scope='Root has run no095 native GPU or build during the reported transient edit.096/097 builds and native316/317/318 used independent source copies made earlier. Existing published095 evidence remains bound to its original source/build receipts.',physical_qualification=False)
(o/'report.json').write_text(json.dumps(record,indent=2)+'\n');(o/'run.py').write_bytes(Path(__file__).read_bytes());(o/'manifest.json').write_text(json.dumps(dict(files={p.name:dict(bytes=p.stat().st_size,sha256=h(p.read_bytes())) for p in o.iterdir() if p.is_file()}),indent=2)+'\n');print(json.dumps(record,indent=2))
