from pathlib import Path
import json,subprocess,hashlib,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');r=q/'checkout-publication032';d=q/'analysis/publication042-upstream-001';d.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old='89da10e5dd7d620db154c3cea279803a112a302a';new='420b4f866decdfaed1f5a81ef102d3cb6efbb442'
git=lambda *a:subprocess.check_output(['git',*a],cwd=r)
assert git('rev-parse','HEAD').decode().strip()==old and git('branch','--show-current').decode().strip()=='main' and not git('status','--porcelain=v1','--untracked-files=all')
assert git('ls-remote','origin','refs/heads/main').decode().split()[0]==new
paths=git('diff','--name-only',old,new).decode().splitlines();assert len(paths)==10 and not any(n.startswith('matter/') or n=='CMakeLists.txt' for n in paths)
p=subprocess.run(['git','merge','--ff-only','origin/main'],cwd=r,stdout=subprocess.PIPE,stderr=subprocess.PIPE);(d/'stdout.log').write_bytes(p.stdout);(d/'stderr.log').write_bytes(p.stderr);assert p.returncode==0
assert git('rev-parse','HEAD').decode().strip()==new and not git('status','--porcelain=v1','--untracked-files=all')
row={'schema':'numi.cardboard.publication042-upstream.v1','old_head':old,'new_head':new,'preserved_paths':{n:sha(r/n) for n in paths},'clean':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row))
