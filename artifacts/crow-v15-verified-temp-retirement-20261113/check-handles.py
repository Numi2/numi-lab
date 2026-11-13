import json,shlex,subprocess
from pathlib import Path
p=Path('/tmp/crow-retirement-verified.json');rows=json.loads(p.read_text());
for row in rows:
 path=row['path'];q=shlex.quote(path)
 cmd=f'if [ -L {q} ] || [ ! -f {q} ]; then exit 8; fi; stat -f "%z %i %m" {q}; /usr/sbin/lsof -t -- {q}; rc=$?; [ "$rc" -eq 1 ]'
 r=subprocess.run(['ssh','-4','-o','HostName=192.168.1.141','macmini',cmd],capture_output=True,text=True)
 if r.returncode:raise SystemExit(f'Not safe to retire {path}: {r.stderr} {r.stdout}')
 bits=r.stdout.strip().split();assert len(bits)==3;row.update({'size_bytes':int(bits[0]),'inode':int(bits[1]),'mtime_epoch':int(bits[2]),'open_handles_at_check':0})
df=subprocess.check_output(['ssh','-4','-o','HostName=192.168.1.141','macmini','df','-k','/System/Volumes/Data'],text=True)
Path('/tmp/crow-retirement-prepared.json').write_text(json.dumps({'scope':'Eight exact uncompressed build outputs, not sole evidence; every gzip archive independently retrieved from GitHub, decompressed and compared with source SHA-256. The owning host must recheck hashes and open handles immediately before deletion.','before_df_k':df,'rows':rows},indent=2)+'\n')
print('Prepared',len(rows),'files, bytes',sum(x['size_bytes'] for x in rows))
