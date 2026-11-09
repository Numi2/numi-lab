from pathlib import Path
import os,subprocess,tempfile
R=Path.cwd();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();assert commit.startswith('aa57267')
with tempfile.TemporaryDirectory(prefix='crow-github-restore-') as t:
 dest=Path(t)/'checkout';env={**os.environ,'GIT_TERMINAL_PROMPT':'0'}
 def run(cmd,cwd=None):return subprocess.run(cmd,cwd=cwd,env=env,check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 run(['git','clone','--filter=blob:none','--depth=1','--no-checkout','--branch','agent/american-crow-rl-20261107','git@github.com:Numi2/numi-lab.git',str(dest)])
 assert run(['git','rev-parse','HEAD'],cwd=dest).stdout.strip()==commit
 run(['git','sparse-checkout','init','--no-cone'],cwd=dest)
 run(['git','sparse-checkout','set','--no-cone','/artifacts/crow-v15-ground-contact-20261109/'],cwd=dest)
 run(['git','checkout'],cwd=dest)
 evidence=dest/'artifacts/crow-v15-ground-contact-20261109';p=run(['shasum','-a','256','-c','SHA256SUMS'],cwd=evidence);rows=p.stdout.splitlines();assert len(rows)==200 and all(x.endswith(': OK') for x in rows),(len(rows),rows[-5:]);print('Verified independently downloaded GitHub commit',commit,'with',len(rows),'matching archived files.');print('Temporary independent restore removed after successful verification.')
