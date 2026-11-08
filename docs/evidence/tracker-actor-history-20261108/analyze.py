from pathlib import Path
import re
p=Path(__file__).parent
old=(p/'old.history').read_bytes()
new=(p/'candidate.history').read_bytes()
rep=(p/'candidate_repeat.history').read_bytes()
assert len(old)==len(new)==len(rep)==30*3*816
assert new==rep
trans=re.findall(r'transition\[(\d+)\] done=\d+ timeout=\d+ physics_error=(\d+)',(p/'candidate.log').read_text())
assert len(trans)==60
fail=[int(i) for i,e in trans if int(e)]
assert fail==[26,28,52,54,55],fail
changed=[]
for step in range(30):
 for env in range(2):
  def snap(data,phase):
   start=step*3*816+phase*816+env*408
   return data[start:start+408]
  before,observed,after=(snap(new,i) for i in range(3))
  obefore,oobserved,oafter=(snap(old,i) for i in range(3))
  assert (before,observed)==(obefore,oobserved)
  assert oafter==oobserved
  assert after==(before if step*2+env in fail else observed),(step,env)
  if before!=observed: changed.append((step,env))
for step in (13,26):
 assert (step,0) in changed and (step,1) in changed
for ext in ('q','v','scene','actor','status'):
 data=(p/f'old.{ext}').read_bytes()
 assert data==(p/f'candidate.{ext}').read_bytes()==(p/f'candidate_repeat.{ext}').read_bytes(),ext
print('passed: 30 steps, 60 environment transitions; failures:',[(i//2,i%2) for i in fail])
print('failed histories restored; accepted histories kept; legacy retained visual writes; mixed steps 13 and 26 both environments changed')
print('q, v, scene, actor, status match old and repeated candidate; candidate history repeats byte-for-byte')
