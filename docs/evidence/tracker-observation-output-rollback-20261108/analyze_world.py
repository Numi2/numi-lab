from pathlib import Path
import re,struct,json,sys
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path('world-captures/final')
for ext in ('q','v','scene','actor','status','reset','history','observation'):
 assert (root/f'first.{ext}').read_bytes()==(root/f'repeat.{ext}').read_bytes(),ext
for ext in ('q','v','scene','status','reset'):
 assert (root/f'first.{ext}').read_bytes()==(Path('../tracker-reset-frame-20261108')/f'world-final.{ext}').read_bytes(),ext
log=Path('world-first.log').read_text(); match=re.search(r'output_per_env=(\d+)',log); assert match
width=int(match.group(1));assert width==102
trans=re.findall(r'transition\[(\d+)\] done=\d+ timeout=\d+ physics_error=(\d+)',log);assert len(trans)==60
failed={int(i) for i,e in trans if int(e)};assert failed=={26,28,52,54,55}
reset=struct.unpack('<60I',(root/'first.reset').read_bytes()); data=(root/'first.observation').read_bytes();assert len(data)==30*3*2*width*4
row=lambda step,phase,env:struct.unpack_from(f'<{width}I',data,((step*3+phase)*2+env)*width*4)
visual={h*34+i for h in range(3) for i in range(7)}
counts={'accepted':0,'rejected_reset':0,'rejected_no_reset':0,'accepted_changed':0,'rejected_changed':0,'mixed_steps':0}
for step in range(30):
 if (2*step in failed)!=(2*step+1 in failed):counts['mixed_steps']+=1
 for env in range(2):
  index=step*2+env;a,b,c=(row(step,p,env) for p in range(3));isfail=index in failed
  if not isfail:
   assert c==b,(step,env,'accepted');counts['accepted']+=1;counts['accepted_changed']+=int(a!=b)
  elif not reset[index]:
   assert c==a,(step,env,'rejected');counts['rejected_no_reset']+=1;counts['rejected_changed']+=int(a!=b)
  else:
   assert all(c[i]==0 if i in visual else c[i]==a[i] for i in range(width)),(step,env,'reset')
   counts['rejected_reset']+=1
assert all(counts[k]>0 for k in ('rejected_reset','rejected_no_reset','accepted_changed','rejected_changed','mixed_steps'))
Path('world-analysis.json').write_text(json.dumps(counts,indent=2)+'\n')
print(json.dumps(counts,indent=2))
