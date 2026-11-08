from pathlib import Path
import struct, re
p=Path(__file__).parent
def f(name,ext): return (p/f'world-{name}.{ext}').read_bytes()
for ext in ('q','v','scene','status','reset'):
 assert f('baseline',ext)==f('final',ext), ext
for ext in ('q','v','scene','status','reset','actor','history'):
 assert f('final',ext)==f('final-repeat',ext), ext
 assert f('baseline',ext)==f('baseline-repeat',ext), ext
reset=struct.unpack('<60I',f('final','reset'))
assert [(i//2,i%2) for i,v in enumerate(reset) if v]==[(0,0),(0,1),(14,0),(15,0),(27,0),(28,0),(28,1)]
h0=f('baseline','history');h1=f('final','history')
def snap(data,step,phase,env):
 o=step*3*816+phase*816+env*408
 return data[o:o+408]
def track(data,step,phase,env,h):
 return struct.unpack_from('<7f',snap(data,step,phase,env),h*34*4)
for step,env in ((14,0),(27,0),(28,0),(28,1)):
 assert reset[step*2+env] and step%5 != 0
 for j in range(3):
  assert track(h0,step,1,env,j)[0]==1.0
  assert track(h1,step,1,env,j)==(0.0,)*7
for step in (14,27): assert snap(h0,step,1,1)==snap(h1,step,1,1)
assert snap(h0,15,1,0)==snap(h1,15,1,0) and track(h1,15,1,0,0)[0]==1.0
trans=re.findall(r'transition\[(\d+)\] done=\d+ timeout=\d+ physics_error=(\d+)',(p/'world-final.log').read_text())
assert len(trans)==60
failed=[int(i) for i,e in trans if int(e)]
assert failed==[26,28,52,54,55]
for step,env in ((14,0),(27,0)):
 assert step*2+env in failed
 assert snap(h1,step,2,env)==snap(h1,step,0,env)
for env in (0,1):
 assert 28*2+env not in failed
 assert snap(h1,28,2,env)==snap(h1,28,1,env)
print('paired real fixture: reset masks and physics arrays match; both arms repeat byte-for-byte')
print('baseline exposed cached track at nonsampled resets 14/0, 27/0, 28/0, 28/1; candidate suppresses all three track slots')
print('non-reset neighbors unchanged at steps 14 and 27; reset frame is visible again at sampled step 15')
print('failed steps 14/0 and 27/0 restore pre-visual history; accepted step 28 retains the invalid observation')
