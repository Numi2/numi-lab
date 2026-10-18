from pathlib import Path
import hashlib,re,json
p=Path(__file__).parent;s=p/'full-source';out=p/'shaders';out.mkdir(exist_ok=True);dirs=[s/'include',s/'matter/include',s/'src/metal',s/'matter/src'];pins={}
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
def expand(q,stack=()):
 q=q.resolve();assert q not in stack,str(q);raw=q.read_text();pins[str(q)]=sha(q)
 def sub(m):
  delim,name=m[1],m[2];target=next((a.resolve() for a in [q.parent/name]+[d/name for d in dirs] if a.is_file()),None)
  if target is None:
   assert delim=='<',(q,name)
   return m[0]
  return '\n'+expand(target,stack+(q,))+'\n'
 result=re.sub(r'^\s*#include\s+([<"])([^">]+)[">][^\n]*',sub,raw,flags=re.M)
 if re.search(r'^\s*#pragma once\s*$',raw,re.M):
  guard='NUMI_INCLUDE_'+hashlib.sha256(str(q).encode()).hexdigest();result=re.sub(r'^\s*#pragma once\s*$','',result,flags=re.M);result=f'\n#ifndef {guard}\n#define {guard}\n{result}\n#endif\n'
 return result
mods=['ArticulatedOperator','ArticulatedOperatorCompensated','MillardMuscleReference','MujocoMuscleReference','MujocoMuscleReferenceCompensated','MujocoMuscleReferenceCached','NumiHumanTendonTransfer','NumiHumanTendonTransferCompensated','NumanXHumanMatterOwner'];entries=[]
for name in mods:
 q=out/(name+'.metal');q.write_text(expand(s/'src/metal'/(name+'.metal')));entries.append((sha(q),str(q)))
manifest=out/'operator-modules.tsv';manifest.write_text(''.join(h+'\t'+q+'\n' for h,q in entries))
for rel,name in [('src/metal/NumiHumanStand.metal','StandSource.metal'),('matter/src/matter_kernels.metal','MatterSource.metal'),('src/metal/NumanXHumanMatter.metal','HumanMatterSource.metal')]:
 q=out/name;q.write_text(expand(s/rel));print(name,sha(q),q.stat().st_size)
(out/'source-pins.json').write_text(json.dumps(pins,indent=2));print('files',len(pins),'manifest',sha(manifest))
