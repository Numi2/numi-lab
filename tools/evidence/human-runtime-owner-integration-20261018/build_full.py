from pathlib import Path
import json,shlex,subprocess,hashlib,time,concurrent.futures
p=Path(__file__).parent;s=p/'full-source';b=p/'full-build';b.mkdir(exist_ok=True)
entries=json.loads((p/'full-library-commands-base.json').read_text());old='/Users/n/numi-lab-complete-body-capture-1276';commands=[]
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
def compile_one(e):
 rel=str(Path(e['file']).relative_to(old));source=s/rel;obj=b/e['output'];obj.parent.mkdir(parents=True,exist_ok=True);a=shlex.split(e['command']);a=[x.replace(old,str(s)) for x in a]
 a[a.index('-o')+1]=str(obj);a[a.index('-c')+1]=str(source)
 defs=[]
 if rel=='matter/src/runtime.mm':defs=['-DNM_MATTER_TEST_SOURCE_LOADER=1']
 if rel=='src/metal/MetalNumanXHumanMatter.mm':defs=['-DMETALROBO_TEST_HUMAN_MATTER_SOURCE_LOADER=1']
 if rel=='src/metal/MetalNumanXCoupledHuman.mm':defs=['-DMETALROBO_TEST_COUPLED_HUMAN_SOURCE_LOADER=1']
 if rel=='src/metal/MetalArticulatedOperator.mm':defs=['-DMETALROBO_TEST_ARTICULATED_SOURCE_LOADER=1']
 if rel=='matter/src/numi_human.mm':
  q=p/'shaders/MatterSource.metal';defs=['-DNUMI_HUMAN_TENDON_FEM_TEST_SOURCE_PATH="'+str(q)+'"','-DNUMI_HUMAN_TENDON_FEM_TEST_SOURCE_SHA256="'+sha(q)+'"']
 a[1:1]=defs+['-I'+str(s/'matter/src'),'-I'+str(s/'src/metal'),'-MMD','-MF',str(obj)+'.d']
 t=time.monotonic();r=subprocess.run(a,cwd=e['directory'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);log=Path(str(obj)+'.log');log.write_text(r.stdout)
 result={'source':rel,'source_sha256':sha(source),'argv':a,'exit_code':r.returncode,'seconds':time.monotonic()-t,'object':str(obj),'object_sha256':sha(obj) if obj.exists() else None,'log':str(log)}
 print(rel,r.returncode,round(result['seconds'],2),flush=True)
 return result
start=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for r in pool.map(compile_one,entries):
  commands.append(r);(b/'compile-receipt.json').write_text(json.dumps({'source_head':'a8d7312ff273e4a4e6707a829504d797e0b551f1','commands':commands,'wall_seconds':time.monotonic()-start},indent=2))
failed=[x for x in commands if x['exit_code']];print('COMPILE_FINISHED',len(commands),'FAILED',len(failed),flush=True)
if failed:raise SystemExit(1)
