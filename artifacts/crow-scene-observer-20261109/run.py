from pathlib import Path
import gzip,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-scene-observer-20261109'; B=R/'build/crow-scene-observer-20261109'
cases={'landing':'artifacts/crow-landing-height-aware-20261109/route-results.json','held-out-a':'artifacts/crow-route-generalization-20261109/held-out-a/route-results.json'}
out={}
for name,path in cases.items():
 original=json.loads((R/path).read_text())['argv'];cmd=original.copy()
 for k,v in [('--steps','100'),('--state-trace',str(B/(name+'.tsv')))]:cmd[cmd.index(k)+1]=v
 env={**os.environ,'CROW_CONTACT_TRACE':'1','CROW_SCENE_TRACE':'1','DYLD_LIBRARY_PATH':str(B),'PYTHONDONTWRITEBYTECODE':'1'}
 p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=120)
 if p.returncode:raise RuntimeError(p.stderr[-5000:])
 def parse(prefix):return [json.loads(s.split(' ',1)[1]) for s in p.stderr.splitlines() if s.startswith(prefix)]
 scene=parse('CROW_SCENE_SAMPLE '); sample=parse('CROW_FLIGHT_SAMPLE ')
 old=R/path; old=old.parent/'tracking.stderr.gz'
 with gzip.open(old,'rt') as f:
  oldsamples=[json.loads(s.split(' ',1)[1]) for s in f if s.startswith('CROW_FLIGHT_SAMPLE ')]
 before={x['lane']:x for x in oldsamples if x['step']==100}
 matches=sum(x['q']==before[x['lane']]['q'] and x['v']==before[x['lane']]['v'] for x in sample)
 (A/(name+'-scene.json')).write_text(json.dumps(scene,indent=2)+'\n')
 out[name]={'scene_count':len(scene),'sample_count':len(sample),'matching_qv_at_100':matches,'all_matched':len(sample)==64 and matches==64,'argv':cmd}
 print(name,out[name],flush=True)
(A/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
