import hashlib,json,os,pathlib,subprocess
p=pathlib.Path.cwd();env=dict(os.environ,NUMI_CCD_EVENT_TRACE='1',DYLD_LIBRARY_PATH='/Users/numi/numi-lab-horizon-main-build-20261108/lib')
base=pathlib.Path('/Users/numi/numi-lab-ccd-thin-evidence-20261107/cold-four-sliding-v1/candidate.metallib');binary=p/'probe'
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
records=[]
for name,lib in [('baseline',base)]+[(f'slot-{i}',p/f'instrumented-{i}.metallib') for i in range(13)]:
 print(name,flush=True)
 with (p/f'{name}.json').open('w') as out,(p/f'{name}.stderr').open('w') as err:
  proc=subprocess.run([str(binary),str(lib)],env=env,stdout=out,stderr=err,timeout=90)
 records.append({'name':name,'exit':proc.returncode,'metallib_sha256':sha(lib),'output_sha256':sha(p/f'{name}.json'),'trace_sha256':sha(p/f'{name}.stderr')})
 (p/'runs.json').write_text(json.dumps({'probe_sha256':sha(binary),'runs':records},indent=2)+'\n')
 if proc.returncode:raise SystemExit('probe failed: '+name)
