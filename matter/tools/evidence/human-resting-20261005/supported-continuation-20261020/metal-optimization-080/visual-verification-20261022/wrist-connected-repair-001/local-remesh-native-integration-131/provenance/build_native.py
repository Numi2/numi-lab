from pathlib import Path
import subprocess,json,hashlib,time
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019')
E=R/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
D=E/'local-remesh-native-integration-131'
B=E/'wholebody-attachment-source-rebind-036-ankle128-regenerated-deltas-001'
old=B/'build-main-0a693-004';build=D/'build-native-001'
if not (D/'provenance/input-hook-ready.json').is_file():raise SystemExit('no tested input hook; refusing build')
build.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
results={}
for stage in ['compile','link']:
    cmd=json.loads((old/(stage+'-command.json')).read_text())
    cmd=[s.replace(str(old),str(build)).replace(str(B/'source-overlay/apps'),str(D/'source-overlay/apps')) for s in cmd]
    (build/(stage+'-command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
    t=time.monotonic();r=subprocess.run(cmd,cwd=R,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (build/(stage+'.stdout.log')).write_bytes(r.stdout);(build/(stage+'.stderr.log')).write_bytes(r.stderr)
    results[stage]={'returncode':r.returncode,'wall_seconds':time.monotonic()-t}
    (build/(stage+'-result.json')).write_text(json.dumps(results[stage],indent=2)+'\n')
    print(stage,results[stage],flush=True)
    if r.returncode:raise SystemExit(r.stderr.decode()[-10000:])
(D/'build-manifest.json').write_text(json.dumps({'binary_sha256':sha(build/'numi-human-native'),'source_overlay_hashes':{f.name:sha(f) for f in sorted((D/'source-overlay/apps').iterdir()) if f.is_file()},'results':results,'physical_qualification':False},indent=2)+'\n')
