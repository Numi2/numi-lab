from pathlib import Path
import hashlib,json,subprocess,time,datetime
B=Path('/Users/n/cardboard-implementation-20261014-host-macmini');P=B/'analysis/contact-scaled-nearest-point-overlay-187-002';S=B/'snapshots/source-portable-paper-reference-185'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not(P/'build-receipt.json').exists()
f=json.loads((S/'frozen-source.json').read_text());assert sha(S/'frozen-source.json')=='4f6d4569ea271ecf604b5e3c1b99299e8fcc9ebefcefcd2e3b171a232ace9cb6'
for rel,h in f['source_sha256'].items(): assert sha(S/rel)==h
commands=[['clang++','-std=c++20','-fobjc-arc','-Wall','-Wextra','-Werror',str(P/'geometry_replay.mm'),'-framework','Foundation','-framework','Metal','-o',str(P/'geometry-replay')],['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-MMD','-MF',str(P/'metal-dependencies.d'),'-I',str(S/'matter/src'),'-I',str(S/'matter/include'),'-I',str(S/'include'),'-c',str(P/'geometry_replay.metal'),'-o',str(P/'geometry-replay.air')],['xcrun','-sdk','macosx','metallib',str(P/'geometry-replay.air'),'-o',str(P/'geometry-replay.metallib')]]
runs=[]
for i,cmd in enumerate(commands):
 t=time.monotonic();r=subprocess.run(cmd,capture_output=True,timeout=120);(P/f'build-{i}.stdout').write_bytes(r.stdout);(P/f'build-{i}.stderr').write_bytes(r.stderr);runs.append({'argv':cmd,'returncode':r.returncode,'seconds':time.monotonic()-t});assert r.returncode==0,r.stderr.decode()
report={'commands':runs,'source185_frozen_sha256':sha(S/'frozen-source.json'),'candidate_functions_sha256':sha(P/'nearest-functions.metalinc'),'host_sha256':sha(P/'geometry-replay'),'kernel_sha256':sha(P/'geometry_replay.metal'),'metallib_sha256':sha(P/'geometry-replay.metallib'),'air_sha256':sha(P/'geometry-replay.air'),'dependency_file_sha256':sha(P/'metal-dependencies.d'),'gpu_dispatch':False,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(P/'build-receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
