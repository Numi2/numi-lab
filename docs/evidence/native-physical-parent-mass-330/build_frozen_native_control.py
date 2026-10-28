from pathlib import Path
import subprocess,shlex,gzip,json,time,hashlib
P=Path(__file__).parent
W=Path('/Users/numi/Work/skin-fat-native-integration-209')
B=W/'build-native'
S=P/'frozen-native-control';S.mkdir()
old=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-prepared-identity-replacement-326/attempts/009-parent-reference-identity-control-root/source/numilab_human_myosim_visual_probe.mm.gz')
source=S/'numilab_human_myosim_visual_probe.mm';source.write_bytes(gzip.decompress(old.read_bytes()))
lines=subprocess.run(['/opt/homebrew/bin/ninja','-t','commands','numi-human-native'],cwd=B,text=True,capture_output=True,check=True).stdout.splitlines()
compile=next(shlex.split(x) for x in lines if 'numilab_human_myosim_visual_probe.mm' in x and ' -c ' in x)
oldobj=compile[compile.index('-o')+1]
compile[compile.index('-o')+1]=str(S/'probe.o')
compile[compile.index('-MF')+1]=str(S/'probe.o.d')
compile[compile.index('-MT')+1]=str(S/'probe.o')
compile[compile.index('-c')+1]=str(source)
compile+=['-iquote',str(W/'apps')]
raw=next(shlex.split(x) for x in lines if 'numilab_human_myosim_visual_probe.mm' in x and ' -c ' not in x)
link=raw[raw.index('&&')+1:];link=link[:link.index('&&')]
link=[str(S/'probe.o') if x==oldobj else x for x in link]
link[link.index('-o')+1]=str(S/'numi-human-native')
manifest={'compile':compile,'link':link,'frozen_probe_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'build_cwd':str(B)}
(S/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for name,argv in [('compile',compile),('link',link)]:
 t=time.perf_counter();r=subprocess.run(argv,cwd=B,capture_output=True)
 (S/(name+'.stdout.txt')).write_bytes(r.stdout);(S/(name+'.stderr.txt')).write_bytes(r.stderr)
 result={'exit_code':r.returncode,'wall_s':time.perf_counter()-t}
 (S/(name+'.result.json')).write_text(json.dumps(result,indent=2)+'\n');print(name,result,flush=True)
 if r.returncode:print(r.stderr.decode(errors='replace')[-4000:]);raise SystemExit(r.returncode)
print('binary_sha256',hashlib.sha256((S/'numi-human-native').read_bytes()).hexdigest())
