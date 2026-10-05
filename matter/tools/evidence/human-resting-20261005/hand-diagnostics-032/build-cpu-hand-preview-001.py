from pathlib import Path
import subprocess,shlex,json,hashlib
E=Path('/Users/n/numi-human-resting-evidence-20261005')
B=Path('/Users/n/numi-human-resting-hand-reduction-build-018')
d=E/'rigid-digits-cpu-preview-001';d.mkdir(exist_ok=False)
flags={}
for line in (B/'CMakeFiles/numi-human-native.dir/flags.make').read_text().splitlines():
    if ' = ' in line:
        k,v=line.split(' = ',1);flags[k]=shlex.split(v)
source=E/'preview-rigid-digits-cpu-001.mm';obj=d/'preview.o';binary=d/'preview'
compile=['/usr/bin/clang++',*flags['OBJCXX_DEFINES'],*flags['OBJCXX_INCLUDES'],*flags['OBJCXX_FLAGS'],'-c',str(source),'-o',str(obj)]
link=shlex.split((B/'CMakeFiles/numi-human-native.dir/link.txt').read_text())
link[link.index('CMakeFiles/numi-human-native.dir/apps/numilab_human_myosim_visual_probe.mm.o')]=str(obj)
link[link.index('-o')+1]=str(binary)
(d/'build-invocation.json').write_text(json.dumps({'compile':compile,'link':link,'cwd':str(B),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},indent=2)+'\n')
with (d/'build.log').open('w') as log:
    subprocess.run(compile,cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run(link,cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
print(hashlib.sha256(binary.read_bytes()).hexdigest())
