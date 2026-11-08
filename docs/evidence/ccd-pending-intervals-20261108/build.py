import hashlib,json,pathlib,shlex,subprocess
p=pathlib.Path.cwd();build=pathlib.Path('/Users/numi/numi-lab-ccd-cold-four-sliding-build-20261108');source=pathlib.Path('/Users/numi/numi-lab-ccd-cold-four-sliding-20261108');runtime='/Users/numi/numi-lab-horizon-main-build-20261108/lib'
probe=['clang++','-std=c++20','-O2','-I',str(source/'include'),str(p/'probe.cpp'),'-L',runtime,'-lmetalrobo','-Wl,-rpath,'+runtime,'-o',str(p/'probe')]
raw=subprocess.check_output(['ninja','-t','commands','shaders/MetalRobo.metallib'],cwd=build,text=True).strip().splitlines()[-1].split(' && ')
base=shlex.split(raw[-1]);assert base[:4]==['xcrun','-sdk','macosx','metallib'];old=str(build/'shaders/MetalRoboCollision.air');assert base.count(old)==1
commands=[probe]
for i in range(13):
 air=str(p/f'collision-{i}.air');out=str(p/f'instrumented-{i}.metallib')
 commands.append(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-D',f'CCD_CAPTURE_SLOT={i}','-I',str(source/'include'),'-c',str(p/'Collision.instrumented.metal'),'-o',air])
 link=base.copy();link[link.index(old)]=air;link[link.index('-o')+1]=out;commands.append(link)
(p/'build-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
with (p/'build.stdout').open('w') as stdout,(p/'build.stderr').open('w') as stderr:
 for i,c in enumerate(commands):
  print('building command',i,flush=True)
  subprocess.run(c,stdout=stdout,stderr=stderr,check=True)
with (p/'air-inputs.sha256').open('w') as out:
 for s in base:
  if s.endswith('.air') and s!=old:
   f=pathlib.Path(s);out.write(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+s+'\n')
