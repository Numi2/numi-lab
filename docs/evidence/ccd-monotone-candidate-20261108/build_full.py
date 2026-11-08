import json,pathlib,shlex,subprocess
p=pathlib.Path.cwd();build=pathlib.Path('/Users/numi/numi-lab-ccd-cold-four-sliding-build-20261108');source=pathlib.Path('/Users/numi/numi-lab-policy-repair-main-20261108')
cmd=['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(source/'include'),'-c',str(p/'Collision.candidate.metal'),'-o',str(p/'Collision.candidate.air')]
raw=subprocess.check_output(['ninja','-t','commands','shaders/MetalRobo.metallib'],cwd=build,text=True).strip().splitlines()[-1].split(' && ')
link=shlex.split(raw[-1]);old=str(build/'shaders/MetalRoboCollision.air');assert link.count(old)==1;link[link.index(old)]=str(p/'Collision.candidate.air');link[link.index('-o')+1]=str(p/'candidate.metallib')
(p/'full-build-commands.json').write_text(json.dumps([cmd,link],indent=2)+'\n')
with (p/'full-build.stdout').open('w') as out,(p/'full-build.stderr').open('w') as err:
 for c in [cmd,link]:subprocess.run(c,stdout=out,stderr=err,check=True)
