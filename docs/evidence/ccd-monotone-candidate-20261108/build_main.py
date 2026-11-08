import json,pathlib,shlex,subprocess
p=pathlib.Path.cwd();build=pathlib.Path('/Users/numi/numi-lab-horizon-main-build-20261108')
raw=subprocess.check_output(['ninja','-t','commands','shaders/MetalRobo.metallib'],cwd=build,text=True).strip().splitlines()[-1].split(' && ')
link=shlex.split(raw[-1]);old=str(build/'shaders/MetalRoboCollision.air');assert link.count(old)==1;link[link.index(old)]=str(p/'Collision.candidate.air');link[link.index('-o')+1]=str(p/'main-response.metallib')
(p/'main-build-command.json').write_text(json.dumps(link,indent=2)+'\n')
with (p/'main-build.stdout').open('w') as out,(p/'main-build.stderr').open('w') as err:subprocess.run(link,stdout=out,stderr=err,check=True)
