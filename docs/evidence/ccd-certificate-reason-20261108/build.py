import json, pathlib, shlex, subprocess
p=pathlib.Path.cwd(); build=pathlib.Path('/Users/numi/numi-lab-ccd-cold-four-sliding-build-20261108'); source=pathlib.Path('/Users/numi/numi-lab-ccd-cold-four-sliding-20261108')
cmd=['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(source/'include'),'-c',str(p/'Collision.instrumented.metal'),'-o',str(p/'Collision.instrumented.air')]
commands=subprocess.check_output(['ninja','-t','commands','shaders/MetalRobo.metallib'],cwd=build,text=True).strip().splitlines()[-1].split(' && ')
link=shlex.split(commands[-1]); assert link[:4]==['xcrun','-sdk','macosx','metallib']
old=str(build/'shaders/MetalRoboCollision.air'); assert link.count(old)==1
link[link.index(old)]=str(p/'Collision.instrumented.air'); link[link.index('-o')+1]=str(p/'instrumented.metallib')
(p/'build-commands.json').write_text(json.dumps([cmd,link],indent=2)+'\n')
with (p/'build.stdout').open('w') as out, (p/'build.stderr').open('w') as err:
 for x in [cmd,link]: subprocess.run(x,check=True,stdout=out,stderr=err)
