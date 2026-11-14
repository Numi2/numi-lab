from pathlib import Path
import hashlib,json,subprocess,sys
name=sys.argv[1];assert name in ('aligned','opposed')
z='0.0998334166f' if name=='aligned' else '-0.0998334166f'
r=Path('/Users/n/Developer/numi-crow-rl-20261107');a=r/'artifacts/crow-v15-recorded-restart-20261114';b=r/'build/crow-v15-recorded-restart-20261114'
old=(a/'restart-capture-x1730.metal').read_text();needle='resetScene[sceneBase+5u].position.xyz=float3(17.73f,0.46f,0.20f);'
assert old.count(needle)==1
src=a/('restart-perch-yaw-'+name+'.metal')
src.write_text(old.replace(needle,needle+'\n            resetScene[sceneBase+5u].orientation=float4(0.0f,0.0f,'+z+',0.9950041653f);'))
build=json.loads((a/'shader-build.json').read_text());cmd=build['compile'];cmd[cmd.index('-c')+1]=str(src)
air=b/('restart-perch-yaw-'+name+'.air');lib=b/('restart-perch-yaw-'+name+'.metallib');cmd[cmd.index('-o')+1]=str(air);subprocess.run(cmd,check=True)
link=build['link'];link[link.index(str(b/'restart.air'))]=str(air);link[link.index('-o')+1]=str(lib);subprocess.run(link,check=True)
(a/('perch-yaw-'+name+'-build.json')).write_text(json.dumps({'perch_yaw_rad':.2 if name=='aligned' else -.2,'compile':cmd,'link':link,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (src,air,lib)}},indent=2)+'\n')
print('built',name)
