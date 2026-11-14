from pathlib import Path
import hashlib,json,subprocess,sys
name=sys.argv[1];assert name in ('x1755','x1765');x='17.755f' if name=='x1755' else '17.765f'
r=Path('/Users/n/Developer/numi-crow-rl-20261107');a=r/'artifacts/crow-v15-recorded-restart-20261114';b=r/'build/crow-v15-recorded-restart-20261114'
old=(a/'restart-round-perch.metal').read_text();needle='float3(17.745f,0.46f,0.23f)';assert old.count(needle)==1
src=a/('restart-round-'+name+'.metal');src.write_text(old.replace(needle,'float3('+x+',0.46f,0.23f)'))
build=json.loads((a/'shader-build.json').read_text());cmd=build['compile'];cmd[cmd.index('-c')+1]=str(src)
air=b/('restart-round-'+name+'.air');lib=b/('restart-round-'+name+'.metallib');cmd[cmd.index('-o')+1]=str(air);subprocess.run(cmd,check=True)
link=build['link'];link[link.index(str(b/'restart.air'))]=str(air);link[link.index('-o')+1]=str(lib);subprocess.run(link,check=True)
(a/('round-'+name+'-build.json')).write_text(json.dumps({'perch_x':x,'compile':cmd,'link':link,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (src,air,lib)}},indent=2)+'\n');print('built',name)
