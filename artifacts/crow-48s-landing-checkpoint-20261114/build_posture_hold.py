from pathlib import Path
import hashlib,json,subprocess,sys
name=sys.argv[1];assert name in ('positive','negative')
sign='1.0f' if name=='positive' else '-1.0f'
r=Path('/Users/n/Developer/numi-crow-rl-20261107');a=r/'artifacts/crow-v15-recorded-restart-20261114';b=r/'build/crow-v15-recorded-restart-20261114'
old=(a/'restart-capture-x1745.metal').read_text()
needle='''        const float target = clamp(
            targetCandidate,
            binding.parameters.y,'''
assert old.count(needle)==1
ins='''        // Contact-gated position-drive balance; no injected body wrench.
        if (state.navigation.z>=5.0f && (action==7u || action==10u) &&
            (state.status.w & 63u)>=12u && ((state.status.w>>6u)&63u)>=12u) {
            const float correction=clamp(SIGN*(0.40f*(state.threatGeometry.z-0.10f)+
                                               0.08f*state.threatGeometry.w),-0.12f,0.12f);
            targetCandidate+=correction;
        }
'''.replace('SIGN',sign)
src=a/('restart-posture-'+name+'.metal');src.write_text(old.replace(needle,ins+needle))
build=json.loads((a/'shader-build.json').read_text());cmd=build['compile'];cmd[cmd.index('-c')+1]=str(src)
air=b/('restart-posture-'+name+'.air');lib=b/('restart-posture-'+name+'.metallib');cmd[cmd.index('-o')+1]=str(air);subprocess.run(cmd,check=True)
link=build['link'];link[link.index(str(b/'restart.air'))]=str(air);link[link.index('-o')+1]=str(lib);subprocess.run(link,check=True)
(a/('posture-'+name+'-build.json')).write_text(json.dumps({'sign':name,'compile':cmd,'link':link,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (src,air,lib)}},indent=2)+'\n')
print('built',name)
