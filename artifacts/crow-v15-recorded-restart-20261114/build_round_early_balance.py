from pathlib import Path
import hashlib,json,subprocess,sys
name='early-negative';sign='-1.0f'
r=Path('/Users/n/Developer/numi-crow-rl-20261107');a=r/'artifacts/crow-v15-recorded-restart-20261114';b=r/'build/crow-v15-recorded-restart-20261114'
old=(a/'restart-round-forward-front-closure.metal').read_text()
needle='''        const float target = clamp(
            targetCandidate,
            binding.parameters.y,''';assert old.count(needle)==1
ins='''        // Delayed, contact-gated position-drive balance only.
        if (state.navigation.z>=5.0f && state.episode.x>=2450u &&
            (action==7u || action==10u) && (state.status.w & 63u)>=12u &&
            ((state.status.w>>6u)&63u)>=12u) {
            targetCandidate+=clamp(SIGN*(0.50f*(state.threatGeometry.z-0.10f)+
                                         0.08f*state.threatGeometry.w),-0.16f,0.16f);
        }
'''.replace('SIGN',sign)
src=a/('restart-round-balance-'+name+'.metal');src.write_text(old.replace(needle,ins+needle))
build=json.loads((a/'shader-build.json').read_text());cmd=build['compile'];cmd[cmd.index('-c')+1]=str(src)
air=b/('restart-round-balance-'+name+'.air');lib=b/('restart-round-balance-'+name+'.metallib');cmd[cmd.index('-o')+1]=str(air);subprocess.run(cmd,check=True)
link=build['link'];link[link.index(str(b/'restart.air'))]=str(air);link[link.index('-o')+1]=str(lib);subprocess.run(link,check=True)
(a/('round-balance-'+name+'-build.json')).write_text(json.dumps({'sign':name,'compile':cmd,'link':link,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (src,air,lib)}},indent=2)+'\n');print('built',name)
