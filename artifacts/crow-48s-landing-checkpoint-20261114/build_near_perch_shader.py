from pathlib import Path
import hashlib,json,subprocess
r=Path('/Users/n/Developer/numi-crow-rl-20261107');a=r/'artifacts/crow-v15-recorded-restart-20261114';b=r/'build/crow-v15-recorded-restart-20261114';old=(a/'restart-phase.metal').read_text()
needle='''                s.angularVelocity.xyz=float3(crowRestartScene[j+10u],crowRestartScene[j+11u],crowRestartScene[j+12u]);
            }
        }
        device float* firstActor ='''
assert old.count(needle)==1
new=needle.replace('''            }
        }
        device float* firstActor =''','''            }
            // Constructed development fixture: original perch moved before restart.
            resetScene[sceneBase+5u].position.xyz=float3(17.76f,0.46f,0.20f);
        }
        device float* firstActor =''')
source=a/'restart-near-perch.metal';source.write_text(old.replace(needle,new));build=json.loads((a/'shader-build.json').read_text());cmd=build['compile'];cmd[cmd.index('-c')+1]=str(source);cmd[cmd.index('-o')+1]=str(b/'restart-near-perch.air');subprocess.run(cmd,check=True)
link=build['link'];link[link.index(str(b/'restart.air'))]=str(b/'restart-near-perch.air');link[link.index('-o')+1]=str(b/'restart-near-perch.metallib');subprocess.run(link,check=True)
(a/'near-perch-shader-build.json').write_text(json.dumps({'perch_center_m':[17.76,.46,.20],'compile':cmd,'link':link,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,b/'restart-near-perch.air',b/'restart-near-perch.metallib')}},indent=2)+'\n');print('built nearby-perch diagnostic shader')
