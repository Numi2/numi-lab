from pathlib import Path
import hashlib,json,math,subprocess
r=Path('/Users/n/Developer/numi-crow-rl-20261107');a=r/'artifacts/crow-v15-recorded-restart-20261114';b=r/'build/crow-v15-recorded-restart-20261114';old=(a/'restart.metal').read_text();needle='''        if (dispatch.counts.x == 1u && dispatch.counts.z == 41u && dispatch.counts.w == 40u && dispatch.strides.w == 6u) {
            for (uint i=0; i<41u; ++i) resetQ[qBase+i]=crowRestartQ[i];''';assert old.count(needle)==1
phase=(2*math.pi*2400*.02*4.6)%(2*math.pi)
replacement='''        if (dispatch.counts.x == 1u && dispatch.counts.z == 41u && dispatch.counts.w == 40u && dispatch.strides.w == 6u) {
            // Partially reconstructed timing and recorded status; history is not restored.
            state.episode.x=2400u; state.status.w=117501952u; state.navigation.z=1.0f;
            state.commandAndPhase.w=PHASEf;
            for (uint i=0; i<41u; ++i) resetQ[qBase+i]=crowRestartQ[i];'''.replace('PHASE',format(phase,'.10g'))
source=a/'restart-phase.metal';source.write_text(old.replace(needle,replacement));build=json.loads((a/'shader-build.json').read_text());compile=build['compile'];compile[compile.index('-c')+1]=str(source);compile[compile.index('-o')+1]=str(b/'restart-phase.air');subprocess.run(compile,check=True)
link=build['link'];link[link.index(str(b/'restart.air'))]=str(b/'restart-phase.air');link[link.index('-o')+1]=str(b/'restart-phase.metallib');subprocess.run(link,check=True)
(a/'phase-shader-build.json').write_text(json.dumps({'phase_radians_inferred':phase,'compile':compile,'link':link,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,b/'restart-phase.air',b/'restart-phase.metallib')}},indent=2)+'\n');print('built partial-clock reconstruction shader',phase)
