from pathlib import Path
import subprocess,sys,time,json
r=Path('/Users/home/cardboard-followup-20261013');repo=Path('/Users/home/numi-cardboard-20261006');frozen=repo/'build-cardboard-frozen035';source=r/'solver/frozen035-source'
while not (r/'solver/operator-fd-scope-034/receipt.json').exists():time.sleep(2)
def run(case, question, prediction, args, envs=(), runtime=frozen):
 cmd=[sys.executable,str(r/'run_bound.py'),'--case',str(r/case),'--question',question,'--prediction',prediction,'--timeout','1200','--source-snapshot',str(source),'--runtime-dir',str(runtime)]
 for e in envs: cmd += ['--env',e]
 cmd += ['--',*args]
 subprocess.run(cmd,check=True)
 receipt=json.loads((r/case/'receipt.json').read_text());print(case,receipt,flush=True);return receipt
base=[str(frozen/'numi-matter-cardboard-probe'),'--without-self-contact','--reference-displacement-gradient','--residual-tolerance','1e-6','--newton-iterations','28','--fgmres-iterations','1024','--steps','48','--hold-steps','12','--unload-steps','24','--relax-steps','12','--nx','16','--bend-angle','.375','--dt','.000025','--metallib',str(frozen/'shaders/NumiMatter.metallib')]
run('solver/local-projection-accuracy-035','Does tenfold tighter local generic projection accuracy reduce the selected global operator/residual disagreement or advance the unchanged 96-step nx16 protocol?','Isolated alternate Metal source changes both local projection factors2e-5to2e-6 only, same16localiterations, physical params and global1e-6 certificate/Newton28/FG1024. Predict either more accurately resolved global response or an explicitly retained local projection failure; do not assume success, mesh convergence, or a specific cause. Requested root19:3 may not be reached or may refer to a different candidate under the altered accepted history; report this rather than substituting another point. All failures/timeout retained. No corrugated calibration or physical qualification.',base+['--output',str(r/'solver/local-projection-accuracy-035/output')],['NM_FEM_RESIDUAL_ARMIJO_ROOT=all','NM_FEM_OPERATOR_FD_SELECTOR=19:3','NM_FEM_OPERATOR_FD_EPSILON_SCALE=8'])
