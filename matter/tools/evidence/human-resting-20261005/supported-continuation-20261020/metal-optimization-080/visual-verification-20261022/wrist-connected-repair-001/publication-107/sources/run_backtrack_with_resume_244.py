"""Bounded audit continuation; preserves frozen exact geometry and selection owners."""
from pathlib import Path
import json,subprocess,sys,hashlib,shutil
H=Path('/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170')
PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def continuation_scale_path(out):
 passes=sorted(Path(out).glob('pass-*/result.json'))
 if not passes or read(passes[-1]).get('status')!='EXACT_AUDIT_BUDGET_REACHED':return None
 checked=[read(p)for p in passes if isinstance(read(p).get('exact_self_pairs'),int)]
 if len(checked)<2 or not 0<checked[-1]['exact_self_pairs']<=16:return None
 if checked[-1]['exact_self_pairs']>=checked[-2]['exact_self_pairs']:return None
 scales=passes[-1].parent/'node-step-scales.npy'
 return scales if scales.is_file()else None
def main():
 cfgpath=Path(sys.argv[1]);cfg=read(cfgpath);out=Path(cfg['out'])
 original=H/'run_local_witness_backtracking_211.py'
 cmd=[PY,str(original),str(cfgpath)]
 subprocess.run(cmd,check=True)
 if(out/'selected-result.json').exists():return
 scales=continuation_scale_path(out)
 if scales is None:return
 # Exactly one further existing eight-audit block, using existing saved local scales.
 nxt={**cfg,'out':str(out/'bounded-audit-continuation'),'initial_scales':str(scales)}
 np=out/'bounded-audit-continuation-config.json';np.write_text(json.dumps(nxt,indent=2)+'\n')
 receipt=dict(original_runner_sha256=sha(original),original_config_sha256=sha(cfgpath),resume_scales_sha256=sha(scales),maximum_additional_exact_audits=8,native_admission=False,accepted_physical_steps=0)
 (out/'bounded-audit-continuation-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 subprocess.run([PY,str(original),str(np)],check=True)
 selected=Path(nxt['out'])/'selected-result.json'
 if selected.exists():shutil.copy2(selected,out/'selected-result.json')
if __name__=='__main__':main()
