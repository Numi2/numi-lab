from pathlib import Path
import hashlib,json
A=Path('artifacts/crow-v15-aero-yaw-sensitivity-20261112');P=Path('artifacts/crow-v15-aerodynamic-load-replay-20261112/output.txt')
def parse(path):
 out={}
 for l in path.read_text().splitlines():
  z=l.split();key=tuple(map(int,z[:3]));vals=list(map(float,z[3:]));assert len(vals)==9 and key not in out;out[key]=vals
 return out
base=parse(P);out=parse(A/'output.txt');meta=json.loads((A/'input-metadata.json').read_text());delta=meta['delta_rad'];coverage=meta['counts'];expected=coverage['baseline']+2*(coverage['sweep_pairs']+coverage['pronation_pairs']);assert len(out)==expected
diff=max(abs(a-b) for key,x in out.items() if key[0]==0 for a,b in zip(x,base[key]));assert diff<1e-7,diff
def stat(v):
 v=sorted(v);return {'count':len(v),'min':v[0],'median':v[(len(v)-1)//2],'max':v[-1],'mean':sum(v)/len(v)}
rows={}
for name,plus,minus in [('sweep',1,2),('pronation',3,4)]:
 vals=[];delta_yaw=[]
 for key in sorted(k for k in out if k[0]==plus):
  _,e,t=key;p=out[key];m=out[minus,e,t];z=(p[3]+p[7])-(m[3]+m[7]);delta_yaw.append(z);vals.append(z/(2*delta))
 assert len(vals)==coverage[name+'_pairs']
 rows[name]={'central_wing_yaw_moment_difference_Nm':stat(delta_yaw),'central_yaw_sensitivity_Nm_per_rad':stat(vals),'absolute_sensitivity_Nm_per_rad':stat([abs(v) for v in vals])}
result={'baseline_comparison_max_component_abs_difference':diff,'repeated_output_identical':(A/'output.txt').read_bytes()==(A/'repeat.txt').read_bytes(),'sample_coverage':coverage,'source_output_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'rows':rows,'scope':'Counterfactual pose perturbations at 72 sparse endpoints in previously used simulated flights. Instantaneous wing contributions only; no tail or full net root moment, no integrated impulse, actual dynamics or crow aerodynamic validation.'}
(A/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
