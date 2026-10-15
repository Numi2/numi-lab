from pathlib import Path
import argparse,hashlib,json
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('native');ap.add_argument('output');a=ap.parse_args();b=Path('/Users/home/cardboard-implementation-20261014');native=Path(a.native);out=Path(a.output);out.mkdir(exist_ok=False);rp=b/'analysis/retained-paper-allpoint-reference-296-001/reference.npz';ref=np.load(rp);h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();keys={tuple(k):i for i,k in enumerate(ref['keys'])};allpolicies={}
for policy in ('default','accurate'):
 path=native/'output'/f'paper-points-{policy}.jsonl';raw=[json.loads(l) for l in path.read_text().splitlines()];header=raw[0];rows=raw[1:];assert len(rows)==len(keys)
 seen=set();valid=[];invalid=[];zero=[]
 for row in rows:
  k=(row['environment'],row['tet_id'],row['material_index']);assert k in keys and k not in seen;seen.add(k);i=keys[k]
  if not(row['stress_valid'] and row['tangent_valid'] and row['tangent_projection_converged']):invalid.append(dict(key=list(k),failure=row['projection_failure']));continue
  metrics={}
  for label,names,rv in [('stress',('direct_stress_pa',),ref['stress'][i]),('tangent',('tangent_pa_per_s',),ref['tangent'][i]),('state',('projected_state_high','projected_state_low'),ref['projected'][i])]:
   nv=sum((np.asarray(row[n],dtype=np.float32).astype(float) for n in names)).ravel();rv=rv.ravel();diff=nv-rv;rn=float(np.linalg.norm(rv));nn=float(np.linalg.norm(nv));dn=float(np.linalg.norm(diff));metrics[label]=dict(reference_norm=rn,native_norm=nn,absolute_l2=dn,max_absolute=float(np.max(abs(diff))),relative_l2=None if rn==0 else dn/rn)
  rec=dict(key=list(k),metrics=metrics);valid.append(rec)
  if not np.any(ref['direction'][i]):zero.append(list(k))
 def stats(label):
  nonzero=[r for r in valid if r['metrics'][label]['reference_norm']>0];ranked=sorted(nonzero,key=lambda r:r['metrics'][label]['relative_l2'],reverse=True)
  return dict(valid_nonzero_reference_rows=len(nonzero),exact_zero_reference_rows=len(valid)-len(nonzero),rows_above_1e_minus_3=sum(r['metrics'][label]['relative_l2']>1e-3 for r in nonzero),worst_rows=ranked[:12])
 allpolicies[policy]=dict(input_sha256=h(path),header=header,valid_rows=len(valid),invalid_rows=invalid,zero_direction_rows=len(zero),stress=stats('stress'),tangent=stats('tangent'),state=stats('state'),per_row=valid)
r=dict(schema='numi.cardboard.allpoint-fp64-comparison.v1',native=str(native),reference_sha256=h(rp),policies=allpolicies,classification='One captured direction per material point; scalar norm comparison against independent FP64 complete implicit response. Not componentwise derivatives, perturbation/noise-floor qualification, element/assembly tests, or complete trajectory. Exact-zero reference rows are not relative-error gates. Failed native rows are excluded from numerical metrics and explicitly remain failures.',reference_warning='Existing FP64 oracle evaluates both numpy.where arms and emitted unused-arm divide-by-zero/invalid warnings; returned arrays were all finite. Canonical oracle retained unchanged.')
(out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');(out/'compare.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({p:{'valid_rows':v['valid_rows'],'invalid_rows':v['invalid_rows'],'stress_nonpass':v['stress']['rows_above_1e_minus_3'],'tangent_nonpass':v['tangent']['rows_above_1e_minus_3'],'worst_stress':v['stress']['worst_rows'][0]['metrics']['stress']['relative_l2'],'worst_tangent':v['tangent']['worst_rows'][0]['metrics']['tangent']['relative_l2']} for p,v in allpolicies.items()}))
