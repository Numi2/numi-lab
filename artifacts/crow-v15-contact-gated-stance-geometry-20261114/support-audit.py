from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-contact-gated-stance-geometry-20261114');P=Path('artifacts/crow-v15-pitch-normal-support-20261114');record={};prior={};gates={}
def read(path,out,withgates=False):
 for line in gzip.open(path,'rt'):
  z=line.split()
  if not z:continue
  if withgates and z[0]=='pitchSupport':gates[int(z[1]),int(z[2])]=int(z[3])
  if z[0] in ('step','control','event','hipcommand','jointtarget','rearKnee','rearAnkle','rightHipCommand','balance','pitchcontrol','stage'):
   out.setdefault((int(z[1]),int(z[2]),z[0]),[]).append(z[3:])
read(A/'trace.log.gz',record,True);read(P/'trace.log.gz',prior)
prior_diff=[t for t in range(450) if any(record.get((0,t,k))!=prior.get((0,t,k)) for k in ('step','control','event','hipcommand','jointtarget','rearKnee','rearAnkle','rightHipCommand','balance','pitchcontrol','stage'))]
gate_diff=[t for t in range(450) if gates.get((0,t))!=gates.get((1,t))]
out={'control_vs_prior_first_difference':next(iter(prior_diff),None),'first_gate_difference_zero_based':next(iter(gate_diff),None),'gate_difference_intervals':gate_diff,'gate_records':len(gates),'scope':'Endpoint normal-only contacts; gate difference is not evidence of successful supported grip.'}
(A/'support-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
