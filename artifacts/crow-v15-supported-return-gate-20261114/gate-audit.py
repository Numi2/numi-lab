from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-supported-return-gate-20261114');P=Path('artifacts/crow-v15-pretransfer-pitch-feedback-20261114');records={}; gates=[];prior={}
def read(path,out,gate=False):
 with gzip.open(path,'rt') as f:
  for line in f:
   z=line.split()
   if not z:continue
   if gate and z[0]=='returnGate':gates.append({'lane':int(z[1]),'t':int(z[2]),'recovery':int(z[3]),'allowed':bool(int(z[4]))})
   if z[0] in ('step','control','event','hipcommand','jointtarget','rearKnee','rearAnkle','rightHipCommand','balance','pitchcontrol','stage'):
    key=(int(z[1]),int(z[2]),z[0]);out.setdefault(key,[]).append(z[3:])
read(A/'trace.log.gz',records,True);read(P/'trace.log.gz',prior)
state_diff=[];command_diff=[];control_diff=[]
for t in range(450):
 if records.get((0,t,'step'))!=records.get((1,t,'step')):state_diff.append(t)
 if any(records.get((0,t,k))!=records.get((1,t,k)) for k in ('hipcommand','jointtarget','rearKnee','rearAnkle','rightHipCommand')):command_diff.append(t)
 if any(records.get((0,t,k))!=prior.get((0,t,k)) for k in ('step','control','event','hipcommand','jointtarget','rearKnee','rearAnkle','rightHipCommand','balance','pitchcontrol','stage')):control_diff.append(t)
closed=[x['t'] for x in gates if x['lane']==1 and not x['allowed']]
result={'control_vs_prior_first_different_interval':next(iter(control_diff),None),'matched_state_first_different_interval':next(iter(state_diff),None),'matched_logged_leg_targets_first_different_interval':next(iter(command_diff),None),'candidate_gate_first_closed_interval':next(iter(closed),None),'candidate_gate_closed_interval_count':len(closed),'candidate_gate_closed_intervals':closed,'gate_record_count':len(gates),'scope':'Exact logged values, zero-based control loop. A closed gate does not necessarily change a target if the corresponding ramp is already stopped.'}
(A/'gate-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
