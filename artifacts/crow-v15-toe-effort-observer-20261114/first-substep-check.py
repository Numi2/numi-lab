from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-toe-effort-observer-20261114');states={};cmd={};eff={}
for line in gzip.open(A/'trace.log.gz','rt'):
 z=line.split()
 if len(z)<4 or z[1]!='0':continue
 t=int(z[2])
 if z[0]=='step':
  i=z.index('q');j=z.index('v');states[t]=(list(map(float,z[i+1:j])),list(map(float,z[j+1:])))
 elif z[0]=='control':cmd[t]=float(z[7])
for line in gzip.open(A/'substeps.log.gz','rt'):
 z=line.split()
 if z and z[0]=='CROW_TOE_DRIVE_EFFORT' and z[2]=='0' and z[3]=='0':eff[int(z[1])]=list(map(float,z[4:]))
rows=[]
for t in range(2,451):
 q,v=states[t-2];target=cmd[t-1];raw=.1*(target-q[31]-(.02/32)*v[30])-.003*v[30];pred=max(-.02,min(.02,raw));actual=eff[t][8];rows.append({'trace_step':t,'predicted_Nm':pred,'observed_Nm':actual,'abs_error_Nm':abs(pred-actual),'nominal_saturated':abs(raw)>=.02})
unsaturated=[r for r in rows if not r['nominal_saturated']]
out={'coordinate':'right_front_toe_1_proximal v30 q31','formula':'clamp(.1*(target-q-(.02/32)*v)-.003*v, -.02, .02)','samples':len(rows),'unsaturated_samples':len(unsaturated),'max_abs_error_Nm':max(r['abs_error_Nm'] for r in rows),'max_unsaturated_abs_error_Nm':max(r['abs_error_Nm'] for r in unsaturated),'examples':[r for r in rows if r['trace_step'] in (270,290,300,302,303,304,305,306,310)],'limitations':'Compares the first substep of each control interval only; this validates the effort against the preceding accepted state under the nominal cap but does not align contact observer epochs or validate later substeps.'}
(A/'first-substep-check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
