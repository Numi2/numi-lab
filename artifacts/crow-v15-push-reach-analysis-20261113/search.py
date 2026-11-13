from pathlib import Path
import hashlib,json,math,os,subprocess
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-push-reach-analysis-20261113';G=R/'artifacts/crow-v15-drive-target-audit-20261113/geometry.json';d=json.loads(G.read_text());q=d['q'];assert len(q)==41
exe=R/'build/crow-v15-hip-roll-walk-launch-matched-20261111/kinematics';lib=R/'build/crow-v15-hip-roll-journey-integration-20261111/libmetalrobo.dylib';n=13
vals=lambda lo,hi:[lo+(hi-lo)*i/(n-1) for i in range(n)]
variants=[('baseline',q[14],q[15],q[16])]+[('grid',h,k,a) for h in vals(-.82,.58) for k in vals(-.1,1.03) for a in vals(-.81,.49)]
lines=[]
for i,(_,h,k,a) in enumerate(variants):
 p=q.copy();p[14]=h;p[15]=k;p[16]=a;p[17]=h;p[18]=k;p[19]=a
 lines.append('5 '+str(i)+' '+' '.join(map(str,p+[0.]*40)))
r=subprocess.run([str(exe)],input='\n'.join(lines)+'\n',text=True,capture_output=True,check=True,env={**os.environ,'DYLD_LIBRARY_PATH':str(lib.parent)});samples={i:{} for i in range(len(variants))}
for line in r.stdout.splitlines():
 z=line.split();e,i=map(int,z[1:3]);assert e==5
 if z[0]=='com':samples[i]['com']=list(map(float,z[4:7]))
 elif z[0]=='body' and z[3] in ('crow_left_foot','crow_right_foot'):samples[i][z[3]]=list(map(float,z[4:7]))
assert all(set(x)=={'com','crow_left_foot','crow_right_foot'} for x in samples.values());base=samples[0];rows=[]
for i,(_,h,k,a) in enumerate(variants[1:],1):
 v=samples[i];dxyz={name:[(v[name][j]-v['com'][j])-(base[name][j]-base['com'][j]) for j in range(3)] for name in ('crow_left_foot','crow_right_foot')};height=-.5*(dxyz['crow_left_foot'][2]+dxyz['crow_right_foot'][2]);x=.5*(dxyz['crow_left_foot'][0]+dxyz['crow_right_foot'][0]);rot=(h-q[14])+(k-q[15])+(a-q[16]);rows.append({'hip':h,'knee':k,'ankle':a,'estimated_com_rise_with_fixed_foot_m':height,'mean_foot_com_horizontal_change_m':x,'joint_angle_sum_change_rad':rot})
filters={'horizontal_10mm_angle_025rad':[r for r in rows if abs(r['mean_foot_com_horizontal_change_m'])<=.01 and abs(r['joint_angle_sum_change_rad'])<=.25],'horizontal_20mm_angle_05rad':[r for r in rows if abs(r['mean_foot_com_horizontal_change_m'])<=.02 and abs(r['joint_angle_sum_change_rad'])<=.5],'unconstrained':rows};best={key:sorted(v,key=lambda z:z['estimated_com_rise_with_fixed_foot_m'],reverse=True)[:10] for key,v in filters.items()};out={'grid_per_joint':n,'samples':len(rows),'baseline_q':q,'best':best,'scope':'Posthoc fixed-root symmetric v15 FK grid, held world body orientation, one used-seed posture. Foot COM approximates foot position, angle-sum is only a rough orientation proxy. Inferred COM rise assumes both feet fixed and vertical translation; no contact, momentum, forces, limits on speed, or dynamic feasibility are solved. Grid is not biological or takeoff qualification.','fk_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'lib_sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),'posture_sha256':hashlib.sha256(G.read_bytes()).hexdigest()};(A/'search.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:(len(filters[k]),best[k][:1]) for k in filters},indent=2))
