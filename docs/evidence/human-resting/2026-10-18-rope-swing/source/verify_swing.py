from pathlib import Path
import sys,json,numpy as np,re,hashlib,gc,csv
sys.path.insert(0,'/Users/n/numi-human-closed-outer-skin-20261017/src')
from numilab_human import resting_anatomy_verify as v
o=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001");run=o/'native-009';rows=[];base=None
local=np.array([-.0545973483,.1411030829,-.00279765767])
for path in sorted((run/'accepted-geometry').glob('*.receipt.json'),key=lambda p:int(p.stem.split('-')[1].split('.')[0])):
 r=json.loads(path.read_text());step=r['accepted_step'];t=r['accepted_time_s']
 pack=path.with_name(path.name.replace('.receipt.json','.mrvpack'));sha=hashlib.file_digest(pack.open('rb'),'sha256').hexdigest();v.validate_capture_receipt(r,sha)
 s=v._pack_sections(pack);vv=np.frombuffer(s[2][0],'<f4').reshape(-1,20);ii=np.frombuffer(s[3][0],'<u4');pp=np.frombuffer(s[4][0],'<u4').reshape(-1,16)
 pr=next(p for p in pp if (int(p[4]),int(p[7]))==(52001,52001));ids=ii[int(pr[0]):int(pr[0])+int(pr[1])];start=int(ids.min());pts=vv[start:start+580,:3].astype(float);anchor=pts[:8].mean(0);endpoint=pts[248:256].mean(0)
 if base is None:base=anchor.copy()
 u=np.clip(t/3,0,1);ramp=u**3*(10+u*(-15+6*u));phase=2*np.pi*t/4;expected=base+.3*ramp*np.array([np.sin(phase),.5*(np.cos(phase)-1),0])
 body=next(b for b in r['accepted_body_poses'] if b['body_index']==20);q=np.array(body['quaternion_xyzw']);q/=np.linalg.norm(q);site=np.array(body['position_m'])+local+2*np.cross(q[:3],np.cross(q[:3],local)+q[3]*local)
 row=dict(step=step,time_s=t,camera_index=r['camera_index'],anchor_m=anchor.tolist(),site_m=site.tolist(),visual_anchor_error_m=float(np.linalg.norm(anchor-expected)),attachment_error_m=float(np.linalg.norm(site-endpoint)),chord_m=float(np.linalg.norm(site-anchor)),fixture_center_error_m=float(np.linalg.norm(pts[416:576].mean(0)-anchor)),pack_sha256=sha)
 rows.append(row);print(json.dumps(row),flush=True);del s,vv,ii,pp,pts;gc.collect()
log=(run/'launcher.log').read_text();complete=re.search(r'resting_integrated_body=completed simulated_s=([\d.e+-]+) wall_s=([\d.e+-]+) real_time_factor=([\d.e+-]+)',log)
with (run/'resting-com-support-impulses.csv').open() as f:contact=list(csv.DictReader(f))
slot=int(re.search(r'resting_rope_requested=true contact_index=(\d+)',log)[1])
other=max(float(r['normal_impulse_ns']) for r in contact if int(r['contact_index'])!=slot)
with (run/'resting-surface-audit.csv').open() as f:audit=list(csv.DictReader(f))
report=dict(status='PASS_DEMONSTRATION' if complete and len(rows)==4 and max(r['visual_anchor_error_m'] for r in rows)<2e-6 and max(r['attachment_error_m'] for r in rows)<2e-6 and other==0 else 'FAIL',captures=rows,completion=None if not complete else dict(simulated_s=float(complete[1]),wall_s=float(complete[2]),real_time_factor=float(complete[3])),other_contact_normal_impulse_ns_max=other,max_zero_area_triangles=max(float(r['mesh_zero_area_triangles']) for r in audit),max_nonfinite_area_triangles=max(float(r['mesh_nonfinite_area_triangles']) for r in audit),scope='Driven rope visual/attachment consistency and completed native path. Position-driven ideal length limit; not distributed rope or full swing-pose anatomical qualification.')
(run/'swing-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='captures'}))
