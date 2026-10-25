from pathlib import Path
import numpy as np,json,hashlib,sys,importlib.util,subprocess,time,socket
D=Path("/Users/numi/Work/numi-human-connected-surface-069");E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");P=D/'connected-layer-remesh-116/run-001'
assert not P.exists();P.mkdir(parents=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
base=D/'whole-reference-tetgen-084-bounded/whole-skin-fat-reference.poly'
proposal=D/'connected-capacity-reference-115/candidate.npz'
assert sha(proposal)=='abff4290774cc96a17f6c68a21fe86fc3933a1878e0bd6512733e99589fb6dd9'
z=np.load(proposal);R=z['proposed_reference_points_m'][:7233]
lines=[x.strip()for x in base.read_text().splitlines()if x.strip()and not x.lstrip().startswith('#')]
assert lines[0].split()==['7233','3','0','1']
nr=7233;fstart=1+nr;nf=int(lines[fstart].split()[0]);assert len(lines[fstart].split())==2
faces=[];marks=[];i=fstart+1
for f in range(nf):
 h=lines[i].split();assert len(h)==3 and h[:2]==['1','0'];marks.append(int(h[2]));i+=1
 a=list(map(int,lines[i].split()));assert len(a)==4 and a[0]==3;faces.append([v-1 for v in a[1:]]);i+=1
tail=lines[i:];assert tail[0]=='0' and tail[1]=='2' and len(tail)==4
F=np.asarray(faces,np.int64);marks=np.asarray(marks,np.int64)
poly=P/'whole-skin-fat-reference.poly'
out=['# Capacity-limited connected reference proposal; native admission false.','7233 3 0 1']
out+=['%d %.17g %.17g %.17g %d'%(j+1,*v,j+1)for j,v in enumerate(R)]
out+=[str(nf)+' 1']
for f,m in zip(F,marks):out+=['1 0 '+str(m),'3 '+' '.join(str(int(v)+1)for v in f)]
out+=tail;poly.write_text('\n'.join(out)+'\n')
sys.path.insert(0,str(E/'incremental-exact-audit-owner-253/src'))
pred=E/'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py'
spec=importlib.util.spec_from_file_location('numilab_human._reference116',pred);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
start=time.monotonic();records,_=m.float32_surface_records(R.astype(np.float32),F,source_seam_groups=())
a=m._audit_self_pairs_touching_faces(records,set(range(len(F))));pairs=np.asarray(a['triangle_pairs'],np.int64).reshape(-1,2)
np.save(P/'surface-self-pairs.npy',pairs)
report={'host':socket.gethostname(),'input_poly_sha256':sha(base),'reference_proposal_sha256':sha(proposal),'output_poly_sha256':sha(poly),
 'point_count':len(R),'PLC_facets':len(F),'exact_surface_self_pairs':int(len(pairs)),'complete_surface_audit':a['complete_surface_audit'],'predicate_sha256':sha(pred),
 'self_audit_seconds':time.monotonic()-start,'accepted_physical_steps':0,'geometry_admitted':False,'source_mass_mapping_reusable':False,
 'surface_pair_facet_markers':[[int(marks[a]),int(marks[b])]for a,b in pairs[:100]],
 'preserved_region_seed_lines':tail[2:]}
if len(pairs):
 report.update(status='STOPPED_AT_SURFACE_GATE',next_change='Reconstruct intersecting inner reference facets from exact witnesses before tetrahedralization. No invalid cells are installed or hidden by changing quality thresholds.')
else:
 T=E/'arm-local-pocket-remesh-116/local-remesh-118/tetgen-v1.6.0/tetgen';cmd=[str(T),'-pq1.2/10AS512',str(poly)]
 t=time.monotonic()
 try:
  cp=subprocess.run(cmd,capture_output=True,text=True,timeout=45);rc=cp.returncode;stdout=cp.stdout;stderr=cp.stderr;timeout=False
 except subprocess.TimeoutExpired as e:
  rc=None;stdout=e.stdout.decode()if isinstance(e.stdout,bytes)else(e.stdout or'');stderr=e.stderr.decode()if isinstance(e.stderr,bytes)else(e.stderr or'');timeout=True
 (P/'tetgen.stdout.log').write_text(stdout);(P/'tetgen.stderr.log').write_text(stderr)
 report.update(status='TETGEN_DIAGNOSTIC_EXECUTED',command=cmd,tetgen_binary_sha256=sha(T),tetgen_return_code=rc,tetgen_timed_out=timeout,tetgen_wall_seconds=time.monotonic()-t)
(P/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
