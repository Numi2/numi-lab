from pathlib import Path
import numpy as np,sys,json,hashlib,subprocess,time
D=Path('/Users/numi/Work/numi-human-connected-surface-069')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [l.split('#')[0].split()for l in p.read_text().splitlines()if l.split('#')[0].strip()]
def main():
 C=D/'connected-boundary-continuation-123/run-001/surface-proposal.npz';P=D/'connected-boundary-remesh-124/run-001';assert not P.exists();P.mkdir(parents=True)
 assert sha(C)=='34f77de89404b60690d156a6a39b136862ae571e45bbd6cdab29418e8e1e85d6'
 gate=json.loads((C.parent/'report.json').read_text());assert gate['surface_and_local_orientation_gate_pass']
 z=np.load(C);F=z['surface_faces_zero_based'];M=z['surface_face_markers'];used=np.unique(F);reindex=np.full(len(z['reference_points_m']),-1,np.int64);reindex[used]=np.arange(len(used));F=reindex[F];R=z['reference_points_m'][used]
 assert np.array_equal(used[:7233],np.arange(7233))and len(used)==7671 and F.min()>=0
 original=D/'whole-reference-tetgen-084-bounded/whole-skin-fat-reference.poly';tail=[r.strip()for r in original.read_text().splitlines()if r.strip()and not r.lstrip().startswith('#')][-4:];assert tail[0]=='0'and tail[1]=='2'
 out=['# Connected O/I/D barrier-continuation proposal. No native admission.',f'{len(R)} 3 0 1']
 out+=['%d %.17g %.17g %.17g %d'%(j+1,*p,j+1)for j,p in enumerate(R)];out+=[f'{len(F)} 1']
 for f,m in zip(F,M):out+=['1 0 '+str(int(m)),'3 '+' '.join(str(int(v)+1)for v in f)]
 out+=tail;poly=P/'whole-skin-fat-reference.poly';poly.write_text('\n'.join(out)+'\n')
 np.savez_compressed(P/'input-layout.npz',input_row_to_123_output_row=used,input_reference_points_m=R,surface_faces_zero_based=F,surface_face_markers=M,input_node_component=z['node_connected_component'][used])
 binary=E/'arm-local-pocket-remesh-116/local-remesh-118/tetgen-v1.6.0/tetgen';assert sha(binary)=='95b1df3923d7405b168ba5a3c4b39498ddfc0e7ba94000dedff7ebf93350de07'
 cmd=[str(binary),'-pq1.2/10AS512',str(poly)];t=time.monotonic();cp=subprocess.run(cmd,capture_output=True,text=True,timeout=45);wall=time.monotonic()-t
 (P/'tetgen.stdout.log').write_text(cp.stdout);(P/'tetgen.stderr.log').write_text(cp.stderr)
 report={'source_candidate_sha256':sha(C),'surface_gate_report_sha256':sha(C.parent/'report.json'),'poly_sha256':sha(poly),'input_points':len(R),'input_facets':len(F),'unused_084_generated_points_excluded':127,'command':cmd,'tetgen_binary_sha256':sha(binary),'tetgen_return_code':cp.returncode,'tetgen_wall_seconds':wall,'accepted_physical_steps':0,'geometry_admitted':False}
 (P/'report.json').write_text(json.dumps(report,indent=2)+'\n');assert cp.returncode==0
 n=rows(P/'whole-skin-fat-reference.1.node');e=rows(P/'whole-skin-fat-reference.1.ele');f=rows(P/'whole-skin-fat-reference.1.face')
 ids=np.asarray([int(r[0])for r in n[1:]],np.int64);NR=np.asarray([[float(v)for v in r[1:4]]for r in n[1:]])
 assert np.array_equal(ids,np.arange(1,len(NR)+1))and np.array_equal(NR[:len(R)],R)
 T=np.asarray([[int(v)-1 for v in r[1:5]]for r in e[1:]],np.int64);mat=np.asarray([int(float(r[5]))for r in e[1:]],np.int8)
 SF=np.asarray([[int(v)-1 for v in r[1:4]]for r in f[1:]],np.int64);SM=np.asarray([int(r[4])for r in f[1:]],np.int64)
 X=NR.astype(np.float32);sys.path.insert(0,str(D/'connected-cell-trust-111/matter/tools'));from human_skin_fat_connected_repair import cell_values_gradients
 J,_,q,_=cell_values_gradients(X,NR,T);V=NR[T];vol=np.einsum('ij,ij->i',V[:,1]-V[:,0],np.cross(V[:,2]-V[:,0],V[:,3]-V[:,0]))/6
 np.savez_compressed(P/'candidate.npz',reference_points_m=NR,points_world_f32=X,tetrahedra=T,material_ids=mat,output_node_ids=ids,source_input_row=np.r_[np.arange(len(R)),np.full(len(NR)-len(R),-1)],surface_faces_zero_based=SF,surface_face_markers=SM,cell_uids=np.asarray([int(r[0])for r in e[1:]],np.int64),input_row_to_123_output_row=used)
 report.update(candidate_sha256=sha(P/'candidate.npz'),points=len(NR),cells=len(T),generated_nodes=len(NR)-len(R),invalid_reference_cells=int((vol<=0).sum()),invalid_Float32_cells=int((J<=0).sum()),below_quality_floor_1e_minus5=int((q<1e-5).sum()),q_min=float(q.min()),J_min=float(J.min()),J_max=float(J.max()),material_volume_m3={str(k):float(vol[mat==k].sum())for k in np.unique(mat)},remaining_gates='Exact full emitted surface, bone and surrounding solids, conservative native mass/ancestry, detail mapping and native qualification.')
 (P/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
