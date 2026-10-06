from pathlib import Path
import hashlib,json,sys,numpy as np

E=Path('/Users/n/numi-human-resting-evidence-20261005');out=E/'taenia-native-diagnostic-099';out.mkdir(exist_ok=False)
sys.path.insert(0,'/Users/n/numi-human-resting-conforming-source-009/matter/tools');import accepted_mrvpack_surface_audit as audit
source=E/'taenia-conditioned-audit-086/surface-457.npz';d=np.load(source);v=d['vertices'].astype(float);f=d['faces']
result=json.loads((E/'taenia-native-audit-095/report.json').read_text())
pairs=sorted({tuple(pair) for r in result['frames'] for pair in r['self']['triangle_pairs']})
vertices=sorted({int(i) for pair in pairs for fid in pair for i in f[fid]})
def describe(x):
    rows=[]
    for a,b in pairs:
        p=x[f[a]];q=x[f[b]];n=np.cross(q[1]-q[0],q[2]-q[0]);n/=np.linalg.norm(n)
        na=np.cross(p[1]-p[0],p[2]-p[0]);na/=np.linalg.norm(na)
        rows.append({'faces':[a,b],'indices':[f[a].tolist(),f[b].tolist()],
                     'shared_vertices':sorted(set(map(int,f[a]))&set(map(int,f[b]))),
                     'a_to_b_plane_m':((p-q[0])@n).tolist(),'b_to_a_plane_m':((q-p[0])@na).tolist(),
                     'normal_dot':float(na@n)})
    return rows
report={'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_pairs':describe(v),'local_source_vertices':{str(i):v[i].tolist() for i in vertices},'native':[]}
for step in [0,639,2783,2999]:
    pack=E/'taenia-native-094/accepted-geometry'/f'step-{step}.mrvpack';mm,stream,offset,surfaces=audit.read_pack(pack)
    ff=np.asarray(surfaces[(51010,457)]['faces']);base=int(ff.min());assert np.array_equal(ff-base,f)
    points=audit.surface_points(mm,offset,ff.tolist());native=np.asarray([points[1][base+i] for i in range(len(v))])
    np.savez_compressed(out/f'native-{step}.npz',vertices=native,faces=f)
    report['native'].append({'step':step,'pairs':describe(native),'local_native_vertices':{str(i):native[i].tolist() for i in vertices}})
    mm.close();stream.close()
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
