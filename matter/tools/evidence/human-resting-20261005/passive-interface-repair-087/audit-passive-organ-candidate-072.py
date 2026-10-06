from pathlib import Path
import hashlib,json,sys,time,numpy as np
E=Path('/Users/n/numi-human-resting-evidence-20261005');source=E/'passive-organ-interface-conditioned-071';out=E/'passive-organ-candidate-audit-072';out.mkdir(exist_ok=False)
sys.path.insert(0,'/Users/n/numi-human-resting-conforming-source-009/matter/tools')
import accepted_mrvpack_surface_audit as audit
predicate=Path('/Users/n/numi-human-neighbor-source-001/src/numilab_human/cardiac_cavity_intersections.py');pred=audit.predicate_module(predicate)
base=E/'passive-neighbor-depth-062';d=np.load(base/'source-surfaces.npz');metadata=json.loads((base/'input.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
meshes={s:(d[f'v{s}'],d[f'f{s}']) for s in metadata['stable_ids']}
for s in [3,13]:
    item=np.load(source/f'surface-{s}.npz');meshes[s]=(item['vertices'],item['faces'])
scale=audit.coordinate_lattice_scale(float(c) for v,f in meshes.values() for p in v for c in p)
records={}
def rec(s):
    if s not in records:
        v,f=meshes[s];points=[tuple(audit.lattice_integer(float(c),scale) for c in p) for p in v];records[s]=pred._records(points,f.tolist())
    return records[s]
report={'scope':'Full exact source self and passive-neighbor triangle-intersection audit of the two inferred interface candidates. No input faces omitted. Accepted dynamic anatomy remains a separate gate.',
        'source_payload_sha256':metadata['payload_sha256'],'driver_sha256':sha(__file__),'predicate_sha256':sha(predicate),
        'candidate_sha256':{str(s):sha(source/f'surface-{s}.npz') for s in [3,13]},'self':{},'pairs':[]}
start=time.monotonic()
for s in [3,13]:
    report['self'][str(s)]=pred._audit_pair(rec(s),rec(s),same_surface=True)
    print('self',s,report['self'][str(s)]['count'],flush=True)
for a,b in sorted({tuple(sorted((a,b))) for a in [3,13] for b in meshes if a!=b}):
    v,f=meshes[a];w,g=meshes[b]
    if np.any(v.max(0)<w.min(0)) or np.any(w.max(0)<v.min(0)):
        result={'count':0,'aabb_disjoint':True,'audit_complete':True}
    else:
        try:result=pred._audit_pair(rec(a),rec(b),same_surface=False)
        except ValueError as err:result={'invalid_geometry':str(err),'audit_complete':False}
    report['pairs'].append({'ids':[a,b],**result})
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if result.get('count') or not result.get('audit_complete',True):print(a,b,result.get('count'),result.get('invalid_geometry'),flush=True)
report['elapsed_seconds']=time.monotonic()-start
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('DONE',report['elapsed_seconds'],flush=True)
