#!/usr/bin/env python3
"""Build one same-point whole-volume TetGen PLC trial and strictly audit its output."""
from __future__ import annotations
import csv, hashlib, json, math, os, pathlib, subprocess, sys, time
from collections import defaultdict, deque
import numpy as np

D = pathlib.Path(__file__).resolve().parent
OID = pathlib.Path('/Users/n/Work/294-source-frame-audit-001/connected-interior-reference-fit-001/retessellation-001/volume-cavity-001/oid-cavity-replacement.npz')
NATIVE = pathlib.Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003/candidate.npz')
CURRENT = pathlib.Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/native-candidate020-all-bank-containment-035/source-preflight-003/initial-skin-fem-nodes.csv')
NATIVE_TETS = pathlib.Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/connected-skin-parent-mass-transfer-067/compiled-cell-export-068/run-004/source-preflight/compiled-skin-fem-tets.csv')
TETGEN = pathlib.Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/arm-local-pocket-remesh-116/local-remesh-118/tetgen-v1.6.0/tetgen')
POLY = D / 'whole-skin-fat-reference.poly'
PREFIX = D / 'whole-skin-fat-reference'
EXPECTED_NATIVE_TET_CSV_SHA = 'a7dc2d97713e43ed2bfae2aa805bb5281fdf2361108cf208df4cbdea10084461'


def sha_file(p: pathlib.Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def sha_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def parse_current():
    X=np.full((7233,3),np.nan,dtype=np.float64); gids=np.full(7233,-1,dtype=np.int64)
    seen=set()
    with CURRENT.open(newline='') as f:
        for r in csv.DictReader(f):
            layer=int(r['layer']); row=int(r['row']); gid=int(r['global_node'])
            if layer not in (0,1,2) or not 0<=row<2411: raise ValueError('invalid layer/row')
            ix=layer*2411+row
            if ix in seen: raise ValueError('duplicate layer/row')
            seen.add(ix); gids[ix]=gid; X[ix]=[float(r['x_m']),float(r['y_m']),float(r['z_m'])]
    if len(seen)!=7233 or not np.isfinite(X).all() or len(set(gids.tolist()))!=7233: raise ValueError('current node CSV coverage/identity invalid')
    return X.astype(np.float32),gids

def dets(P,T):
    Q=np.asarray(P)[T].astype(np.float64)
    return np.einsum('ij,ij->i',np.cross(Q[:,1]-Q[:,0],Q[:,2]-Q[:,0]),Q[:,3]-Q[:,0])

def qualities(P,T):
    Q=np.asarray(P)[T].astype(np.float64)
    d=dets(P,T)
    e2=sum(np.einsum('ij,ij->i',Q[:,i]-Q[:,j],Q[:,i]-Q[:,j]) for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)))
    return 12.0*np.maximum(d/2,0.0)**(2.0/3.0)/e2

# For a positive tet [a,b,c,d], these are the induced outward-oriented faces.
FACE_LOCAL=((1,2,3),(0,3,2),(0,1,3),(0,2,1))
def face_key(f): return tuple(sorted(map(int,f)))
def cyc_key(f):
    f=tuple(map(int,f)); i=f.index(min(f)); return f[i:]+f[:i]

def build_incidence(T,M,R):
    inc=defaultdict(list)
    for ti,(tet,mat) in enumerate(zip(T,M)):
        for opp,loc in enumerate(FACE_LOCAL):
            tri=tuple(int(tet[j]) for j in loc)
            inc[face_key(tri)].append((ti,int(mat),tri))
    bad=[(k,len(v)) for k,v in inc.items() if len(v)>2]
    if bad: raise ValueError(f'nonmanifold tetra face incidence {bad[:3]}')
    return inc

def connected_material_components(T,M,inc,R):
    graph=[[] for _ in range(len(T))]
    for entries in inc.values():
        if len(entries)==2 and entries[0][1]==entries[1][1]:
            a,b=entries[0][0],entries[1][0]; graph[a].append(b); graph[b].append(a)
    unseen=set(range(len(T))); comps=[]
    while unseen:
        s=unseen.pop(); q=[s]; cc=[s]
        while q:
            u=q.pop()
            for v in graph[u]:
                if v in unseen and M[v]==M[s]: unseen.remove(v); q.append(v); cc.append(v)
        comps.append((int(M[s]),np.asarray(cc,dtype=np.int64)))
    comps.sort(key=lambda x:(x[0],int(x[1].min())))
    seeds=[]
    for attr,rows in comps:
        row=int(rows.min()); seed=np.mean(R[T[row]],axis=0)
        seeds.append({'material':attr,'component_cells':int(len(rows)),'seed_tet_row':row,'seed_m':seed.tolist()})
    return seeds,graph

def load_inputs():
    with np.load(OID,allow_pickle=False) as z:
        R=z['reference_points_AP_m'].copy()
        O=z['old_O_faces'].copy(); I=z['old_I_faces'].copy(); Df=z['old_D_faces'].copy()
    with np.load(NATIVE,allow_pickle=False) as z:
        T=z['old_tetrahedra'].copy(); M=z['old_material_ids'].copy()
    X,gids=parse_current()
    if R.shape!=(7233,3) or T.shape!=(28632,4) or M.shape!=(28632,): raise ValueError('unexpected baseline dimensions')
    if not np.array_equal(np.unique(M),np.array([1,2],dtype=M.dtype)) or np.count_nonzero(M==1)!=14316 or np.count_nonzero(M==2)!=14316: raise ValueError('native skin/fat material counts mismatch')
    if not np.all((T>=0)&(T<7233)) or not np.isfinite(R).all(): raise ValueError('invalid node references')
    if sha_file(NATIVE_TETS)!=EXPECTED_NATIVE_TET_CSV_SHA: raise ValueError('native compiled tet CSV pin mismatch')
    # Prove actual native connectivity rows through compiled global-node identities.
    with NATIVE_TETS.open(newline='') as f: native_rows=list(csv.DictReader(f))
    if len(native_rows)!=len(T): raise ValueError('native compiled tet CSV row count mismatch')
    g2i={int(g):i for i,g in enumerate(gids)}
    nativeT=np.empty_like(T); nativeM=np.empty_like(M)
    seen=np.zeros(len(T),bool)
    for r in native_rows:
        sr=int(r['source_tet_row'])
        if not 0<=sr<len(T) or seen[sr]: raise ValueError('native source_tet_row not unique/contiguous')
        seen[sr]=True
        nativeT[sr]=[g2i[int(r[f'global_node{i}'])] for i in range(4)]
        mi=int(r['authored_material_index'])
        nativeM[sr]=1 if mi==15 else 2 if mi==16 else 0
    if not seen.all() or not np.array_equal(nativeT,T): raise ValueError('026 old tetrahedra do not exactly match pinned native compiled CSV after global ID mapping')
    if not np.array_equal(nativeM,M): raise ValueError('026 material rows differ from native compiled 15/16 mapping')
    if np.any(dets(R,T)<=0): raise ValueError('baseline reference contains nonpositive oriented cells')
    if np.any(dets(X,T)<=0): raise ValueError('035 initial current contains nonpositive oriented cells')
    qR=qualities(R.astype(np.float32),T); qX=qualities(X,T); detR=dets(R.astype(np.float32),T); detX=dets(X,T); J=detX/detR
    if np.any(detR<=0) or np.any(detX<=0): raise ValueError('F32 reference/current has nonpositive cells')
    if np.any(qR<1e-5) or np.any(qX<1e-5): raise ValueError('baseline reference/current violates existing q floor')
    inc=build_incidence(T,M,R)
    # Check source O/I/D triangles are exactly the actual-native exterior/interface partition.
    setO={face_key(x) for x in O}; setI={face_key(x) for x in I}; setD={face_key(x) for x in Df}
    ext={k for k,v in inc.items() if len(v)==1}
    interface={k for k,v in inc.items() if len(v)==2 and v[0][1]!=v[1][1]}
    if len(setO)!=len(O) or len(setI)!=len(I) or len(setD)!=len(Df): raise ValueError('duplicate archive O/I/D triangles')
    if not setO.issubset(ext) or not setD.issubset(ext) or setO&setD: raise ValueError(f'archive O/D faces are not subsets of actual exterior ext={len(ext)} O={len(setO)} D={len(setD)}')
    if interface != setI: raise ValueError(f'archive interface mismatch {len(interface)} vs {len(setI)}')
    side_caps=ext-(setO|setD)
    # These are the expected O-I and I-D closure strips where the source O/D
    # surfaces terminate: 45 open boundary edges × two side triangles × two bands.
    if len(side_caps)!=180: raise ValueError(f'unexpected unmatched exterior side-cap count {len(side_caps)}')
    if any(len(v)==2 and v[0][1]!=v[1][1] for k,v in inc.items() if k not in setI): raise ValueError('unexpected material interface face')
    return R,X,gids,T,M,O,I,Df,inc,detR,detX,qR,qX,J

def orient_input_boundaries(inc):
    facets=[]
    for k,entries in inc.items():
        if len(entries)==1:
            ti,mat,tri=entries[0]
            kind='external'
        elif entries[0][1]!=entries[1][1]:
            e=entries[0] if entries[0][1]==1 else entries[1]
            ti,mat,tri=e; kind='interface'
        else: continue
        facets.append((kind,k,mat,tri))
    return facets

def write_poly(R,T,M,O,I,Df,inc,seeds,gids):
    # Source triangle row IDs become unique facet markers, namespaced by surface.
    marker_by_key={}
    marker_desc={}
    for i,f in enumerate(O): marker_by_key[face_key(f)]=100000+i+1; marker_desc[face_key(f)]=('O',i)
    for i,f in enumerate(I): marker_by_key[face_key(f)]=200000+i+1; marker_desc[face_key(f)]=('I',i)
    for i,f in enumerate(Df): marker_by_key[face_key(f)]=300000+i+1; marker_desc[face_key(f)]=('D',i)
    facets=orient_input_boundaries(inc)
    extra=[(kind,k,mat,tri) for kind,k,mat,tri in facets if k not in marker_by_key]
    for j,(kind,k,mat,tri) in enumerate(sorted(extra,key=lambda x:x[1])):
        marker_by_key[k]=400000+j+1; marker_desc[k]=(f'SIDE-{tuple(sorted({v//2411 for v in k}))}',j)
    if len(facets)!=len(marker_by_key): raise ValueError('surface marker map does not cover every exterior/interface face')
    with POLY.open('w') as f:
        f.write('# Frozen whole-reference PLC; node markers encode exact input row+1.\n')
        f.write(f'{len(R)} 3 0 1\n')
        for i,p in enumerate(R): f.write(f'{i+1} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g} {i+1}\n')
        f.write(f'{len(facets)} 1\n')
        for kind,k,mat,tri in facets:
            marker=marker_by_key[k]
            f.write(f'1 0 {marker}\n')
            f.write(f'3 {tri[0]+1} {tri[1]+1} {tri[2]+1}\n')
        f.write('0\n') # no holes; O/I/D are a closed skin/fat PLC
        f.write(f'{len(seeds)}\n')
        for j,s in enumerate(seeds,1):
            x,y,z=s['seed_m']; f.write(f'{j} {x:.17g} {y:.17g} {z:.17g} {s["material"]} -1\n')
    with (D/'facet-marker-map.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['marker','surface','source_face_row','vertex_ids_zero_based'])
        for kind,k,mat,tri in facets:
            name,source_row=marker_desc[k]
            w.writerow([marker_by_key[k],name,source_row,' '.join(map(str,tri))])
    (D/'region-seeds.json').write_text(json.dumps(seeds,indent=2,sort_keys=True)+'\n')
    return facets

def read_tetgen_nodes(path,R):
    lines=path.read_text().splitlines(); h=list(map(int,lines[0].split()[:4])); n,dim,attrs,markers=h
    if dim!=3 or attrs!=0 or markers!=1: raise ValueError('unexpected TetGen node header')
    nodes={}; input_index={}; seenmarkers=set()
    for line in lines[1:]:
        q=line.split()
        if not q or q[0].startswith('#'): continue
        if len(q)<5: raise ValueError('malformed node record')
        nid=int(q[0]); p=np.array(list(map(float,q[1:4])),np.float64); marker=int(q[4])
        ix=marker-1
        if not 0<=ix<len(R) or marker in seenmarkers: raise ValueError('node marker mapping not a unique input identity')
        if not np.array_equal(p,R[ix]): raise ValueError(f'node {nid} marker {marker} moved from input reference position')
        seenmarkers.add(marker); nodes[nid]=p; input_index[nid]=ix
    if len(nodes)!=n or len(seenmarkers)!=len(R): raise ValueError(f'point identity coverage mismatch: out {len(nodes)} input {len(R)}')
    return nodes,input_index

def read_ele(path,input_index):
    lines=path.read_text().splitlines(); h=list(map(int,lines[0].split()[:3])); nt,nv,na=h
    if nv!=4 or na<1: raise ValueError('TetGen output must contain 4-node tets and region attributes')
    T=np.empty((nt,4),np.int32); M=np.empty(nt,np.int8); seen=np.zeros(nt,bool)
    for line in lines[1:]:
        q=line.split()
        if not q or q[0].startswith('#'): continue
        ti=int(q[0])-1
        if not 0<=ti<nt or seen[ti]:raise ValueError('invalid/duplicate tetra row id')
        T[ti]=[input_index[int(s)] for s in q[1:5]]
        attr=float(q[5]); mat=int(round(attr))
        if abs(attr-mat)>1e-8 or mat not in (1,2):raise ValueError(f'bad material attribute {attr}')
        M[ti]=mat; seen[ti]=True
    if not seen.all():raise ValueError('incomplete tet output')
    return T,M

def boundary_chains(T,M,coords):
    inc=build_incidence(T,M,coords)
    by={'external_skin':[],'external_fat':[],'interface_skin':[],'interface_fat':[]}
    for k,v in inc.items():
        if len(v)==1:
            _,m,tri=v[0]; by['external_skin' if m==1 else 'external_fat'].append(cyc_key(tri))
        elif v[0][1]!=v[1][1]:
            a=next(x for x in v if x[1]==1); b=next(x for x in v if x[1]==2)
            by['interface_skin'].append(cyc_key(a[2]));by['interface_fat'].append(cyc_key(b[2]))
    return {k:sorted(v) for k,v in by.items()},inc

def audit_output(R,X,Told,Mold,detRold,detXold,qRold,qXold,Jold,seeds,facets):
    nodesfile=PREFIX.with_suffix('.1.node'); elefile=PREFIX.with_suffix('.1.ele')
    nodes,imap=read_tetgen_nodes(nodesfile,R)
    T,M=read_ele(elefile,imap)
    # Basic topology and positivity at original reference/current positions.
    canon=np.sort(T,axis=1); uniq,counts=np.unique(canon,axis=0,return_counts=True)
    inc=build_incidence(T,M,R)
    dR=dets(R.astype(np.float32),T); dX=dets(X,T)
    qR=qualities(R.astype(np.float32),T); qX=qualities(X,T); J=dX/dR
    if np.any(dR<=0) or np.any(dX<=0): pass # retained as a diagnostic outcome, not silently repaired
    oldchains,_=boundary_chains(Told,Mold,R)
    newchains,_=boundary_chains(T,M,R)
    chain_cmp={k:{'old_faces':len(oldchains[k]),'new_faces':len(newchains[k]),'exact_oriented_equal':oldchains[k]==newchains[k]} for k in oldchains}
    volold={};volnew={}
    for m in (1,2):
        volold[str(m)]=float(np.sum(dets(R.astype(np.float64),Told[Mold==m]))/6)
        volnew[str(m)]=float(np.sum(dets(R.astype(np.float64),T[M==m])/6))
    report={
      'status':'AUDITED_TETGEN_OUTPUT',
      'input_node_identity':{'input_nodes':len(R),'output_nodes':len(nodes),'all_exact_same_positions':True,'all_input_markers_unique':True},
      'tet_counts':{'old':len(Told),'new':len(T),'new_material_counts':{str(m):int(np.count_nonzero(M==m)) for m in (1,2)},'region_seeds':len(seeds)},
      'topology':{'unique_tet_sets':int(len(uniq)),'duplicate_tet_count':int(np.sum(counts-1)),'face_count':len(inc),'nonmanifold_faces_gt2':int(sum(len(v)>2 for v in inc.values())),'external_faces':int(sum(len(v)==1 for v in inc.values())),'material_interfaces':int(sum(len(v)==2 and v[0][1]!=v[1][1] for v in inc.values()))},
      'oriented_boundary_chain_comparison':chain_cmp,
      'volume_m3_by_material':{'old':volold,'new':volnew,'relative_change':{str(m):(volnew[str(m)]-volold[str(m)])/volold[str(m)] for m in (1,2)}},
      'reference_float32':{'nonpositive':int(np.count_nonzero(dR<=0)),'min_det_m3':float(dR.min()),'q_below_1e-5':int(np.count_nonzero(qR<1e-5)),'min_q':float(qR.min())},
      'current035_float32':{'nonpositive':int(np.count_nonzero(dX<=0)),'min_det_m3':float(dX.min()),'q_below_1e-5':int(np.count_nonzero(qX<1e-5)),'min_q':float(qX.min())},
      'jacobian_current_over_reference':{'below_0_35':int(np.count_nonzero(J<0.35)),'above_2_5':int(np.count_nonzero(J>2.5)),'min':float(J.min()),'max':float(J.max())},
      'acceptance':{'same_reference_nodes':True,'zero_steiner_expected':True,'positive_reference_and_current':bool(np.all(dR>0) and np.all(dX>0)),'reference_and_current_q_floor':bool(np.all(qR>=1e-5) and np.all(qX>=1e-5)),'all_J_within_0_35_2_5':bool(np.all((J>=0.35)&(J<=2.5))),'oriented_external_and_interface_boundary_exact':all(v['exact_oriented_equal'] for v in chain_cmp.values()),'no_duplicate_tets':bool(np.sum(counts-1)==0),'no_nonmanifold_faces':bool(all(len(v)<=2 for v in inc.values()))},
      'caveat':'Geometry/connectivity diagnostic only. TetGen does not establish physical/material mass conservation; attributes derive from original face-connected components and volume is separately compared.'
    }
    (D/'output-audit.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report

def prepare():
    R,X,gids,T,M,O,I,Df,inc,detR,detX,qR,qX,J=load_inputs()
    seeds,graph=connected_material_components(T,M,inc,R)
    facets=write_poly(R,T,M,O,I,Df,inc,seeds,gids)
    pre={
      'schema':'numi.whole-reference-tetgen-083.preflight.v1',
      'status':'READY_FOR_ONE_BOUNDED_TETGEN_TRIAL',
      'inputs':{str(p):sha_file(p) for p in (OID,NATIVE,CURRENT,NATIVE_TETS,TETGEN)},
      'native_tet_row_identity':{'native_csv_sha256':sha_file(NATIVE_TETS),'expected_sha256':EXPECTED_NATIVE_TET_CSV_SHA,'row_for_row_after_global_ids':True,'native_array_sha256':sha_array(T),'native_materials_sha256':sha_array(M)},
      'dimensions':{'points':len(R),'tets':len(T),'materials':{str(m):int(np.count_nonzero(M==m)) for m in (1,2)},'O_faces':len(O),'I_faces':len(I),'D_faces':len(Df),'derived_side_cap_faces':len(facets)-len(O)-len(I)-len(Df),'PLC_facets':len(facets)},
      'material_components':[],
      'region_seeds':seeds,
      'baseline':{'reference_nonpositive_F32':int(np.count_nonzero(detR<=0)),'reference_q_min':float(qR.min()),'current035_nonpositive_F32':int(np.count_nonzero(detX<=0)),'current035_q_min':float(qX.min()),'J_min':float(J.min()),'J_max':float(J.max()),'J_outside_0_35_2_5':int(np.count_nonzero((J<.35)|(J>2.5)))},
      'poly_sha256':sha_file(POLY),
      'limitations':'Preflight is based on reference-geometry PLC and frozen current035 positions. It is not an anatomical/material admission.'
    }
    pre['material_components']=[{'material':m,'component_cells':n,'seed_tet_row':int(s['seed_tet_row'])} for m,n,s in [(x['material'],x['component_cells'],x) for x in seeds]]
    (D/'preflight.json').write_text(json.dumps(pre,indent=2,sort_keys=True)+'\n')
    print(json.dumps(pre,indent=2,sort_keys=True))


def run():
    if not POLY.exists(): prepare()
    R,X,gids,T,M,O,I,Df,inc,detR,detX,qR,qX,J=load_inputs()
    seeds,graph=connected_material_components(T,M,inc,R)
    facets=orient_input_boundaries(inc)
    cmd=[str(TETGEN),'-pYYq1.2/10AS0',POLY.name]
    start=time.monotonic()
    try:
        cp=subprocess.run(cmd,cwd=D,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=45,check=False)
        elapsed=time.monotonic()-start
        (D/'tetgen.log').write_text(cp.stdout)
        (D/'tetgen-run.json').write_text(json.dumps({'command':cmd,'cwd':str(D),'returncode':cp.returncode,'elapsed_s':elapsed,'timed_out':False,'tetgen_sha256':sha_file(TETGEN),'poly_sha256':sha_file(POLY)},indent=2,sort_keys=True)+'\n')
        print('tetgen return',cp.returncode,'elapsed',elapsed)
        print(cp.stdout[-5000:])
        if cp.returncode==0 and PREFIX.with_suffix('.1.node').exists() and PREFIX.with_suffix('.1.ele').exists():
            r=audit_output(R,X,T,M,detR,detX,qR,qX,J,seeds,facets)
            print(json.dumps(r,indent=2,sort_keys=True))
    except subprocess.TimeoutExpired as e:
        out=(e.stdout or b'').decode(errors='replace') if isinstance(e.stdout,bytes) else (e.stdout or '')
        (D/'tetgen.log').write_text(out)
        (D/'tetgen-run.json').write_text(json.dumps({'command':cmd,'cwd':str(D),'timed_out':True,'timeout_s':45,'elapsed_s':time.monotonic()-start,'tetgen_sha256':sha_file(TETGEN),'poly_sha256':sha_file(POLY)},indent=2,sort_keys=True)+'\n')
        raise

if __name__=='__main__':
    if len(sys.argv)==2 and sys.argv[1]=='--prepare': prepare()
    elif len(sys.argv)==2 and sys.argv[1]=='--run': run()
    else: raise SystemExit('usage: build_whole_reference_plc_083.py --prepare|--run')
