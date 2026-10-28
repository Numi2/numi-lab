"""Frozen accepted-viewer geometry and bone-frame replay; no dynamics implementation."""
from pathlib import Path
import csv, hashlib, json, subprocess
import numpy as np

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read_receipt(path, expected_sha=None):
    path=Path(path)
    if expected_sha is not None and sha(path)!=expected_sha:
        raise ValueError("accepted receipt hash mismatch")
    r=json.loads(path.read_text())
    if (r.get("schema")!="numi.human.accepted-render-geometry.v1" or
        r.get("physical_endpoint")!="accepted" or r.get("surface_audit_endpoint")!="passed" or
        type(r.get("accepted_step")) is not int or r["accepted_step"]<=0 or
        not np.isfinite(r.get("accepted_time_s",float("nan"))) or r["accepted_time_s"]<=0 or
        any(type(r.get(k)) is not int or not 0<r[k]<2**64 for k in ("accepted_transaction_fingerprint","accepted_root_fingerprint"))):
        raise ValueError("receipt is not a positive-time accepted native endpoint")
    p=Path(r["accepted_pack_path"])
    if sha(p)!=r["pack_file_sha256"]:
        raise ValueError("accepted native pack hash mismatch")
    poses=r.get("accepted_body_poses",[])
    if len(poses)!=r.get("accepted_body_pose_count") or not poses:
        raise ValueError("accepted native endpoint has incomplete body poses")
    ids=[x["body_index"] for x in poses]
    if len(set(ids))!=len(ids) or sorted(ids)!=list(range(len(ids))):
        raise ValueError("accepted native endpoint has duplicate or missing body IDs")
    return r

def pose_bank_frames(receipt_path, bodies, required):
    r=read_receipt(receipt_path)
    poses={x["body_index"]:x for x in r["accepted_body_poses"]}
    matrices={};positions={};rows=[]
    for bi,old in bodies.items():
        if bi not in poses:
            raise ValueError("accepted native pose missing source body "+str(bi))
        rec=dict(old);p=poses[bi]
        position=np.asarray(p["position_m"],np.float32)
        q=np.asarray(p["quaternion_xyzw"],np.float32)
        if position.shape!=(3,) or q.shape!=(4,) or not np.isfinite(position).all() or not np.isfinite(q).all():
            raise ValueError("invalid accepted Float32 body pose")
        # Preserve raw stored quaternion arithmetic. Renormalizing would change
        # the source frame and differ from the pinned bank reader convention.
        if abs(float(np.dot(q.astype(float),q.astype(float)))-1)>2e-5:
            raise ValueError("accepted source quaternion outside normalization tolerance")
        inverse=q.astype(float).copy();inverse[:3]*=-1
        def rotate(v):
            return v+2*np.cross(inverse[:3],np.cross(inverse[:3],v)+inverse[3]*v)
        matrix=np.column_stack([rotate(v) for v in np.eye(3)])
        for axis,value,bits in zip("xyz",position,position.view(np.uint32)):
            rec["position_"+axis+"_m"]=format(float(value),".17g")
            rec["position_"+axis+"_bits_u32"]=str(int(bits))
        for axis,value,bits in zip("xyzw",q,q.view(np.uint32)):
            rec["orientation_q"+axis]=format(float(value),".17g")
            rec["orientation_q"+axis+"_bits_u32"]=str(int(bits))
        for i in range(3):
            for j in range(3):rec[f"world_to_body_m{i}{j}"]=format(float(matrix[i,j]),".17g")
        matrices[bi]=matrix;positions[bi]=position.astype(float);rows.append(rec)
    if any(int(bi) not in matrices for bi in required):
        raise ValueError("accepted frame coverage misses a typed bone bank")
    return matrices,positions,rows

def read_world_banks(directory, inventory_path, expected_pack_sha=None):
    directory=Path(directory)
    if expected_pack_sha is not None:
        policy=json.loads((directory.parent/'coordinate-policy.json').read_text())
        if policy['pack_sha256']!=expected_pack_sha or any(
                sha(directory/name)!=policy['files_sha256'][name]
                for name in ('exact-emitted.triangles.csv','exact-emitted.bodies.csv')):
            raise ValueError('accepted world bank does not match its pinned native snapshot')
    raw=np.loadtxt(directory/'exact-emitted.triangles.csv',delimiter=',',skiprows=1)
    if raw.ndim!=2 or raw.shape[1]!=23 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError('invalid accepted native bone CSV')
    if not np.array_equal(raw[:,0],np.arange(len(raw))):
        raise ValueError('accepted bone ordinal ordering mismatch')
    values=raw[:,5:14].astype(np.float32)
    if not np.array_equal(values.view(np.uint32),raw[:,14:23].astype(np.uint32)):
        raise ValueError('accepted world bone coordinates fail Float32 bit replay')
    ids=raw[:,1:5].astype(np.int64)
    if not np.array_equal(ids.astype(float),raw[:,1:5]):
        raise ValueError('accepted bone typed IDs are not exact integers')
    unique,counts=np.unique(ids[:,:2],axis=0,return_counts=True)
    inventory={(int(r['body']),int(r['stable_bone_id'])):int(r['triangle_count'])
               for r in csv.DictReader(Path(inventory_path).open())}
    if dict(zip(map(tuple,unique.tolist()),counts.tolist()))!=inventory or len(inventory)!=185:
        raise ValueError('accepted bone geometry changed the complete typed-bank inventory')
    for key in unique:
        rows=ids[np.all(ids[:,:2]==key,axis=1),3]
        if not np.array_equal(np.sort(rows),np.arange(len(rows))):
            raise ValueError('accepted bone triangle IDs are incomplete or duplicated')
    bodies={int(r['body_index']):r for r in csv.DictReader((directory/'exact-emitted.bodies.csv').open())}
    if any(int(key[0]) not in bodies for key in unique):
        raise ValueError('accepted world bank lost its physical body identity')
    if len(bodies)!=len(list(csv.DictReader((directory/'exact-emitted.bodies.csv').open()))):
        raise ValueError('accepted world bank repeats a body row')
    for rec in bodies.values():
        for key,value in rec.items():
            if key=='body_index':continue
            expected=1 if key in ('orientation_qw','world_to_body_m00','world_to_body_m11','world_to_body_m22') else (0x3f800000 if key=='orientation_qw_bits_u32' else 0)
            if float(value)!=expected:raise ValueError('accepted world geometry requires exact identity audit frames')
    return values.astype(float).reshape(-1,3,3),ids,bodies

def export_skin(receipt_path, expected_sha, exporter, folder):
    r=read_receipt(receipt_path,expected_sha)
    mapping=r.get("skin_source_mapping",{})
    count=mapping.get("source_vertex_count");first=mapping.get("pack_first_vertex")
    if not isinstance(count,int) or count<4 or not isinstance(first,int) or first<0:
        raise ValueError("accepted skin source interval missing")
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    prefix=folder/"accepted-skin"
    bank=folder/'accepted-world-banks';bank.mkdir()
    cp=subprocess.run([str(exporter),r["accepted_pack_path"],str(first),str(count),str(prefix),
                       str(bank/'exact-emitted.triangles.csv')],
                      capture_output=True,text=True,timeout=30)
    (folder/"export.log").write_text(cp.stdout+cp.stderr)
    if cp.returncode:raise ValueError("native skin export rejected: "+cp.stdout+cp.stderr)
    nodes=list(csv.DictReader(Path(str(prefix)+".nodes.csv").open()))
    faces=list(csv.DictReader(Path(str(prefix)+".faces.csv").open()))
    if len(nodes)!=count or [int(n["source_vertex_row"]) for n in nodes]!=list(range(count)):
        raise ValueError("native skin CSV has invalid source ordering")
    X=np.asarray([[float(n[k+"_m"]) for k in ("x","y","z")] for n in nodes],np.float32)
    F=np.asarray([[int(f[k]) for k in ("node0","node1","node2")] for f in faces],np.int64)
    # Existing NPZ geometry format; unreferenced source rows are preserved so
    # topology acceptance cannot hide inactive or lost surface ancestry.
    candidate=folder/"candidate.npz"
    np.savez_compressed(candidate,points_world_m=X,source_O_faces=F)
    frozen=Path(__file__).parent.parent/'publication-206-stage-001/inputs/bank-export-039/source-preflight'
    body_fields=next(csv.reader((frozen/'exact-emitted.bodies.csv').open()))
    body_rows=[]
    for pose in r['accepted_body_poses']:
        rec={key:'0' for key in body_fields};rec['body_index']=str(pose['body_index'])
        for key in ('orientation_qw','world_to_body_m00','world_to_body_m11','world_to_body_m22'):
            rec[key]='1'
        rec['orientation_qw_bits_u32']=str(0x3f800000)
        body_rows.append(rec)
    with (bank/'exact-emitted.bodies.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=body_fields);w.writeheader();w.writerows(body_rows)
    (folder/'coordinate-policy.json').write_text(json.dumps({
        'pack_sha256':r['pack_file_sha256'],
        'files_sha256':{name:sha(bank/name) for name in ('exact-emitted.triangles.csv','exact-emitted.bodies.csv')},
        'skin':'accepted GPU positions, unchanged source vertex ordering',
        'bones':'same accepted pack vertex data and frozen instance transforms; presentation-only Float32 world evaluation',
        'precision_limit':'CPU and GPU contraction/normalization may differ by rounding; no claim of bit-identical rendered fragment positions',
        'body_frames':'identity because exported triangle coordinates are world coordinates; anatomical body/bone IDs preserved',
        'qualification':'sampled geometry only, not dynamics or continuous motion'},indent=2)+'\n')
    read_world_banks(bank,frozen/'all-bank-containment.csv.banks.csv',r['pack_file_sha256'])
    return candidate,r
