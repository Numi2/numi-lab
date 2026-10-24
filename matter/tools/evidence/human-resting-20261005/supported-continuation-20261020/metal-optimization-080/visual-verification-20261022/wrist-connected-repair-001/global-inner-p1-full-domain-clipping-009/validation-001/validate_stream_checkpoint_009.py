#!/usr/bin/env python3
"""Fail-closed integrity validation for a streamed full-domain P1 clip prefix.

This validates provenance and the processed face prefix represented by checkpoint
parts. It does not imply full-domain completion or physical/native admission.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import numpy as np

EXPECTED_TARGET = 0.341

class ValidationError(ValueError):
    pass

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def _require(ok, msg):
    if not ok:
        raise ValidationError(msg)

def _compact_capacity_rows(bank_id, local_face_ids, flat_face_ids, tet_row_ids, barycentric, capacities, zero_mask, tets):
    """Deduplicate witness rows within each (bank, face, tet) key, preserving bary supports."""
    local_face=np.asarray(local_face_ids,np.int64); face=np.asarray(flat_face_ids,np.int64); tet=np.asarray(tet_row_ids,np.int64)
    bary=np.asarray(barycentric,np.float64); cap=np.asarray(capacities,np.float64); zero=np.asarray(zero_mask,bool)
    _require(local_face.ndim==face.ndim==tet.ndim==cap.ndim==zero.ndim==1 and bary.shape==(len(face),4)
             and len(local_face)==len(face), 'invalid compact-index source rows')
    if not len(face):
        return {'keys':np.empty((0,4),np.uint32),'raw_row_counts':np.empty(0,np.uint32),
                'stored_row_counts':np.empty(0,np.uint32),'barycentric':np.empty((0,4),np.float64),
                'capacities':np.empty(0,np.float64),'zero_mask':np.empty(0,bool),
                'support_node_ids':np.empty((0,4),np.uint32)}
    localgroup=np.column_stack((np.full(len(face),int(bank_id),np.int64),local_face,face,tet))
    order=np.lexsort((bary[:,3],bary[:,2],bary[:,1],bary[:,0],tet,face,local_face))
    group=localgroup[order]; sb=bary[order]; sc=cap[order]; sz=zero[order]
    group_start=np.r_[True,np.any(group[1:]!=group[:-1],axis=1)]
    group_ids=np.cumsum(group_start,dtype=np.int64)-1
    group_count=int(group_start.sum())
    raw_counts=np.bincount(group_ids,minlength=group_count)
    keep=group_start.copy()
    if len(sb)>1:
        same_bary=np.all(sb[1:]==sb[:-1],axis=1)
        keep[1:]|=~same_bary
    # A new key always starts a retained barycentric row; otherwise exact repeated
    # barycentric witnesses are stored once and their raw multiplicity stays in raw_counts.
    stored_group_ids=group_ids[keep]
    stored_counts=np.bincount(stored_group_ids,minlength=group_count)
    unique_bary=sb[keep]; unique_cap=sc[keep]; unique_zero=sz[keep]
    keys=group[group_start]
    nodes=np.asarray(tets,np.int64)[keys[:,3]]
    support=np.where(unique_bary>0,nodes[np.repeat(np.arange(len(keys)),stored_counts)],np.iinfo(np.uint32).max).astype(np.uint32)
    return {'keys':keys.astype(np.uint32),'raw_row_counts':raw_counts.astype(np.uint32),
            'stored_row_counts':stored_counts.astype(np.uint32),'barycentric':unique_bary,
            'capacities':unique_cap,'zero_mask':unique_zero,'support_node_ids':support}

def _write_capacity_witness_index(path, chunks, bank_summary, state, target, points_path):
    """Write a compact, hash-pinned index for zero-capacity face/tet witnesses."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if chunks:
        keys=np.concatenate([c['keys'] for c in chunks],axis=0)
        raw_counts=np.concatenate([c['raw_row_counts'] for c in chunks])
        stored_counts=np.concatenate([c['stored_row_counts'] for c in chunks])
        bary=np.concatenate([c['barycentric'] for c in chunks],axis=0)
        capacities=np.concatenate([c['capacities'] for c in chunks])
        zero=np.concatenate([c['zero_mask'] for c in chunks])
        support=np.concatenate([c['support_node_ids'] for c in chunks],axis=0)
    else:
        keys=np.empty((0,4),np.uint32);raw_counts=stored_counts=np.empty(0,np.uint32)
        bary=np.empty((0,4),np.float64);capacities=np.empty(0,np.float64)
        zero=np.empty(0,bool);support=np.empty((0,4),np.uint32)
    row_offsets=np.r_[0,np.cumsum(stored_counts,dtype=np.int64)]
    bank_ids=np.asarray(keys[:,0],np.int64) if len(keys) else np.empty(0,np.int64)
    tet_ids=np.asarray(keys[:,3],np.int64) if len(keys) else np.empty(0,np.int64)
    bank_tet_offsets=[0];bank_tet_ids=[]
    bank_keys=[]
    for bi in range(len(bank_summary)):
        ids=np.unique(tet_ids[bank_ids==bi]) if len(keys) else np.empty(0,np.int64)
        bank_tet_ids.extend(ids.tolist());bank_tet_offsets.append(len(bank_tet_ids))
        bank_keys.append(int(np.count_nonzero(bank_ids==bi)))
        bank_summary[bi]['affected_key_count']=bank_keys[-1]
        bank_summary[bi]['affected_tet_count']=int(len(ids))
    tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,
        key_bank_ids=keys[:,0] if len(keys) else np.empty(0,np.uint32),
        key_local_face_ids=keys[:,1] if len(keys) else np.empty(0,np.uint32),
        key_flat_face_ids=keys[:,2] if len(keys) else np.empty(0,np.uint32),
        key_tet_ids=keys[:,3] if len(keys) else np.empty(0,np.uint32),
        key_witness_row_offsets=row_offsets,
        key_raw_capacity_row_counts=raw_counts,
        key_stored_witness_row_counts=stored_counts,
        barycentric=bary,free_capacity=capacities,zero_capacity_mask=zero,
        support_node_ids=support,
        bank_affected_tet_offsets=np.asarray(bank_tet_offsets,np.int64),
        bank_affected_tet_ids=np.asarray(bank_tet_ids,np.uint32),
        bank_low_capacity_rows=np.asarray([x['low_capacity_rows'] for x in bank_summary],np.uint64),
        bank_zero_capacity_rows=np.asarray([x['zero_capacity_rows'] for x in bank_summary],np.uint64),
        bank_affected_key_counts=np.asarray(bank_keys,np.uint32))
    os.replace(tmp,path)
    meta={
      'schema':'numi.human.p1-zero-capacity-witness-index.v1',
      'created_utc':dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z'),
      'status':'DIAGNOSTIC_CAPACITY_INDEX_ONLY',
      'source_checkpoint_sha256':sha256_file(Path(state['_checkpoint_path'])),
      'mesh_path':str(points_path),'mesh_nodes':int(state['mesh_nodes']),'mesh_tets':int(state['mesh_tets']),
      'processed_face_prefix':[0,int(state['faces_processed'])],
      'input_face_count':int(state.get('_input_face_count',state['faces_processed'])),
      'target_free_capacity':float(target),
      'key_columns':['bank_id','local_face_id','flat_face_id','tet_row_id'],
      'witness_arrays':{
        'key_witness_row_offsets':'CSR rows into barycentric/free_capacity/zero_capacity_mask/support_node_ids',
        'support_node_ids':'four tet-node IDs in tet order; UINT32_MAX where barycentric weight is zero',
        'bank_affected_tet_offsets':'CSR into bank_affected_tet_ids, one segment per inventory bank',
      },
      'unique_key_count':int(len(keys)),'stored_unique_barycentric_rows':int(len(bary)),
      'raw_zero_capacity_rows':int(sum(x['zero_capacity_rows'] for x in bank_summary)),
      'bank_summaries':bank_summary,
      'npz_path':str(path),'npz_sha256':sha256_file(path),'npz_bytes':path.stat().st_size,
      'scope_limit':'Index locates construction witnesses only; it is not a complete LP/contact constraint set or a physical/native admission result.',
    }
    meta_path=path.with_suffix('.json'); meta_path.write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
    return meta_path,meta

def validate_parts_and_prefix(state, inventory, tets, fixed, parts_dir, points=None, target=EXPECTED_TARGET, verify_part_sha=True, index_output=None, index_state=None, mesh_path=None):
    """Validate contiguous domain coverage and every saved capacity row.

    `inventory` is the ordered list of bank metadata records from inventory.json.
    `tets` and `fixed` are the exact source mesh tetrahedra and fixed-node mask.
    Every chunk contributes its full processed face interval, including faces
    with no saved intersection polygons.
    """
    tets=np.asarray(tets)
    fixed=np.asarray(fixed,bool)
    points=None if points is None else np.asarray(points)
    _require(tets.ndim==2 and tets.shape[1]==4 and np.issubdtype(tets.dtype,np.integer), 'invalid tet domain')
    _require(fixed.ndim==1 and (not len(tets) or int(tets.max())<len(fixed)), 'invalid fixed-node domain')
    if points is not None:
        _require(points.ndim==2 and points.shape==(len(fixed),3) and np.isfinite(points).all(), 'invalid world-point domain')
    _require(0<target<1 and np.isfinite(target), 'invalid target')
    _require(isinstance(inventory,list) and inventory, 'empty or invalid inventory')
    counts=[]
    offsets=[0]
    for i,m in enumerate(inventory):
        n=int(m['triangles'])
        _require(n>=0, f'inventory bank {i} has negative face count')
        counts.append(n); offsets.append(offsets[-1]+n)
    offsets=np.asarray(offsets,np.int64)
    total_faces=int(offsets[-1])
    parts=state.get('parts')
    _require(isinstance(parts,list), 'checkpoint parts missing')
    parts_dir=Path(parts_dir)
    listed=[]
    flat_cursor=0
    bank_cursor=0
    local_cursor=0
    sum_faces=sum_polygons=sum_vertices=sum_bad=sum_zero=0
    polygon_hit_faces=0
    index_chunks=[]
    bank_summary=[{'bank_id':i,'bank_key':m.get('key'),'stable_id':m.get('stable_id'),'kind':m.get('kind'),
                   'layer':m.get('layer'),'body_indices':m.get('body_indices'),
                   'low_capacity_rows':0,'zero_capacity_rows':0} for i,m in enumerate(inventory)]
    part_names=set()
    # Empty banks are allowed in the inventory but require no chunk; advance over them.
    def advance_empty(bi):
        while bi<len(counts) and counts[bi]==0:
            bi+=1
        return bi
    bank_cursor=advance_empty(bank_cursor)
    for pi,meta in enumerate(parts):
        expected_name=f'part-{pi:05d}.npz'
        name=meta.get('file')
        _require(name==expected_name, f'part sequence/name mismatch at index {pi}: {name!r}')
        _require(name not in part_names, f'duplicate part file {name}')
        part_names.add(name); path=parts_dir/name
        _require(path.is_file(), f'missing part file {name}')
        if verify_part_sha:
            _require(sha256_file(path)==meta.get('sha256'), f'part SHA mismatch: {name}')
        _require(bank_cursor<len(counts), f'part {name} extends beyond inventory')
        bi=int(meta.get('bank_id',-1))
        _require(bi==bank_cursor, f'part {name} bank gap/order: expected {bank_cursor}, got {bi}')
        m=inventory[bi]
        for field in ('key','kind','stable_id','layer','body_indices'):
            part_field={'key':'bank_key','kind':'kind','stable_id':'stable_id','layer':'layer','body_indices':'body_indices'}[field]
            _require(meta.get(part_field)==m.get(field), f'part {name} typed identity mismatch for {part_field}')
        lr=meta.get('local_face_range'); fr=meta.get('flat_face_range')
        _require(isinstance(lr,list) and len(lr)==2 and isinstance(fr,list) and len(fr)==2, f'part {name} missing face intervals')
        lo,hi=map(int,lr)
        _require(lo==local_cursor, f'face-domain gap/overlap before {name}: expected local {local_cursor}, got {lo}')
        _require(0<=lo<hi<=counts[bi], f'invalid local face range in {name}: {lr}')
        _require(fr==[int(offsets[bi]+lo),int(offsets[bi]+hi)], f'flat/local face mapping mismatch in {name}')
        _require(int(meta.get('faces_processed',-1))==hi-lo, f'processed-face count mismatch in {name}')
        _require(hi-lo<=1024, f'chunk larger than configured face batch in {name}')
        with np.load(path,allow_pickle=False) as z:
            needed={'tet_ids','face_ids','bank_ids','poly_offsets','polygon_barycentric',
                    'insufficient_capacity_polygon_vertex_rows','zero_capacity_polygon_vertex_rows'}
            _require(set(z.files)==needed, f'part schema mismatch in {name}: {sorted(z.files)}')
            tids=np.asarray(z['tet_ids']); face_ids=np.asarray(z['face_ids']); bank_ids=np.asarray(z['bank_ids'])
            offsets_p=np.asarray(z['poly_offsets']); bary=np.asarray(z['polygon_barycentric'])
            bad=np.asarray(z['insufficient_capacity_polygon_vertex_rows']); zero=np.asarray(z['zero_capacity_polygon_vertex_rows'])
        pcount=int(meta.get('polygons',-1)); vcount=int(meta.get('polygon_vertices',-1))
        _require(tids.shape==(pcount,) and face_ids.shape==(pcount,) and bank_ids.shape==(pcount,), f'polygon vector count mismatch in {name}')
        _require(np.issubdtype(tids.dtype,np.integer) and np.issubdtype(face_ids.dtype,np.integer) and np.issubdtype(bank_ids.dtype,np.integer), f'noninteger polygon IDs in {name}')
        _require(pcount>=0 and vcount>=0 and offsets_p.shape==(pcount+1,), f'invalid polygon offsets in {name}')
        _require(offsets_p.dtype.kind in 'iu' and offsets_p[0]==0 and offsets_p[-1]==vcount and np.all(np.diff(offsets_p)>0) if pcount else offsets_p.shape==(1,) and offsets_p[0]==0 and vcount==0, f'bad polygon offset domain in {name}')
        _require(bary.shape==(vcount,4) and np.issubdtype(bary.dtype,np.floating) and np.isfinite(bary).all(), f'bad barycentric array in {name}')
        _require(not len(bary) or (float(bary.min())>=-2e-12 and float(np.max(np.abs(bary.sum(axis=1)-1)))<=2e-12), f'invalid barycentric rows in {name}')
        _require(not len(tids) or (int(tids.min())>=0 and int(tids.max())<len(tets)), f'tet ID outside domain in {name}')
        _require(not len(face_ids) or (int(face_ids.min())>=fr[0] and int(face_ids.max())<fr[1]), f'face ID outside processed chunk in {name}')
        _require(not len(bank_ids) or np.all(bank_ids==bi), f'polygon bank ID mismatch in {name}')
        _require(bad.ndim==1 and bad.dtype.kind in 'iu' and zero.ndim==1 and zero.dtype.kind in 'iu', f'bad capacity row index arrays in {name}')
        _require(not len(bad) or (int(bad.min())>=0 and int(bad.max())<vcount), f'orphan insufficient-capacity row in {name}')
        _require(not len(zero) or (int(zero.min())>=0 and int(zero.max())<vcount), f'orphan zero-capacity row in {name}')
        _require(len(np.unique(bad))==len(bad) and len(np.unique(zero))==len(zero), f'duplicate capacity row index in {name}')
        _require(set(map(int,zero)).issubset(set(map(int,bad))), f'zero-capacity rows not included in insufficient rows in {name}')
        if pcount:
            row_poly=np.repeat(np.arange(pcount,dtype=np.int64),np.diff(offsets_p))
            row_tet=tids[row_poly].astype(np.int64,copy=False)
            row_fixed=fixed[tets[row_tet]]
            cap=np.sum(bary*~row_fixed,axis=1)
            expected_bad=np.flatnonzero(cap<target-1e-12)
            expected_zero=np.flatnonzero(cap<=1e-12)
        else:
            row_poly=np.empty(0,np.int64); row_tet=np.empty(0,np.int64);cap=np.empty(0,np.float64)
            expected_bad=expected_zero=np.empty(0,np.int64)
        _require(np.array_equal(bad.astype(np.int64),expected_bad), f'insufficient-capacity rows do not match barycentric capacities in {name}')
        _require(np.array_equal(zero.astype(np.int64),expected_zero), f'zero-capacity rows do not match barycentric capacities in {name}')
        _require(int(meta.get('insufficient_capacity_rows',-1))==len(expected_bad), f'insufficient-capacity count mismatch in {name}')
        _require(int(meta.get('zero_capacity_rows',-1))==len(expected_zero), f'zero-capacity count mismatch in {name}')
        _require(int(meta.get('polygon_vertices',-1))==len(bary), f'polygon-vertex metadata mismatch in {name}')
        _require(int(meta.get('aabb_candidates',-1))>=pcount, f'AABB candidate count below emitted polygon count in {name}')
        bank_summary[bi]['low_capacity_rows']+=int(len(expected_bad))
        bank_summary[bi]['zero_capacity_rows']+=int(len(expected_zero))
        if index_output is not None and len(expected_zero):
            zero_poly=row_poly[expected_zero]
            zero_flat_face=face_ids[zero_poly].astype(np.int64,copy=False)
            zero_tet=row_tet[expected_zero]
            zero_bary=bary[expected_zero]
            zero_cap=cap[expected_zero]
            zero_local_face=zero_flat_face-int(offsets[bi])
            zero_mask=zero_cap<=1e-12
            idx=_compact_capacity_rows(bi,zero_local_face,zero_flat_face,zero_tet,zero_bary,zero_cap,zero_mask,tets)
            if len(idx['keys']):
                index_chunks.append(idx)
        polygon_hit_faces+=len(np.unique(face_ids))
        # Verify retained witness rows against typed identity, barycentric capacity and world point.
        for w in meta.get('first_witnesses',[]):
            _require(int(w.get('bank_id',-1))==bi and w.get('kind')==m.get('kind') and int(w.get('stable_id',-1))==int(m.get('stable_id')) and int(w.get('layer',-1))==int(m.get('layer')), f'witness typed identity mismatch in {name}')
            local_face=int(w.get('local_face',-1)); tetrow=int(w.get('tet_row',-1))
            _require(lo<=local_face<hi and 0<=tetrow<len(tets), f'witness face/tet outside domain in {name}')
            _require(np.array_equal(np.asarray(w.get('tet_nodes'),np.int64),tets[tetrow]), f'witness tet-node identity mismatch in {name}')
            candidates=np.flatnonzero((face_ids==offsets[bi]+local_face)&(tids==tetrow))
            _require(len(candidates)>0, f'witness has no polygon row in {name}')
            wb=np.asarray(w.get('barycentric'),np.float64)
            matched=False
            for q in candidates:
                a,b=int(offsets_p[q]),int(offsets_p[q+1])
                matches=np.flatnonzero(np.all(np.isclose(bary[a:b],wb[None,:],rtol=0,atol=2e-12),axis=1))
                for k in matches:
                    ri=a+int(k)
                    if abs(float(cap[ri])-float(w.get('capacity',np.nan)))<=2e-12:
                        if points is not None:
                            expected_world=wb@points[tets[tetrow]].astype(np.float64)
                            actual_world=np.asarray(w.get('world_m'),np.float64)
                            _require(actual_world.shape==(3,) and np.isfinite(actual_world).all() and np.max(np.abs(expected_world-actual_world))<=2e-12,
                                     f'witness world point mismatch in {name}')
                        support=np.flatnonzero(wb>0)
                        fixed_only=bool(np.all(fixed[tets[tetrow][support]]))
                        _require(bool(w.get('positive_support_on_fixed_only'))==fixed_only, f'witness fixed-only support mismatch in {name}')
                        matched=True;break
                if matched:break
            _require(matched, f'witness barycentric row is orphaned in {name}')
        # The polygons themselves cover only hits; the face interval above is authoritative
        # for work performed, so zero-hit faces are not inferred from max(face_ids).
        sum_faces+=hi-lo;sum_polygons+=pcount;sum_vertices+=vcount;sum_bad+=len(expected_bad);sum_zero+=len(expected_zero)
        flat_cursor=fr[1];local_cursor=hi
        if hi==counts[bi]:
            bank_cursor=advance_empty(bi+1);local_cursor=0
    # No uncheckpointed part files may masquerade as covered work or capacity output.
    present={p.name for p in parts_dir.glob('part-*.npz')}
    _require(present==part_names, f'orphan/unlisted part files: {sorted(present-part_names)[:5]}')
    _require(sum_faces==flat_cursor, 'face interval sum does not equal contiguous prefix length')
    _require(int(state.get('faces_processed',-1))==sum_faces, 'checkpoint processed-face count mismatch')
    _require(int(state.get('polygons',-1))==sum_polygons, 'checkpoint polygon count mismatch')
    _require(int(state.get('insufficient_capacity_rows',-1))==sum_bad, 'checkpoint insufficient-capacity row count mismatch')
    _require(int(state.get('zero_capacity_rows',-1))==sum_zero, 'checkpoint zero-capacity row count mismatch')
    cursor_bank=int(state.get('next_bank',-1));cursor_local=int(state.get('next_local_face',-1))
    _require(0<=cursor_bank<=len(counts), 'checkpoint cursor bank outside inventory')
    if cursor_bank==len(counts):
        _require(cursor_local==0, 'terminal checkpoint cursor has nonzero local face')
        expected_flat=total_faces
    else:
        _require(0<=cursor_local<=counts[cursor_bank], 'checkpoint local cursor outside bank')
        expected_flat=int(offsets[cursor_bank]+cursor_local)
    _require(expected_flat==flat_cursor, 'checkpoint cursor does not match processed prefix')
    _require(bool(state.get('complete'))==(flat_cursor==total_faces), 'complete flag does not match full input-domain coverage')
    if flat_cursor<total_faces:
        _require(state.get('native_admission') is False and state.get('physical_admission') is False, 'partial checkpoint improperly claims admission')
    _require(sum_vertices==sum(int(p['polygon_vertices']) for p in parts), 'barycentric row total mismatch')
    # A zero-hit face is any processed face without a saved clipped polygon. It still lies in the verified input prefix.
    report={
      'status':'PASS_PARTIAL_PREFIX_INTEGRITY' if flat_cursor<total_faces else 'PASS_COMPLETE_DOMAIN_INTEGRITY',
      'processed_face_prefix':[0,int(flat_cursor)],'processed_faces':int(flat_cursor),'total_input_faces':total_faces,
      'coverage_fraction':float(flat_cursor/total_faces),'complete':bool(state.get('complete')),
      'next_bank':cursor_bank,'next_local_face':cursor_local,'parts':len(parts),
      'polygons':sum_polygons,'polygon_barycentric_rows':sum_vertices,
      'insufficient_capacity_rows':sum_bad,'zero_capacity_rows':sum_zero,
      'distinct_hit_faces_in_prefix':polygon_hit_faces,
      'zero_hit_faces_in_processed_prefix':int(flat_cursor-polygon_hit_faces),
      'target_free_capacity':float(target),
      'native_admission':bool(state.get('native_admission')),'physical_admission':bool(state.get('physical_admission')),
      'pruning':state.get('pruning'),
    }
    if index_output is not None:
        if index_state is None or mesh_path is None:
            raise ValidationError('capacity index requested without checkpoint state/mesh path')
        index_state=dict(index_state); index_state['_checkpoint_path']=str(Path(parts_dir).parent/'checkpoint.json')
        index_state['_input_face_count']=total_faces
        index_path=Path(index_output)
        meta_path,index_meta=_write_capacity_witness_index(index_path,index_chunks,bank_summary,index_state,target,mesh_path)
        report['capacity_witness_index']={'npz_path':str(index_path),'npz_sha256':sha256_file(index_path),
                                          'json_path':str(meta_path),'json_sha256':sha256_file(meta_path),
                                          'unique_key_count':index_meta['unique_key_count'],
                                          'stored_witness_rows':index_meta['stored_unique_barycentric_rows']}
    return report

def validate_run(run_dir: Path, inventory_path: Path, mesh_path: Path, bank_path: Path, clip_path: Path,
                 runner_path: Path, broadphase_path: Path, target=EXPECTED_TARGET, verify_input_sha=True, index_output=None):
    run_dir=Path(run_dir); cp_path=run_dir/'checkpoint.json'; parts_dir=run_dir/'constraint-parts'
    state=json.loads(cp_path.read_text())
    if verify_input_sha:
        for pstr,digest in state.get('inputs',{}).items():
            p=Path(pstr)
            _require(p.is_file(),f'missing pinned run input: {p}')
            _require(sha256_file(p)==digest,f'input SHA mismatch: {p}')
    with np.load(mesh_path,allow_pickle=False) as z:
        points=np.asarray(z['points_world_m'])
        tets=np.asarray(z['tetrahedra'])
        fixed=np.zeros(len(points),bool)
        outer=np.asarray(z['physical_outer_node_ids'],np.int64)
        _require(not len(outer) or (outer.min()>=0 and outer.max()<len(points)),'fixed outer-node ID outside mesh domain')
        fixed[outer]=True
    _require(len(points)==int(state.get('mesh_nodes',-1)) and len(tets)==int(state.get('mesh_tets',-1)),'checkpoint mesh domain count mismatch')
    inventory=json.loads(Path(inventory_path).read_text())['banks']
    _require(len(inventory)==689,'unexpected typed bank inventory count')
    report=validate_parts_and_prefix(state,inventory,tets,fixed,parts_dir,points=points,target=target,
                                     index_output=index_output,index_state=state,mesh_path=mesh_path)
    report.update({
      'checkpoint_path':str(cp_path),'checkpoint_sha256':sha256_file(cp_path),
      'input_sha256':{str(p):sha256_file(Path(p)) for p in state.get('inputs',{})},
      'mesh_nodes':len(points),'mesh_tets':len(tets),'inventory_banks':len(inventory),
      'input_face_count':int(sum(int(x['triangles']) for x in inventory)),
    })
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True,help='global-inner-p1-full-domain-clipping-009 directory')
    ap.add_argument('--run-name',default='run-001',help='checkpoint subdirectory name')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--capacity-index',type=Path,help='optional compact zero-capacity support witness index output NPZ')
    ap.add_argument('--skip-input-sha',action='store_true')
    args=ap.parse_args()
    E=args.root.parent
    report=validate_run(args.root/args.run_name,
      E/'registered-surrounding-tissue-native013-191/inventory.json',
      E/'global-inner-p1-active-layer-refinement-008/candidate-with-transfer-partial-field.npz',
      E/'registered-surrounding-tissue-native013-191/registered-tissue-banks.npz',
      E/'robust-p1-construction-clip-owner-302/tools/human_skin_p1_obstacles.py',
      args.root/'run_stream_clip_009.py',args.root/'stream_clip_009.py',
      verify_input_sha=not args.skip_input_sha,index_output=args.capacity_index)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    report['report_path']=str(args.output);report['report_sha256']=sha256_file(args.output)
    print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__':main()
