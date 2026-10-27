#!/usr/bin/env python3
"""Exact single-bank screen for the bounded component line search; not a full acceptance audit."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
root=Path('/Users/numi/Work/numi-human-connected-surface-069')
S=root/'D/surface-clearance-546/component-sat-selector-638/outward-boneface-filter-640/root-component-linesearch-002'
code=root/'audit_skin_bone_surfaces_159.py'
spec=importlib.util.spec_from_file_location('bank159',code); m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
helper=m.module('connected_bone_bank_159',m.BANKHELPER); exact=m.exact_module()
bankdir=m.E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
T,ids,gids,bodies,M,off,pos,keys,pins,exact_unused=helper.read_bank_inputs(bankdir,m.E)
body,bone=43,16
mask=(ids[:,0]==body)&(ids[:,1]==bone)
assert mask.any()
out=[]
for alpha in [5,15,30,50,75]:
    path=S/f'alpha-{alpha:03d}/candidate.npz'
    if not path.exists(): continue
    with np.load(path,allow_pickle=False) as z:
        P=np.asarray(z['points_world_m'],dtype=np.float32).astype(np.float64); F=np.asarray(z['source_O_faces'],dtype=np.int64)
    t=time.monotonic(); result=m.intersect_bank((P-pos[body])@M[body].T-off[body],F,T[mask],ids[mask,3],exact)
    out.append({'alpha':alpha/100,'candidate_sha256':sha(path),'seconds':time.monotonic()-t,'bank':{'body':body,'stable_bone_id':bone,**result}})
    print(json.dumps({'alpha':alpha/100,'result':result['result']}),flush=True)
receipt={'status':'ONE_BANK_SCREEN_ONLY','source_sha256':sha(code),'bank':[body,bone],'samples':out,'geometry_or_native_admission':False}
(S/'bank43-screen-001.json').write_text(json.dumps(receipt,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v))+'\n')
