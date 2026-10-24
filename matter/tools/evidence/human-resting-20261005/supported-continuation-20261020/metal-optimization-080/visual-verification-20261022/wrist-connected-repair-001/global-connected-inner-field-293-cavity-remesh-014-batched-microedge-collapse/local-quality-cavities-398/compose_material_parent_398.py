from pathlib import Path
import numpy as np,json,hashlib
D=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse')
O=D/'local-quality-cavities-398';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=D/'p1-interior-face-support-395/refined-background-checkpoint.npz';a=np.load(p,allow_pickle=False)
b=np.load(O/'candidate-with-transfer.npz',allow_pickle=False)
r=json.loads((O/'transfer-report.json').read_text())
assert r['geometric_transfer_gate_pass'] and sha(p)==r['input395_sha256'] and sha(O/'candidate-with-transfer.npz')==r['output_sha256']
parent=a['child_source_tetrahedron_rows'].astype(np.int64);frac=a['child_volume_fraction_of_parent'].astype(float)
np014=164176
sums=np.bincount(parent,weights=frac,minlength=np014)
assert len(sums)==np014 and np.max(abs(sums-1))<2e-12
keep=b['unchanged_395_tet_rows'];localoffs=b['overlap_child_offsets'];pc395=b['overlap_parent_395_tet_row'];pct=b['overlap_parent_conservative_fraction']
nunchanged=len(keep);nlocal=len(localoffs)-1
oldChildRows=pc395
row=np.r_[np.arange(nunchanged),np.repeat(np.arange(nlocal)+nunchanged,np.diff(localoffs))]
col=np.r_[parent[keep],parent[oldChildRows]]
value=np.r_[frac[keep],pct*frac[oldChildRows]]
# Coalesce multiple old395 children from the same014 parent into each398 row.
order=np.lexsort((col,row));row=row[order];col=col[order];value=value[order]
starts=np.r_[0,np.flatnonzero((row[1:]!=row[:-1])|(col[1:]!=col[:-1]))+1]
row=row[starts];col=col[starts];value=np.add.reduceat(value,starts)
offs=np.r_[0,np.cumsum(np.bincount(row,minlength=nunchanged+nlocal))]
columnsums=np.bincount(col,weights=value,minlength=np014)
rng=np.random.default_rng(398);mass=rng.uniform(.01,2,np014)
childmass=np.bincount(row,weights=value*mass[col],minlength=len(b['tetrahedra']))
res=abs(childmass.sum()-mass.sum())/mass.sum()
out=O/'material-parent-014-conservative-map.npz'
np.savez_compressed(out,overlap_child_offsets=offs,overlap_parent_original_input_tet_row=col,overlap_parent_conservative_fraction=value)
rr={'input395_sha256':sha(p),'input398_with_transfer_sha256':sha(O/'candidate-with-transfer.npz'),'old014_tet_count':np014,'new398_tet_count':len(b['tetrahedra']),'nonzeros':len(value),'column_sum_max_abs_error':float(np.max(abs(columnsums-1))),'deterministic_synthetic_mass_relative_error':float(res),'map_sha256':sha(out),'method':'compose395 parent-conservative fractions with398 local overlap parent-conservative fractions, coalesced by actual014 parent identity','limitations':['No per-tet physical kg input exists in014/395; synthetic conservation checks transfer only.','Rows correspond to398 geometry; independent physical material partition and native compiler admission remain required.'],'pass':bool(np.max(abs(columnsums-1))<2e-12 and res<1e-12),'native_admission':False}
(O/'material-parent-014-conservative-report.json').write_text(json.dumps(rr,indent=2)+'\n');print(json.dumps(rr));assert rr['pass']

