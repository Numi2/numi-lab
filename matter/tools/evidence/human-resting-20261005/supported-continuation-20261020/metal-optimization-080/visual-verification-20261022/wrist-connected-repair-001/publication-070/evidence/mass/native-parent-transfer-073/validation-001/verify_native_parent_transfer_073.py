#!/usr/bin/env python3
# Independent arithmetic and CSR checks for frozen native-parent transfer 073.
from pathlib import Path
import hashlib, json
import numpy as np
HERE=Path(__file__).resolve().parent
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
O=E/"connected-skin-parent-mass-transfer-067/native-parent-transfer-073"
P=O/"native-parent-transfer-073.npz"
a=np.load(P,allow_pickle=False)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ind=a["child_parent_indptr"]; par=a["child_native_parent_rows"]; ov=a["overlap_native_reference_volume_m3"]
child_vol=a["child_native_reference_volume_m3"]; parent_geom=a["native_parent_native_f32_geometry_volume_m3"]
childfrac=a["child_parent_fraction_overlap_over_child_volume"]; parentfrac=a["parent_mass_fraction_overlap_over_native_parent_geometry_volume"]
child_mass=a["child_transferred_mass_kg"]; parent_mass=a["native_parent_compiled_mass_kg"]
cmat=a["child_native_material_ids"]; pmat=a["native_parent_material_ids"]
dens=a["child_effective_reference_density_kg_m3"]; cind=a["native_parent_corner_source_contribution_row_ids"]; cbody=a["native_parent_corner_frame_owner_body_ids"]; clocal=a["native_parent_corner_donor_local_points_m"]; corner_mass=a["native_parent_corner_lumped_cell_mass_kg"]
assert ind.dtype==np.int64 and ind.shape==(len(child_vol)+1,) and ind[0]==0 and ind[-1]==len(par) and np.all(np.diff(ind)>0)
assert len(ov)==len(par)==len(childfrac)==len(parentfrac) and np.all(ov>0) and np.all(np.isfinite(ov))
assert np.all((par>=0)&(par<len(parent_mass)))
child_sum=np.add.reduceat(ov,ind[:-1]); child_rel=(child_sum-child_vol)/child_vol
parent_sum=np.bincount(par,weights=ov,minlength=len(parent_geom)); parent_rel=(parent_sum-parent_geom)/parent_geom
childfrac_sum=np.add.reduceat(childfrac,ind[:-1]); parentfrac_sum=np.bincount(par,weights=parentfrac,minlength=len(parent_geom))
assert np.max(np.abs(child_rel))<1e-10, float(np.max(np.abs(child_rel)))
assert np.max(np.abs(parent_rel))<1e-10, float(np.max(np.abs(parent_rel)))
assert np.max(np.abs(childfrac_sum-1))<1e-10 and np.max(np.abs(parentfrac_sum-1))<1e-10
for ci in range(len(child_vol)):
    lo,hi=ind[ci:ci+2]
    assert np.all(pmat[par[lo:hi]]==cmat[ci]), (ci,cmat[ci],np.unique(pmat[par[lo:hi]]))
recalc=np.zeros_like(child_mass)
for ci in range(len(child_vol)):
    lo,hi=ind[ci:ci+2]
    recalc[ci]=np.dot(parent_mass[par[lo:hi]],parentfrac[lo:hi])
assert np.max(np.abs(recalc-child_mass))<1e-14, float(np.max(np.abs(recalc-child_mass)))
assert np.max(np.abs(corner_mass-parent_mass[:,None]/4))==0.0
assert cind.shape==cbody.shape==(len(parent_mass),4) and clocal.shape==(len(parent_mass),4,3)
assert np.all(np.isfinite(dens))
bounds={15:(900.,1100.),16:(850.,950.)}
for m,(lo,hi) in bounds.items():
    d=dens[cmat==m]; assert len(d)>0 and d.min()>=lo and d.max()<=hi
material={str(m):{"parent_mass_kg":float(parent_mass[pmat==m].sum()),"child_mass_kg":float(child_mass[cmat==m].sum())} for m in (15,16)}
assert all(abs(x["parent_mass_kg"]-x["child_mass_kg"])<1e-10 for x in material.values())
report={"status":"PASS_ARITHMETIC_AND_CSR_ONLY","map_sha256":sha(P),"source_export_sha256":{"compiled_nodes":"26c607f9835241ef5d274169e4ac2aeae9c0f6ab7ad94a06b307ca25844862c9","compiled_tets":"a7dc2d97713e43ed2bfae2aa805bb5281fdf2361108cf208df4cbdea10084461","premerge_contributions":"75bd67cc6f7aeab9dccbb73da29220fcd35896b217042e8ce9c8181d89ca7c6f"},"children":int(len(child_vol)),"parents":int(len(parent_mass)),"nonzeros":int(len(par)),"max_child_coverage_relative_error":float(np.max(np.abs(child_rel))),"max_parent_coverage_relative_error":float(np.max(np.abs(parent_rel))),"max_parent_fraction_closure_error":float(np.max(np.abs(parentfrac_sum-1))),"max_child_mass_recompute_error_kg":float(np.max(np.abs(recalc-child_mass))),"per_material_mass_kg":material,"per_material_density_bounds_pass":True,"source_provenance_rows":int(cind.size),"admission":"No current-geometry, bone-clearance, native-emission, or physical/native qualification is implied."}
out=HERE/"verification.json"; out.write_text(json.dumps(report,indent=2,sort_keys=True)+chr(10))
print(json.dumps(report,indent=2,sort_keys=True))
