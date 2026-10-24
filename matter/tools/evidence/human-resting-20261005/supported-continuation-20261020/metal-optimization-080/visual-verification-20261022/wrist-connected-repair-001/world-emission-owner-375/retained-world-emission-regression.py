from pathlib import Path
import importlib.util,json,numpy as np,hashlib,shutil
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001');D=E/'world-emission-owner-375'
s=importlib.util.spec_from_file_location('g',D/'tools/human_skin_patch_geometry.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
out=[]
for name,expected in [('left-connected-expanded-forearm-envelope-369',False),('left-world-float32-conditioned-envelope-373',True)]:
 p=E/name;field=np.load(p/'conditioned-field.npz');mesh=np.load(p/'extracted-interface.npz')
 ref=mesh['vertices_local_float32'].astype(float)@field['basis_world_rows']+field['origin_world_m']
 try:g.validate_float32_surface_emission(ref,mesh['vertices_current_world_m'],mesh['faces']);passed=True;error=None
 except ValueError as e:passed=False;error=str(e)
 assert passed==expected,(name,passed,error)
 out.append(dict(name=name,expected_pass=expected,passed=passed,error=error,mesh_sha256=hashlib.sha256((p/'extracted-interface.npz').read_bytes()).hexdigest()))
(D/'retained-world-emission-regression.json').write_text(json.dumps({'cases':out,'physical_admission':False},indent=2)+'\n')
print(json.dumps(out))

