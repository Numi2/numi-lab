from pathlib import Path
import importlib.util,sys,json,hashlib
D=Path("/Users/numi/Work/numi-human-connected-surface-069")
src=D/"D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
helper=D/"D/D/whole-face-objective-435/whole_face_objective_435.py"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(src)=="5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2"
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
if __name__=="__main__":
    pins={str(p):sha(p)for p in [src,helper,Path(__file__)]}
    sys.path.insert(0,str(src.parent))
    owner=load("fitter408",src);variant=load("objective435",helper)
    owner.build_clearance_rows=variant.replacement_for(owner.build_clearance_rows)
    owner.main()
    out=Path(sys.argv[sys.argv.index("--output")+1]);rp=out/"report.json"
    report=json.loads(rp.read_text());report["objective_variant_435"]={"name":"one_exit_plane_per_face_bone_positive_corner_deficit","pins":pins,"certifies_clearance":False,"native_admission":False}
    rp.write_text(json.dumps(report,indent=2,sort_keys=True)+chr(10))
    assert all(sha(Path(p))==v for p,v in pins.items())
