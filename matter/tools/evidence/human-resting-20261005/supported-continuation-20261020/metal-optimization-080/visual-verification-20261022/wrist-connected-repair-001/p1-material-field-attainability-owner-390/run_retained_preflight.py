"""Reproduce the retained failed-field target preflight without cutting or simulation."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"tools"))
from human_skin_p1_material_volume import check_material_volume_targets

def run():
    e=HERE.parent
    sources={
      "mesh":(e/"global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/candidate.npz",
              "e70b3e08aa385c8a321ebb0849257ec4afa04d13050e26fc28647d36146d26ba"),
      "field":(e/"global-connected-inner-field-293-cavity-remesh-014-independent-volume-001/field-transfer-preflight.npz",
               "3dfab2cdb6b4d0ee0f170ce9bd3a2ee30fd7cd064e92542c58b600a61653336b")}
    for name,(path,pin) in sources.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=pin:
            raise ValueError(name+" source hash mismatch")
    z=np.load(sources["mesh"][0],allow_pickle=False)
    f=np.load(sources["field"][0],allow_pickle=False)
    t=time.monotonic()
    result=check_material_volume_targets(z["points_world_m"],z["tetrahedra"],f["phi"],
       {"skin":.003452570641581807,"skin_plus_fat":.015104913880526613})
    result["elapsed_s"]=time.monotonic()-t
    result["inputs"]={str(path):pin for path,pin in sources.values()}
    result["scope"]="Only positive Float32 volume cells and transferred scalar values used. Boundary ancestry and native admission are not certified. The cache is an LP initializer, not a solved thickness field."
    return result

if __name__=="__main__":print(json.dumps(run(),indent=2))
