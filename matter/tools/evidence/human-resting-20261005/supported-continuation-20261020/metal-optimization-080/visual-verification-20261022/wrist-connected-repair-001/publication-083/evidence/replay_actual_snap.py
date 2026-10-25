from pathlib import Path
import sys,json,time,hashlib,numpy as np
from human_skin_fat_layer_reconstruction import snap_shared_scalar_interface_vertices
D=Path(sys.argv[1]);O=Path(sys.argv[2]);O.mkdir(exist_ok=False)
I=D/"deep-positive-input.npz";L=D/"deep-positive-snap-ledger.npz"
z=np.load(I);l=np.load(L);start=time.monotonic()
phi,stats,ledger=snap_shared_scalar_interface_vertices(z["points_world_m"],z["tetrahedra_zero_based"],l["original_phi_m"],l["local_hskin_m"],fraction=float(sys.argv[3]))
assert np.array_equal(phi,l["snapped_phi_m"])
assert np.array_equal(phi,z["scalar_values_m"])
for key in ledger:
 assert np.array_equal(ledger[key],l[key]),key
r={"status":"EXACT_ACTUAL_INPUT_AND_LEDGER_PARITY","input_sha256":hashlib.sha256(I.read_bytes()).hexdigest(),"ledger_sha256":hashlib.sha256(L.read_bytes()).hexdigest(),"owner_sha256":hashlib.sha256(Path(__file__).with_name("human_skin_fat_layer_reconstruction.py").read_bytes()).hexdigest(),"statistics":stats,"wall_seconds":time.monotonic()-start,"native_admission":False}
(O/"report.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r))
