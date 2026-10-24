from pathlib import Path
import sys,json,time,hashlib,traceback
D=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/source-route-weighting-owner-385')
sys.path.insert(0,"/Users/n/numi-human-anatomy-finish-1378/src")
from numilab_human import model
rp=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/brachio109-local810-existing-full-parent-source-853b/assets/candidate-current-source.json")
reg=json.loads(rp.read_text());src=reg["source"]["bodyparts"]
anatomy=dict(source_id=src["id"],version=src["version"],archives=src["archives"])
output=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/working-source-integration-397/compiler-eight-bank-working-005');output.mkdir(exist_ok=False)
start=time.monotonic()
try:
    manifest=model.bodyparts_myosim_fullbody_soft_tissue_visual_payload(
        D/"Sources",anatomy,rp,
        Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/current-manifest-recovery-004/recovered-artifact"),
        output,stable_id_subset={131,132,133,134,137,138,141,142})
except Exception:
    (output/"failure.txt").write_text(traceback.format_exc())
    raise
(output/"execution.json").write_text(json.dumps(dict(elapsed_s=time.monotonic()-start,
    model_sha256=hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest(),
    registration_sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),
    payload=manifest.get("payload"),physical_admission=False),indent=2)+"\n")
print(json.dumps(dict(elapsed_s=time.monotonic()-start,payload=manifest.get("payload"))),flush=True)
