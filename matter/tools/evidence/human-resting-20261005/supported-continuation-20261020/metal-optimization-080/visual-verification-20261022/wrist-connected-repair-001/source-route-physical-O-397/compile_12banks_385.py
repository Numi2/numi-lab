from pathlib import Path
import hashlib, json, sys, time

E = Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
OWNER = E / 'source-route-weighting-owner-385'
OUT = E / 'source-route-physical-O-397' / 'compiler-12bank-001'
REG = Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/brachio109-local810-existing-full-parent-source-853b/assets/candidate-current-source.json')
MYOSIM = Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/current-manifest-recovery-004/recovered-artifact')
BASE = Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/muscle158-precision-002/composed-receipt/composed/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue')
TARGET = {75,79,81,83,85,89,93,97,103,120,122,132}
EXPECTED_MODEL = '5bd314fa7ceaebc6c0e72a0115befdda7a19e11b081bc2009cd2a1e94ce01483'
EXPECTED_BASE = '8874b3fede93825d2e4ddb248231dc9d2c598e0155f8473c624fcb935d6b1042'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
model_path = OWNER / 'src/numilab_human/model.py'
assert sha(model_path) == EXPECTED_MODEL, '385 compiler model pin mismatch'
assert sha(BASE) == EXPECTED_BASE, 'raw NHTISS baseline pin mismatch'
assert not OUT.exists(), 'refusing to overwrite compiler output folder'
sys.path.insert(0, str(OWNER / 'validation/src'))
from numilab_human import model
reg = json.loads(REG.read_text())
src = reg['source']['bodyparts']
anatomy = dict(source_id=src['id'], version=src['version'], archives=src['archives'])
OUT.mkdir(parents=True)
t0 = time.monotonic()
try:
    manifest = model.bodyparts_myosim_fullbody_soft_tissue_visual_payload(
        OWNER / 'Sources', anatomy, REG, MYOSIM, OUT, stable_id_subset=TARGET)
except Exception as exc:
    (OUT / 'failure.json').write_text(json.dumps({'exception':repr(exc)},indent=2)+'\n')
    raise
result = {
  'schema':'numi.human.physical-O-source-route-12-bank-emission.001.v1',
  'scope':'Source-only 385 compiler emission of the 12 banks implicated by the 26 physical-O inside witnesses; no physics or native promotion.',
  'stable_ids':sorted(TARGET),
  'model_path':str(model_path),'model_sha256':sha(model_path),
  'source_registration_path':str(REG),'source_registration_sha256':sha(REG),
  'raw_nhtiss_path':str(BASE),'raw_nhtiss_sha256':sha(BASE),
  'myosim_artifact_path':str(MYOSIM),
  'payload_path':str(OUT/'bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'),
  'payload_sha256':manifest['payload']['sha256'],
  'manifest_sha256':sha(OUT/'bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json'),
  'manifest':manifest,
  'elapsed_s':time.monotonic()-t0,
  'physical_admission':False,
}
(OUT/'compile-report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:result[k] for k in ['payload_sha256','manifest_sha256','elapsed_s','stable_ids']},indent=2))
