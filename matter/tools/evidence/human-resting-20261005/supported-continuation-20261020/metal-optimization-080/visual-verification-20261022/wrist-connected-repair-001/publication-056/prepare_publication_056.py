from __future__ import annotations
import hashlib, json, os, pathlib, shutil, subprocess, sys

TEST_PY = "/Users/n/numi-human-prep-venv-20261005/bin/python"
REPO = pathlib.Path('/Users/n/numi-human-aortic-contact-opt-20261019')
D = REPO / 'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
OUT_REL = pathlib.Path('matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/publication-056')
OUT = REPO / OUT_REL
PARENT = '805765221ea8b01d8c0c115381f226db670bb29b'
EXPECTED_HEAD = 'a1a4b5340181fa48caf3006b5a1b8c565d2cc471'

def sha(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def git(*args, env=None, check=True):
    return subprocess.run(['git', *args], cwd=REPO, env=env, text=True, capture_output=True, check=check)

def live_origin():
    r = subprocess.run(['git','ls-remote','origin','refs/heads/main'], cwd=REPO, text=True, capture_output=True, check=True)
    return r.stdout.split()[0]

if OUT.exists():
    raise SystemExit(f'refusing to overwrite existing {OUT}')
if live_origin() != PARENT:
    raise SystemExit(f'live origin/main changed: {live_origin()} != {PARENT}')
head_before = git('rev-parse','HEAD').stdout.strip()
index_tree_before = git('write-tree').stdout.strip()
if head_before != EXPECTED_HEAD:
    raise SystemExit(f'checkout HEAD changed: {head_before}')
normal_index = os.environ.get('GIT_INDEX_FILE')
if normal_index:
    raise SystemExit('unexpected GIT_INDEX_FILE in preparation environment')

# Exact files frozen by the respective owners. None means hash is recorded from the frozen source bytes at preparation time;
# known hashes are asserted before copying.
files = []
def add(src_rel, dst_rel, expected=None):
    src = D / src_rel
    files.append((src, pathlib.Path('payload') / dst_rel, expected))

# Candidate 020 and its exact/reference, retained-pose, and witness receipts.
base020 = pathlib.Path('bilateral-hand-functional-rebind-020')
for name, h in [
 ('recover_hand_joint_influences_020.py','c00b30da4859d17a400efb9ae61a550c1602d6f1f267840a0d2009079531db20'),
 ('probe_palette_response_020.py','3c7f734d725453027832460cc13ed45c64028c39c3e54de1ac5af9e6d3126b56'),
 ('audit_reference_surface_020.py','5430b6f9ba6eedd7655457e89eebb0979935d7f744f1d38fe78388fb5feb12f2'),
 ('audit_retained_wrist_poses_020.py','0775f77df751163d7bbfb859f4e35c0eab7baf89059b6299cfb972df5aad9163'),
 ('audit_remaining_pose_witnesses_020_v2.py','3f50d9a8246c4fc69133106c1c7eaec3f272cc91c2b173d768db7068b0d12e73'),
 ('audit_remaining_pose_witnesses_020-failed-float-classifier.py','0f23be4c660e66512fdedf9e80a532890ffe0b808864e53c23f04c5cb20dbe3e'),
 ('test_recover_hand_joint_influences_020.py','19cf658bbd42c58006653c517c2a6977a4126cae7736bd9e66e7a585397ad570'),
 ('tests.log','5d28f121fae9f49728804dd6419cc9df67a773f2b584e894e7b9a2c8a8349710'),
 ('run-001.log','2e754e448756a74d8841216cdfc6ef9e2609b50a30d3b2bfc48ebdd21d792a12'),
 ('exact-reference.log','2eb42d10aa18d2ffb248bc31e345ec6fa9d5680e95b89a15019f96d13563a9a1'),
 ('palette-response.log','daa5492c9ff3877e9e7ad42bc4d753e8900ca663c42bfdc6847f0fd654938728'),
 ('remaining-pose-witnesses-v2.log','7b6ded1b902244f9c0e034abfa34aa61671688795640567ab03f0d1d6052541d'),
 ('remaining-pose-witnesses.log','613f43a347f5e19ba594477539247d6c48d9bceb6649f0f9e0f?'),
 ('retained-wrist-pose-regression.log',None),
]:
    if name == 'remaining-pose-witnesses.log':
        h = '613f43a347f5e19ba594477539247d6c48d9bceb6649f0f7db10287615e7acdf'
    add(base020/name, pathlib.Path('bilateral-hand-functional-rebind-020')/name, h)
for name,h in [
 ('candidate.nhskin','7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb'),
 ('selected-hand-rows.npz','0e82da742218cbdc10549d924eb2851244848cec22645710a84e5d1a54fe057c'),
 ('report.json','063c304f3ac2ac83846feb6fbe8c89721925abcb7ca568c0c3cc9550563b70a9'),
 ('palette-response.json',None),
]:
    add(base020/'run-001'/name, pathlib.Path('bilateral-hand-functional-rebind-020/run-001')/name,h)
for name,h in [
 ('exact-reference-regression/report.json','cdaf1794d187429a9a0859b1bf61a73104aa549851a010d66a7622e11ebe0fcb'),
 ('retained-wrist-pose-regression/report.json','ef24c53830e917f81cf94e4de98132eac40dc8aa1e1541d07e22e01b9774e97d'),
 ('remaining-pose-witnesses-002/report.json','0828593609c71f15441a5e171d8c451731b1df9c07a0fd61569ac97794a43e3a'),
]:
    add(base020/'run-001'/name,pathlib.Path('bilateral-hand-functional-rebind-020/run-001')/name,h)

# Accepted v3 connected transfer and the explicitly separate candidate020 mapping counterfactual.
base015 = pathlib.Path('E/connected-skin-transfer-owner-015')
for name,h in [
 ('ConnectedSkinTransferOwner015.hpp','5567ede8'),
 ('prepare_actual_013_to_294.py','35ab89b3'),
 ('actual-013-to-294-input.bin','868da296'),
 ('actual-013-to-294-input.json','6b055057'),
 ('actual-013-to-294-two-way-embedding-v2.npz','7caa180a'),
 ('actual-013-to-294-two-way-embedding-v2.json','003d0a85'),
 ('source-connected-representative-diagnostic-v2.py','163a5e6c'),
 ('representative-source-graph-v2.bin','f85d81e3'),
 ('representative-source-graph-diagnostic-v2.json','6feaaaa7'),
 ('run_actual_013_to_294_v3.cpp','ddf08a26'),
 ('run_actual_013_to_294-v3','6dd6f6be'),
 ('finalize_actual_013_to_294_v3.py','7127a1d1'),
 ('actual-013-to-294-two-way-map-v3.bin','76e1e3a2'),
 ('actual-013-to-294-two-way-embedding-v3.npz','63f8c527'),
 ('actual-013-to-294-two-way-embedding-v3.json','643e6fa6'),
 ('test_ConnectedSkinTransferOwner015.cpp','8996ae12'),
 ('test_ConnectedSkinTransferOwner015-v3','a663b5ec'),
]:
    # Accept full known digest from report, but abbreviated ones below are checked as prefixes only.
    dst = pathlib.Path('connected-skin-transfer-owner-015/v3')/name
    src = D/base015/name
    files.append((src,pathlib.Path('payload')/dst,h))

base020map = base015/'candidate020-counterfactual'
for name,h in [
 ('prepare_counterfactual020_input.py','9d4e824b'),
 ('build_source_connected_representatives_020_v2.py','d2b51383'),
 ('finalize_actual_013_to_294_020_v2.py','0ae59f20'),
 ('run_actual_013_to_294_020.cpp','839655ef'),
 ('run_actual_013_to_294-020','6f6c36bf'),
 ('summarize_candidate020_coverage_v2.py','4a62a41f'),
 ('actual-013-to-294-020-input.bin','a5046a59'),
 ('actual-013-to-294-020-input.json','3ad8729c'),
 ('actual-013-to-294-020-map.bin','e8c1fe8f'),
 ('representative-source-graph-020-v2.bin','caf6b5b1'),
 ('representative-source-graph-020-v2.json','240f6701'),
 ('actual-013-to-294-020-two-way-embedding-v2.npz','16cf5507'),
 ('actual-013-to-294-020-two-way-embedding-v2.json','f7043bf9'),
 ('candidate020-owner-coverage-v2.json','f4703320'),
]:
    files.append((D/base020map/name,pathlib.Path('payload/connected-skin-transfer-owner-015/candidate020-counterfactual')/name,h))
# Identity-004 source-emission attempt. Omit the 23 MB CSV, retaining its external hash pin.
base_emission=base020map/'native-source-emission-identity004-001'
for name,h in [
 ('native-source-emission-lineage.json','5412f45ac08224c8963d82c2a742759a04551b0f2132645e5ca4c088436b3710'),
 ('owner194-native-emission-comparison.json','8b5d6bf0aa3ee3cf376f2e71aa7f8222a40a5028407e658ccf3d150fa8a41f49'),
 ('execution-receipt.json','df5b3ea9dc14816689846ebcd92aee7bd73c2d0eba2ec95f7d174a0407ff2cdc'),
 ('launch-manifest.snapshot.json','2f06f2bad80df5c73cbb9159150d6ab138720f19aed6907c7b277a78b57772ce'),
 ('stdout.log','0562992da3fdf430155288bf744c7f09f35f24357d3bf3b2fb33b960751b1f2c'),
 ('stderr.log','febeeba71964f0e56ae5ed4bb31dfc5692430185bbf1fc9f7cba768a8030b678'),
 ('registered-skin-poses.json','0e476dcacbf89bc79b986a312c75b0dca3894972db87ecff0f9e2d95f3ce48cd'),
]:
    add(base_emission/name,pathlib.Path('bilateral-hand-functional-rebind-020/native-source-emission-identity004-001')/name,h)
# Physical O source used by transfer generation and the actual transfer owner dependency.
files.append((D/'E/connected-skin-mechanical-transfer-013/boundary-refinement-002/candidate.npz', pathlib.Path('payload/inputs/physical-O-013-candidate.npz'),'1849d24c14f8c0ddfedf3aa131877bdd38a852f9e48baebcf900a0efadf75db1'))
files.append((D/'wholebody-attachment-source-rebind-033-final091-regenerated-deltas-001/source/apps/NumiHumanAcceptedFEMSkinTransfer.hpp', pathlib.Path('payload/dependencies/NumiHumanAcceptedFEMSkinTransfer.hpp'),'b7f23c1d9f7279e05f995fcc1cb1d417675f3ffac845220e42ffc8204cf2a72d'))

# Exact audit of the actual emitted identity-004 source CSV (no FEM deformation).
base023=pathlib.Path('native-source-exact-surface-023')
for name,h in [
 ('audit_registered_skin_surface.py','708c1145cf2321e6d9326c40dc78afe7a1262d4a58d6997ee1903c5fb3a562e7'),
 ('test_registered_skin_surface.py','b06689760dd4205ad4e455a6b0634fc5008f68db6fbadd53379131f1dec2e556'),
 ('tests.log','5934fed4baa50a092a170aa1c874aee3a2385289dcde69bb02762f34db6be620'),
 ('run-001.log','f5c42cb0c940630a64df6d1e7d3672566612b0259cdb31eed8bc1cd7d1130846'),
 ('run-001/report.json','d13bf1d9b79111da1ad5636ad5f0f3f58ab866a6297df11c5a39665692e05925'),
]:
    add(base023/name,pathlib.Path('native-source-exact-surface-023')/name,h)

# Latest rejected torso-to-distal-arm counterfactual 022; omit its unadmitted candidate payload.
base022=pathlib.Path("torso-distal-arm-decoupling-022")
for name,h in [
 ("recover_local_torso_weights_022.py","8df4f73beeff69175f9158603612f0e81638237fe274129950c06617000049cc"),
 ("body-identity-inputs.json","a2a8415a1e11b96b14a960b7e4a76c66c39b3df64892a88d68eb9616ec22341f"),
 ("run-001/report.json","9b9d6c6d7780adc7b27dae1891d12b35b33f42098ee56313f647d9d8e10f859c"),
 ("run-001/selected-torso-rows.npz","6b6d3a9690e4cd50abeacb86a0f4c8e52105a5e575ec3b07d12dd6a97d92c24d"),
 ("run-001/right-flex-probe.json","4232d403a0799512f50f0bc68065a8ea0a1092d48c39099a708e7c0bbb53b914"),
 ("run-001.log","cfd27fa6ed3d6b2b78649f1368ab76405cd3789e1455b3a00983f4ce2a62ad41"),
]:
    add(base022/name,pathlib.Path("torso-distal-arm-decoupling-022")/name,h)

# Rejected 018 local flip experiment (not an admitted mesh update).
base018 = pathlib.Path('connected-skin-cavity-remesh-018/run-002-flips-002')
add(pathlib.Path("connected-skin-cavity-remesh-018/flip_local_cavity_018_checkpoint.py"), pathlib.Path("payload/connected-skin-cavity-remesh-018/flip_local_cavity_018_checkpoint.py"),"6f18e6fcec7fb58dfd6975dcc758846c932fb67bf667df7589c7df6f9b860909")
for name,h in [
 ('freeze-manifest.json','f19e3d368a52dfb473d37f13b7fd53e950cb95cffe10fd9993ac09e0ea0e2545'),
 ('audit_flip_validity_018.py','01cb89c52ddc1b0efd40ecc306b7bda7e4e60a15c9bfb1cb19f2980b5b49216c'),
 ('test_flip_validity_018.py','f670e8828b421b2dde4976d53b633b8df38c56f3ec988e6784519912d0da7fb1'),
 ('flip-report-quality-only-as-run.json','f51fa6c5c0819cf93cb1c8ffa098072044a0bf16c32368fda89720738ea414b8'),
 ('flip-report.json','9f25546ff69320cf56eaf43b65ab941b6588c148fef6c685bba00dd9c7b46a10'),
 ('flip-validity-audit.json','0c4612d030d606b8ed1a8e8e6a0ef93d87fb7d249f409f91a3faeb9c096bf3dd'),
 ('local-flip-geometry.npz','5b5a33f1294d4a33fc1600e7ad323cc10e40ca630bed0d7da88c3a66f2f5a7db'),
 ('flip-overlap-validity-witness.npz','8787dce76c9aed87389455702a0a1345620ba4037582574de4b5d773ed75a7ff'),
]:
    add(base018/name,pathlib.Path('connected-skin-cavity-remesh-018/run-002-flips-002')/name,h)

# Separate conforming edge-split geometry checkpoint, with its exact audit and F32 bound.
base018split=pathlib.Path('connected-skin-cavity-remesh-018/run-002-edge-splits-001')
for name,h in [
 ('freeze-manifest.json','4a22a3916d1fd2cb50604bc1288e6afe450e65c86bd681b8dbb2315fbd407c8a'),
 ('freeze_manifest_001.py','d63d546c2187045ffe715f38b9dc5c740b6eeedc111a7ac89d1ae10793357dad'),
 ('build_edge_splits_001.py','4ca61bfd7aee716ec5b058a73d77d689f36796b13b41703dcc6101ef4c2d70ef'),
 ('edge-split-report.json','e6088df8cf717ae266848e9cbac1e2e2865ebceda58d23db13edd6d037a54d44'),
 ('edge-split-geometry.npz','c3ca2e5383a6424ec3feeb83ccfde2d269950eced21cf4d9e578c4a0385ac417'),
 ('certify_float32_roundoff_001.py','baece68ae956932d34c6ad585bb300a3099e5c328921a4f8c22753783e86f033'),
 ('float32-roundoff-certificate.json','2b04beb66624e88d77549614e12d67b1d0be026235812f1a93619bec945421a0'),
 ('exact-self-audit-001/audit_edge_split_surface_001.py','3bd4b768ddab3515ab84bcde355156effe3eae274d159feb82cc316302246bf5'),
 ('exact-self-audit-001/report.json','2d07d9ef628ac1e036d7d569b0d88b3410c6c2fbe95b527a02a8a671a966d121'),
]:
    add(base018split/name,pathlib.Path('connected-skin-cavity-remesh-018/run-002-edge-splits-001')/name,h)
# Regeneration inputs and splitter used by the as-run builder.
for src_rel,dst_rel,h in [
 ('connected-skin-cavity-remesh-018/run-002-holefix-001/quality-witness.json','connected-skin-cavity-remesh-018/inputs-002/quality-witness.json','c636d0d22ca4bfd29797c96d075f88978c99d52c8bbed04cc3debe03ef9cfe06'),
 ('connected-skin-cavity-remesh-018/run-002-holefix-001/cavity-selection.npz','connected-skin-cavity-remesh-018/inputs-002/cavity-selection.npz','646ab7648cf2f8d20c19904190fab8f40328b58f7da77f855b5e102185e19727'),
 ('connected-skin-safe-flow-017/run-002/connected-rest-proposal.npz','connected-skin-cavity-remesh-018/inputs-002/connected-rest-proposal-017.npz','64f40e915703e35950ba2d3ab95b01a5e75681616c316cace25518c3a0ccb054'),
 ('global-inner-p1-edge-support-403/refine_internal_o_edges_403.py','dependencies/refine_internal_o_edges_403.py','231219282f03a830e9857c327e44d2835ce5c831d9eab09ebc598e37a6775826'),
]:
    add(pathlib.Path(src_rel),pathlib.Path(dst_rel),h)
files.append((pathlib.Path('/tmp/numi-human-geometry-slot.py'),pathlib.Path('payload/dependencies/numi-human-geometry-slot.py'),'716fbdeea6fac8f3426e2c083ed4573e8554017ea75f4d4d20318400fa74c8e1'))

# Failed torso-weight counterfactual: keep sources and result evidence, omit 26 MB unadmitted payload.
base021=pathlib.Path('torso-distal-hand-decoupling-021')
for name,h in [
 ('recover_local_torso_weights_021.py','9a9a536388da48db262c9ca146e718acc54bfaa5407ba6d6ae7cf70b0cdece8b'),
 ('run-001/report.json','5fcdfc5fa58b0419954cff428de7b3b286cc5b0e07c172ff36dd353564b634a6'),
 ('run-001/right-flex-probe.json','1c11cdd9a99fa609755d00f58df584ebe013c1cf6ffa78239593342f9a4c1e65'),
 ('run-001/selected-torso-rows.npz','6b6d3a9690e4cd50abeacb86a0f4c8e52105a5e575ec3b07d12dd6a97d92c24d'),
 ('run-001.log','98966a4454e5cff985484eb4fbab8838f23b8b86905a53786b5078bc832da77f'),
]:
    add(base021/name,pathlib.Path('torso-distal-hand-decoupling-021')/name,h)

# Copy only after all source pins pass.
for src,dst,expect in files:
    if not src.is_file():
        raise SystemExit(f'missing frozen source: {src}')
    actual=sha(src)
    if expect:
        if len(expect)<64:
            if not actual.startswith(expect):
                raise SystemExit(f'hash prefix mismatch {src}: {actual} not {expect}')
        elif actual != expect:
            raise SystemExit(f'hash mismatch {src}: {actual} != {expect}')

OUT.mkdir(parents=True)
for src,rel,expect in files:
    dst=OUT/rel
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)

# Evidence interpretation and rights/provenance limits.
rights = '''# Publication 056: evidence scope, provenance, and rights\n\nThis bundle records a bounded hand-influence rebind counterfactual, its source-transfer maps, and rejected local reconstruction/weight trials. It is evidence only: **no candidate is activated or admitted as native, anatomical, physical, or production output.**\n\nThe bilateral 020 candidate preserves the source 294 surface reference and bindings while changing selected hand influence rows using an inferred historical 128 weighting. It passes the exact reference replay and has no new exact self-pairs in the eight retained wrist probes. Existing pose pairs remain (7 in bilateral flex, 3 in bilateral extension, 7 in right flex, 3 in right extension, and 3 in asymmetric flex/extension); the separate witness report classifies these as unresolved transverse interior pairs. This is not a zero-crossing or motion-qualification result.\n\nThe 015 v3 transfer map is an inferred construction embedding, not measured ancestry: it maps 54,551 active rows, using 54,525 direct rows and 26 source-connected fallbacks (12 donor 10 and 14 donor 102), with maximum fallback projection 48.275 mm under a 50 mm construction cap. The separate candidate020 map is also an unadmitted counterfactual. It maps all 54,551 active rows using 54,539 direct rows and 12 donor-10 fallbacks; 8,963 hand rows change dominant owner. Its maximum projection remains 48.275 mm.\n\nThe 018 local-flip experiment is rejected because the two quality-passing 2-to-3 trials fail same-domain overlap; no eligible 3-to-2 star was found. The original TetGen mesh retains one normalized-quality-floor failure. A separate 018 conforming edge-split candidate has positive Float32 tetrahedra, minimum normalized quality 1.03055e-5, exact oriented boundary parity, zero degenerate outer faces, and a complete exact self-surface audit with zero pairs. Its measured parent-area error (maximum relative 9.36e-6) is within the explicit per-parent Float32 roundoff bound; the raw trial's stricter 1e-6 construction threshold is therefore superseded by that independent certificate. The candidate remains unadmitted: full original-parent ancestry, conservative material overlap/mass, regional material closure, and native integration are pending. The 021 torso-weight counterfactual is rejected: it removes four old pairs but adds seven, leaving ten versus seven baseline pairs in right flex. The 26 MB candidate payload is intentionally omitted; its digest is retained in the report. The additional 022 arm-weight counterfactual is also rejected: it removes seven old pairs but introduces seven new pairs. Its unadmitted payload is omitted and its digest is retained in the report.

The identity-004 source-emission replay emitted a 61,950-row registered-skin CSV and byte-identical pose snapshot, then failed the declared mechanical source-hash guard before Matter runtime initialization. Owner-194 comparison measured a maximum position difference of 0.597465 µm (9,716 bitwise-identical rows). No tangent source, state/pipeline, physics step, or native qualification resulted. The 23 MB CSV is retained externally and pinned by SHA-256; this bundle is not self-contained for that emitted table. A separate exact source-surface audit of that emitted CSV found 0 self-intersecting pairs and 0 degenerate faces across 108,987 triangles; it is limited to source emission before FEM deformation and does not qualify native simulation or physical admission. The exact 018 TetGen node and element inputs are retained externally and hash-pinned rather than normalized; `git diff --check` reports trailing whitespace in those generated inputs, and preserving their bytes takes precedence.\n\nProvenance is pinned by paths and SHA-256 values in the reports and manifests. The checked local source records did not provide explicit license or redistribution terms. This publication makes no ownership claim and grants no license; the original asset terms continue to govern.\n'''
(OUT/'provenance-and-rights-056.md').write_text(rights,encoding='utf-8')

# Run included test executables against the copied bytes, without modifying source trees.
logs=[]
def run(cmd,cwd,label):
    env=os.environ.copy(); env['PYTHONDONTWRITEBYTECODE']='1'
    r=subprocess.run(cmd,cwd=cwd,env=env,text=True,capture_output=True)
    body=f'$ {" ".join(cmd)}\nexit_code={r.returncode}\n'+r.stdout+r.stderr
    p=OUT/'test-results'/f'{label}.log'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(body)
    logs.append({'path':str(p.relative_to(OUT)),'exit_code':r.returncode})
    if r.returncode: raise SystemExit(f'test failed {label}; see {p}:\n{body}')

run([TEST_PY,'-m','unittest','-v','test_recover_hand_joint_influences_020'],OUT/'payload/bilateral-hand-functional-rebind-020','020-unit')
run(['./test_ConnectedSkinTransferOwner015-v3'],OUT/'payload/connected-skin-transfer-owner-015/v3','015-v3-owner')
run([TEST_PY,'-m','unittest','-v','test_flip_validity_018'],OUT/'payload/connected-skin-cavity-remesh-018/run-002-flips-002','018-rejected-method-regression')
run([TEST_PY,'-m','unittest','-v','test_registered_skin_surface'],OUT/'payload/native-source-exact-surface-023','023-source-surface-identity')

# Hash inventory includes all evidence, scripts, and test logs except the inventory and final manifest themselves.
records=[]
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.name not in {'payload-files-056.json','publication-manifest-056.json'}:
        records.append({'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
filelist={'schema':'numi.human.publication.payload-files.v1','count':len(records),'total_bytes':sum(x['bytes'] for x in records),'files':records}
(OUT/'payload-files-056.json').write_text(json.dumps(filelist,indent=2,sort_keys=True)+'\n',encoding='utf-8')

# Pin reproducibility inputs that remain external or are inherited from publication 055.
external=[
 {'role':'source 294 NHSKIN reference asset','path':'/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin','sha256':'d4353d5379f552909f8035da2603cfe9d2dc15cd02f4da78a367caab985f1a04','location':'external source; also pinned by publication-055'},
 {'role':'source 128 NHSKIN influence reference','path':'/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/left-ankle-rest-preserving-weight-repair-128/candidate.nhskin','sha256':'4fd80f449b2c4e6a94b0808c2e54a5735684711de1f7fc1c34e893c2494250a5','location':'external evidence input'},
 {'role':'physical O 013 candidate','path':str(OUT/'payload/inputs/physical-O-013-candidate.npz'),'sha256':'1849d24c14f8c0ddfedf3aa131877bdd38a852f9e48baebcf900a0efadf75db1','location':'copied'},
 {'role':'identity 004 registered pose snapshot','path':'/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-004/registered-skin-poses.json','sha256':'0e476dcacbf89bc79b986a312c75b0dca3894972db87ecff0f9e2d95f3ce48cd','location':'external input'},
 {'role':'identity 007 registered pose snapshot','path':'/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/native-app-main-header-integration-151/source-preflight-007/registered-skin-poses.json','sha256':'e26eddfa625d29d6136e164726b83b70394d2d936466d5d558ef046ddb7447c6','location':'external input'},
 {'role':'exact seam-record intersection owner','path':'/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py','sha256':'f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d','location':'external frozen evidence source; hash-pinned'},
 {'role':'registered pose-frame helper 194','path':'/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/registered-pose-frame-helper-194/tools/rebind_human_skin_reference.py','sha256':'be36923a62ebae0ecd475b57d5a413f424712c4fa129c23e6fe4620479869693','location':'external frozen helper source'},
 {'role':'023 exact-surface audit imported current detailed-skin helper','path':'/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/current-detailed-skin-crossing-audit-018/audit_current_detailed_skin_018.py','sha256':'87b756150f663aa12786a6f45ebf7acdb37d77d4925f3328e028e1d2a14988aa','location':'external frozen source dependency'},
 {'role':'023 exact-pair source imported by current detailed-skin helper','path':'/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/global-inner-p1-full-domain-clipping-009/exact-O-pair-011/audit_exact_pairs_011.py','sha256':'baa5280affb42574cc7e209e963ca0843be92c2daab986c19bf2435547967d71','location':'external frozen source dependency'},
 {'role':'identity-004 source-emission registered-skin CSV (retained external output; intentionally not copied)','path':'/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/E/connected-skin-transfer-owner-015/candidate020-counterfactual/native-source-emission-identity004-001/registered-skin.csv','sha256':'eccbc683f8450f27c8c2559471aff8fa8c62644871011d46fe3c3dbfdf3f2524','location':'retained external output; intentionally omitted from bundle'},
 {'role':'018 TetGen element input (retained external input; exact source whitespace intentionally preserved)','path':'/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/connected-skin-cavity-remesh-018/run-002-holefix-001/cavity.1.ele','sha256':'488b29ddee6ababfb089849a847b2519ee2bdd2a45823c2c54af7b8d623f8588','location':'retained external input; omitted byte-for-byte because git diff --check reports existing trailing whitespace'},
 {'role':'018 TetGen node input (retained external input; exact source whitespace intentionally preserved)','path':'/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/connected-skin-cavity-remesh-018/run-002-holefix-001/cavity.1.node','sha256':'00777aca8500ddac0e74fd74fe94db0d6ff45170546370a62166b5a04b200f7b','location':'retained external input; omitted byte-for-byte because git diff --check reports existing trailing whitespace'},
 {'role':'native binary used by rejected source-emission preflight','path':'/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/build-014/numi-human-native','sha256':'27ebf5eace0f4d7872b218c1b210871ad32c9b19dad520c31fae10188aba8aa0','location':'retained external runtime input'},
 {'role':'runtime source revision used by rejected source-emission preflight','path':'/Users/n/numi-human-aortic-contact-opt-20261019/matter/src/runtime.mm','sha256':'ae77770eac80699d1987f912b61b927fd4b55ebcac026d1ae308caf32c0fecc3','location':'repository source at pinned revision a1a4b5340181fa48caf3006b5a1b8c565d2cc471'},
 {'role':'runtime dylib used by rejected source-emission preflight','path':'/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/native-app-main-header-integration-151/build-runtime-004/libmetalrobo.dylib','sha256':'b7cba8777b071eb1df0dbdadc24f9a4963eb92b2e85bf1121a97d29328d5c26b','location':'retained external runtime input'},
 {'role':'metallib used by rejected source-emission preflight','path':'/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/native-app-main-header-integration-151/build-runtime-004/candidate.metallib','sha256':'544c8e8a8a5019ca728d1026b4a8a52ed88ec54b782610e5fae5dd81c5bf3845','location':'retained external runtime input'},
 {'role':'accepted transfer owner','path':str(OUT/'payload/dependencies/NumiHumanAcceptedFEMSkinTransfer.hpp'),'sha256':'b7f23c1d9f7279e05f995fcc1cb1d417675f3ffac845220e42ffc8204cf2a72d','location':'copied'}
]
(OUT/'source-input-pins-056.json').write_text(json.dumps({'schema':'numi.human.publication.external-input-pins.v1','inputs':external},indent=2,sort_keys=True)+'\n',encoding='utf-8')

# Recompute payload list so it includes newly written rights, tests and input pins.
records=[]
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.name not in {'payload-files-056.json','publication-manifest-056.json'}:
        records.append({'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
filelist={'schema':'numi.human.publication.payload-files.v1','count':len(records),'total_bytes':sum(x['bytes'] for x in records),'files':records}
(OUT/'payload-files-056.json').write_text(json.dumps(filelist,indent=2,sort_keys=True)+'\n',encoding='utf-8')

# Publication preparation script is copied in package for reproducibility; make a self-copy.
script_src=pathlib.Path('/tmp/prepare_publication_056.py')
shutil.copy2(script_src,OUT/'prepare_publication_056.py')
# Rebuild final inventory after adding builder.
records=[]
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.name not in {'payload-files-056.json','publication-manifest-056.json'}:
        records.append({'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
filelist={'schema':'numi.human.publication.payload-files.v1','count':len(records),'total_bytes':sum(x['bytes'] for x in records),'files':records}
(OUT/'payload-files-056.json').write_text(json.dumps(filelist,indent=2,sort_keys=True)+'\n',encoding='utf-8')

manifest={
 'schema':'wrist-connected-repair-publication-056-v1',
 'status':'PREPARED_FOR_REVIEW_UNPUBLISHED',
 'created_files_are_evidence_only':True,
 'repository':'numi-human-aortic-contact-opt-20261019',
 'repository_root':str(REPO),
 'publication_path_from_repository_root':OUT_REL.as_posix(),
 'target_parent_commit':PARENT,
 'origin_main_at_prepare':live_origin(),
 'current_checkout_head_at_prepare':head_before,
 'normal_index_tree_before_and_after':index_tree_before,
 'payload_file_count':len(records),
 'payload_total_bytes':sum(x['bytes'] for x in records),
 'payload_filelist_sha256':sha(OUT/'payload-files-056.json'),
 'source_input_pins_sha256':sha(OUT/'source-input-pins-056.json'),
 'prepare_script_sha256':sha(OUT/'prepare_publication_056.py'),
 'scope':['bilateral hand influence recovery 020 and retained-pose exact checks','identity-004 native source-emission preflight and owner-194 parity comparison (rejected before Matter initialization)','023 exact self-surface and degenerate-face audit of the actual emitted source CSV before FEM deformation','connected skin transfer 015 v3 inferred maps and candidate020 mapping counterfactual','rejected local cavity bistellar-flip experiment 018','unadmitted conforming edge-split geometry candidate 018 with roundoff certificate and exact surface audit','failed torso-distal-hand weight counterfactual 021','failed torso-distal-arm weight counterfactual 022'],
 'excluded':['candidate activation/native scene changes','claim of anatomical/physical/native qualification','021 and 022 unadmitted candidate payloads (hashes retained in reports)','020/015 external raw source assets except the explicitly copied 013 physical-O candidate','23 MB identity-004 emitted CSV (retained externally and hash-pinned)','018 TetGen node and element inputs (retained externally and hash-pinned byte-for-byte)','all unlisted mutable experiments'],
 'interpretation':['020 has no new exact self-pairs across eight retained wrist probes but retains pre-existing exact pairs; unresolved pairs are transverse interior witnesses.','Identity-004 source emission stopped at the declared source-payload hash guard before Matter initialization; 023 separately found no exact self-pairs or degenerate faces in the emitted source CSV, before FEM deformation.','015 v3 and candidate020 transfer maps are inferred construction embeddings, not measured ancestry; 48.275 mm is a construction projection, not anatomy.','018 bistellar flip is rejected by same-domain overlap; the separate edge-split geometry candidate passes the included geometry/topology checks but has unresolved material ancestry/mass and native admission.','021 weight counterfactual is rejected because it creates more pose pairs than it removes.','022 arm-weight counterfactual is rejected because it removes seven old pairs and introduces seven new pairs.'],
 'omitted_candidate_payload_hashes':{'021':'f3b200c292238e317109bf82cfebd978a9b82abf923f8b22d5339acf7805bacc','022':'582dd74a879da31e2af553b91c8211ca45b8eeff0c329f23cca6270c27726c73'},
 'included_unadmitted_candidate_hashes':{'020-hand-influence-rebind':'7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb','018-edge-split-geometry':'c3ca2e5383a6424ec3feeb83ccfde2d269950eced21cf4d9e578c4a0385ac417'},
 'rights_note':'No explicit license or redistribution terms were found in the checked local source records. This publication makes no ownership claim or license grant; original asset terms remain controlling.',
 'tests':logs,
 'whitespace_check':{
   'status':'PASS_WITH_PINNED_RETAINED_BLANK_AT_EOF_IN_SOURCE_EVIDENCE',
   'allowed_findings':[
     'payload/bilateral-hand-functional-rebind-020/audit_remaining_pose_witnesses_020-failed-float-classifier.py:33',
     'payload/bilateral-hand-functional-rebind-020/audit_remaining_pose_witnesses_020_v2.py:34',
     'payload/bilateral-hand-functional-rebind-020/audit_retained_wrist_poses_020.py:52',
     'payload/dependencies/NumiHumanAcceptedFEMSkinTransfer.hpp:609',
     'payload/dependencies/numi-human-geometry-slot.py:14',
     'payload/native-source-exact-surface-023/audit_registered_skin_surface.py:39',
     'payload/native-source-exact-surface-023/test_registered_skin_surface.py:17',
     'payload/torso-distal-hand-decoupling-021/recover_local_torso_weights_021.py:54',
     'payload/torso-distal-arm-decoupling-022/recover_local_torso_weights_022.py:54'
   ]
 }
}
(OUT/'publication-manifest-056.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')

# Validate every payload digest, all copied-source digests, input pins where local files exist, and documentation hashes.
for rec in records:
    p=OUT/rec['path']
    if sha(p)!=rec['sha256'] or p.stat().st_size!=rec['bytes']:
        raise SystemExit(f'payload verification failed: {rec["path"]}')
for pin in external:
    p=pathlib.Path(pin['path'])
    if p.is_absolute() and p.exists() and p.is_file() and sha(p)!=pin['sha256']:
        raise SystemExit(f'external input hash mismatch: {p}')

# Create a commit object through an alternate index only. No ref update and no push.
alt='/tmp/publication-056.index'
try: os.unlink(alt)
except FileNotFoundError: pass
env=os.environ.copy(); env['GIT_INDEX_FILE']=alt
git('read-tree',PARENT,env=env)
git('add','-f','--',OUT_REL.as_posix(),env=env)
tree=git('write-tree',env=env).stdout.strip()
diffcheck=subprocess.run(['git','diff','--check',PARENT,tree],cwd=REPO,env=env,text=True,capture_output=True)
allowed=set(manifest['whitespace_check']['allowed_findings'])
actual=set()
for line in (diffcheck.stdout+diffcheck.stderr).splitlines():
    if not line.strip(): continue
    if 'new blank line at EOF' not in line: raise SystemExit(f'unapproved whitespace finding: {line}')
    prefix=line.rsplit(':',2)
    if len(prefix)<3: raise SystemExit(f'unparseable whitespace finding: {line}')
    item=prefix[0].replace(OUT_REL.as_posix()+"/","")+":"+prefix[1]
    actual.add(item)
if actual != allowed: raise SystemExit(f'whitespace findings differ: actual={sorted(actual)} allowed={sorted(allowed)}')
commit=git('commit-tree',tree,'-p',PARENT,'-m','Add scoped hand rebind evidence publication 056',env=env).stdout.strip()
# Check normal checkout/index and remote tracking state remained unchanged.
head_after=git('rev-parse','HEAD').stdout.strip()
index_tree_after=git('write-tree').stdout.strip()
if head_after!=head_before or index_tree_after!=index_tree_before or live_origin()!=PARENT:
    raise SystemExit('normal checkout/index or remote main changed during alternate-index preparation')
summary={'commit_object':commit,'tree':tree,'parent':PARENT,'publication_path':OUT_REL.as_posix(),'manifest_sha256':sha(OUT/'publication-manifest-056.json'),'payload_filelist_sha256':sha(OUT/'payload-files-056.json'),'files':len(records),'bytes':sum(x['bytes'] for x in records),'head_unchanged':head_after,'normal_index_tree_unchanged':index_tree_after,'origin_main_unchanged':live_origin(),'pushed':False,'ref_updated':False}
pathlib.Path('/tmp/publication-056-summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps(summary,indent=2,sort_keys=True))
