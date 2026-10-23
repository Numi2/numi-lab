#!/usr/bin/env python3
"""Rebuild D17's projected local frames from matched source C and reference residual."""
from __future__ import annotations
import contextlib, csv, hashlib, io, json, pathlib, re, runpy, shutil
import numpy as np

R=pathlib.Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
D16=R/'elbow-refinement-current-main-016-owner-orientation-005-001'
D17=R/'elbow-refinement-current-main-017-single-residual-prolongation-001'
BASE=R/'d16-source-transfer-diagnostic-002/d16_transfer_diagnostic_base_preserved.py'

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
def f32(x):
    return float(np.float32(x))
def cppf(x):
    return format(f32(x),'.9g')+'f'
def cppd(x):
    return format(float(x),'.17g')
def parse3(text):
    return np.array([float(x.strip().removesuffix('f')) for x in text.split(',')],dtype=np.float64)
def row3(v,fmt):
    return ','.join(fmt(x) for x in v)

parent_manifest_sha=sha(D16/'candidate-manifest-016.json')
assert parent_manifest_sha=='444fff29852e9a41b9dd84e0a8a5fe737a998337fd97ebdbbfce301ab9edce0a', parent_manifest_sha
# Validate all matched source references before computing any generated table.
registered_csv=D16/'native-source-only-016/registered-skin.csv'
pose_json=D16/'native-source-only-016/registered-skin-poses.json'
assert sha(registered_csv)=='76f5b1bdda4b64d6c3074ec66544ca1bd4221fc61934899e8aa38836c8ccc36e'
assert sha(pose_json)=='e26eddfa625d29d6136e164726b83b70394d2d936466d5d558ef046ddb7447c6'
with contextlib.redirect_stdout(io.StringIO()):
    d=runpy.run_path(str(BASE))
emb,df,src,fr,faces,weights,binding_body,bodies,refpts,residual=[d[k] for k in ('emb','df','src','fr','faces','weights','binding_body','bodies','refpts','residual')]
nodes=list(csv.DictReader((D16/'generated/nodes.csv').open()))
assert len(emb)==len(df)==len(nodes)==len(fr)==2411 and len(src)==61950
assert refpts.shape==(2411,3) and residual.shape==(2411,3)
assert all(int(emb[i]['source_vertex'])==0xffffffff for i in range(2332,2411))

# Recompute projected outer donor-local frames from the exact registered source C,
# not from the erroneous compiled local value. Compare to the independent captured CSV.
new_o_local={}; new_i_local={}; new_d_local={}; rows=[]
for i in range(2332,2411):
    e=emb[i]
    beta=np.array([float(e[k]) for k in ('b0','b1','b2')],dtype=np.float64)
    tri=faces[int(e['face'])]
    blend=beta @ weights[tri].astype(np.float64)
    owner_binding=int(np.argmax(blend)); owner=int(binding_body[owner_binding])
    corners=[int(src[int(t)]['donor']) for t in tri]
    if corners != [owner,owner,owner]:
        raise RuntimeError(f'projected row {i}: source corners {corners} do not share owner {owner}')
    Mcur,bodypos=d['bodyM'](owner)
    q=np.array([float(x) for x in bodies[owner]['orientation_world_xyzw']],dtype=np.float64)
    qi=np.array([-q[0],-q[1],-q[2],q[3]],dtype=np.float64)
    Ccur=beta @ np.array([[float(src[int(t)][k]) for k in ('wx','wy','wz')] for t in tri],dtype=np.float64)
    Clocal=d['quat_rotate'](qi,Ccur-bodypos)
    Mref=np.asarray(fr[i][2],dtype=np.float64).reshape(3,3)
    # This is the required single registration-residual application: O_local=C_local+Mref*(O-C).
    Olocal=Clocal + Mref @ residual[i]
    csvlocal=np.array([float(df[i][k]) for k in ('frame_lx','frame_ly','frame_lz')],dtype=np.float64)
    expected_world=np.asarray(refpts[i],dtype=np.float64)
    c_ref=np.array([float(df[i][k]) for k in ('source_cx','source_cy','source_cz')],dtype=np.float64)
    residual_check=expected_world-c_ref
    if np.max(np.abs(residual_check-residual[i]))>3e-7:
        raise RuntimeError(f'row {i}: registered residual is not O-C (max {np.max(np.abs(residual_check-residual[i]))})')
    csv_err=float(np.max(np.abs(Olocal-csvlocal)))
    if csv_err>2e-6:
        raise RuntimeError(f'row {i}: independently rebuilt O-local differs from native CSV by {csv_err}m')
    node=nodes[i]
    Oworld=np.array([float(node[k]) for k in ('ox','oy','oz')],dtype=np.float64)
    Iworld=np.array([float(node[k]) for k in ('ix','iy','iz')],dtype=np.float64)
    Dworld=np.array([float(node[k]) for k in ('dx','dy','dz')],dtype=np.float64)
    if np.max(np.abs(Oworld-expected_world))>2e-12:
        raise RuntimeError(f'row {i}: nodes.csv O differs from exact candidate reference O')
    Ilocal=Olocal + Mref @ (Iworld-Oworld)
    Dlocal=Olocal + Mref @ (Dworld-Oworld)
    new_o_local[i]=Olocal; new_i_local[i]=Ilocal; new_d_local[i]=Dlocal
    rows.append({'row':i,'owner':owner,'source_face':int(e['face']),'source_c_local_m':Clocal.tolist(),
                 'registered_residual_m':residual[i].tolist(),'O_local_m':Olocal.tolist(),
                 'native_csv_O_local_m':csvlocal.tolist(),'csv_O_local_error_m':csv_err,
                 'I_local_m':Ilocal.tolist(),'D_local_m':Dlocal.tolist(),
                 'world_O_I_D_unchanged':True})

# Update only the 79 projected rows in the D16 donor-frame / inner-local / deep-local arrays.
header=D17/'apps/NumiHumanWholeBodySkinReduction.hpp'
original=header.read_text()
pattern=re.compile(r'WholeBodySkinDonorFrameV1\{(\d+)u,\{([^{}]*)\},\{([^{}]*)\},\{([^{}]*)\},\{([^{}]*)\}\}')
seen=[]
def update_frame(m):
    i=len(seen); seen.append(i)
    if i not in new_o_local: return m.group(0)
    return ('WholeBodySkinDonorFrameV1{'+m.group(1)+'u,{'+row3(new_o_local[i],cppf)+'},{'+m.group(3)+'},{'+m.group(4)+'},{'+m.group(5)+'}}')
updated=pattern.sub(update_frame,original)
if len(seen)!=2411:
    raise RuntimeError(f'donor frame parse mismatch: {len(seen)} rows')

def update_local_array(text,name,new_values):
    start=text.index(f'{name}={{{{')
    end=text.index('}};',start)+3
    block=text[start:end]
    num=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    rowpat=re.compile(r'(?:std::array<double,3>)?\{\s*('+num+r')\s*,\s*('+num+r')\s*,\s*('+num+r')\s*\}')
    ids=[]
    def replace(m):
        i=len(ids); ids.append(i)
        if i not in new_values:return m.group(0)
        return 'std::array<double,3>{'+row3(new_values[i],cppd)+'}'
    b2=rowpat.sub(replace,block)
    if len(ids)!=2411: raise RuntimeError(f'{name}: parsed {len(ids)} rows')
    return text[:start]+b2+text[end:]
updated=update_local_array(updated,'kWholeBodySkinInnerLocal',new_i_local)
updated=update_local_array(updated,'kWholeBodyFatDeepLocal',new_d_local)
header.write_text(updated)

# Copy hash-pinned 065 baseline prolongation implementation.
for rel in ['NumiHumanAcceptedFEMSkinTransfer.hpp','NumiHumanElbowBaselineProlongation065.hpp']:
    source=R/'skin-transfer-subdivision-consistency-065/apps'/rel
    target=D17/'apps'/rel
    if sha(source)!=sha(target): raise RuntimeError(f'065 copy hash mismatch: {rel}')
render_source=R/'ankle-main-app-integration-001/source/apps/NumiHumanAcceptedFEMSkinRenderPass.hpp'
render_target=D17/'apps/NumiHumanAcceptedFEMSkinRenderPass.hpp'
shutil.copy2(render_source,render_target)
if sha(render_source)!='9c1bb707b5cb8a756bab3c531172d812fcb6c936491a17545a2693c21b8b0718':
    raise RuntimeError('accepted render-pass dependency hash changed')
boundary_source=R/'ankle-main-app-integration-001/source/apps/NumiHumanAcceptedFEMBoundaryRenderPass.hpp'
boundary_target=D17/'apps/NumiHumanAcceptedFEMBoundaryRenderPass.hpp'
visual_source=R/'ankle-main-app-integration-001/source/apps/NumiHumanRestingVisual.hpp'
visual_target=D17/'apps/NumiHumanRestingVisual.hpp'
shutil.copy2(boundary_source,boundary_target)
shutil.copy2(visual_source,visual_target)
if sha(boundary_source)!='391483a87abeb5859a768d756b58a9373b73806ee10ae402dd113f55de6c9927':
    raise RuntimeError('accepted boundary render-pass dependency hash changed')
if sha(visual_source)!='92c3ba5b1cab09aee31871d46ba8eee74277392953872f3cbf037476ca958a1e':
    raise RuntimeError('resting visual dependency hash changed')

# Add prolongation to the existing transfer builder and a candidate-locked diagnostic gate.
app=D17/'apps/numilab_human_myosim_visual_probe.mm'
s=app.read_text()
if '#include "NumiHumanElbowBaselineProlongation065.hpp"' not in s:
    needle='#include "NumiHumanAcceptedFEMSkinTransfer.hpp"\n'
    if needle not in s: raise RuntimeError('accepted FEM transfer include not found')
    s=s.replace(needle,needle+'#include "NumiHumanElbowBaselineProlongation065.hpp"\n',1)
needle='''            wholeBodySkinTransfer=std::make_unique<AcceptedSkinTransfer>();
            std::string transferError;'''
repl='''            wholeBodySkinTransfer=std::make_unique<AcceptedSkinTransfer>();
            const auto wholeBodySkinBaselineProlongation=
                numi::human::makeElbowPhysicalBaselineProlongation065(region.worldPoints.size());
            std::string transferError;'''
if needle not in s: raise RuntimeError('transfer construction insertion point not found')
s=s.replace(needle,repl,1)
needle='''                        &wholeBodySkinSourceRows.sourceFaces,
                        &wholeBodySkinSourceRows.sourceBarycentric),'''
repl='''                        &wholeBodySkinSourceRows.sourceFaces,
                        &wholeBodySkinSourceRows.sourceBarycentric,
                        &wholeBodySkinBaselineProlongation),'''
if needle not in s: raise RuntimeError('baseline prolongation build argument insertion point not found')
s=s.replace(needle,repl,1)
# Candidate fingerprint is computed below and inserted into this unique placeholder.
needle='''            require(kWholeBodySkinLayersAdmitted,
                    "whole-body skin/fat is disabled until its exact Float32 geometry and layer receipt is admitted");'''
repl='''            constexpr std::string_view kUnadmittedDiagnosticCandidateID=
                "elbow-refinement-current-main-017-single-residual-prolongation-001";
            constexpr std::string_view kUnadmittedDiagnosticCandidateSHA256="__CANDIDATE_FINGERPRINT__";
            const char* unadmittedDiagnostic=std::getenv("NUMI_HUMAN_EXPERIMENTAL_UNADMITTED_LAYER_DIAGNOSTIC");
            const char* unadmittedCandidateID=std::getenv("NUMI_HUMAN_EXPERIMENTAL_UNADMITTED_LAYER_CANDIDATE_ID");
            const char* unadmittedCandidateSHA=std::getenv("NUMI_HUMAN_EXPERIMENTAL_UNADMITTED_LAYER_CANDIDATE_SHA256");
            const bool exactUnadmittedDiagnostic=unadmittedDiagnostic&&std::string_view(unadmittedDiagnostic)=="1"&&
                unadmittedCandidateID&&std::string_view(unadmittedCandidateID)==kUnadmittedDiagnosticCandidateID&&
                unadmittedCandidateSHA&&std::string_view(unadmittedCandidateSHA)==kUnadmittedDiagnosticCandidateSHA256;
            require(kWholeBodySkinLayersAdmitted||exactUnadmittedDiagnostic,
                    "whole-body skin/fat is unadmitted; diagnostic requires the exact private candidate ID and SHA-256");
            if(!kWholeBodySkinLayersAdmitted&&exactUnadmittedDiagnostic)
                std::cout<<"whole_body_skin_layer_status=UNADMITTED_DIAGNOSTIC candidate_id="
                         <<kUnadmittedDiagnosticCandidateID<<" candidate_sha256="
                         <<kUnadmittedDiagnosticCandidateSHA256<<" physical_qualification=not_claimed\\n";'''
if needle not in s: raise RuntimeError('admission check not found')
s=s.replace(needle,repl,1)
app.write_text(s)

# Candidate identity binds immutable geometry and all source/local-frame tables, excluding app source to avoid a hash cycle.
identity_paths=[
 ('parent_manifest',D16/'candidate-manifest-016.json'),
 ('generated_embedding',D17/'generated/embedding.csv'),
 ('generated_nodes',D17/'generated/nodes.csv'),
 ('generated_faces',D17/'generated/faces.csv'),
 ('generated_tetrahedra',D17/'generated/tetrahedra.csv'),
 ('generated_parent_tetrahedra',D17/'generated/parent-tetrahedra.csv'),
 ('generated_midpoint_ancestry',D17/'generated/midpoint-ancestry.csv'),
 ('generated_donor_frames',D17/'generated/donor-frames-fullweights.csv'),
 ('reduction_header',header),
 ('topology_header',D17/'apps/NumiHumanElbowExplicitPairedTopology008.hpp'),
 ('reference_world_header',D17/'apps/NumiHumanElbowReferenceWorldPoints014.hpp'),
 ('embedding_header',D17/'apps/NumiHumanWholeBodySkinTransferAncestry.hpp'),
 ('accepted_transfer_065',D17/'apps/NumiHumanAcceptedFEMSkinTransfer.hpp'),
 ('baseline_prolongation_065',D17/'apps/NumiHumanElbowBaselineProlongation065.hpp'),
 ('accepted_render_pass',D17/'apps/NumiHumanAcceptedFEMSkinRenderPass.hpp'),
 ('accepted_boundary_render_pass',D17/'apps/NumiHumanAcceptedFEMBoundaryRenderPass.hpp'),
 ('resting_visual',D17/'apps/NumiHumanRestingVisual.hpp'),
]
identity_files={name:sha(path) for name,path in identity_paths}
identity_blob=json.dumps(identity_files,sort_keys=True,separators=(',',':')).encode()
identity_sha=hashlib.sha256(identity_blob).hexdigest()
if '__CANDIDATE_FINGERPRINT__' not in app.read_text():
    raise RuntimeError('candidate fingerprint placeholder missing')
app.write_text(app.read_text().replace('__CANDIDATE_FINGERPRINT__',identity_sha))

# Check all world O/I/D and old prefix row data are byte-identical to D16 inputs.
for rel in ['generated/nodes.csv','generated/faces.csv','generated/tetrahedra.csv','generated/parent-tetrahedra.csv','generated/midpoint-ancestry.csv','generated/embedding.csv','generated/donor-frames-fullweights.csv','apps/NumiHumanElbowReferenceWorldPoints014.hpp','apps/NumiHumanElbowExplicitPairedTopology008.hpp','apps/NumiHumanWholeBodySkinTransferAncestry.hpp']:
    if sha(D16/rel)!=sha(D17/rel): raise RuntimeError(f'immutable input changed: {rel}')
# Parse final donor O local rows and verify expected CSV agreement after float32 emission.
final_text=header.read_text(); donor_match=list(pattern.finditer(final_text))
if len(donor_match)!=2411: raise RuntimeError('final donor frame count mismatch')
for i in range(2332):
    a=pattern.search(original)
    # Exact prefix check is performed by comparing each parsed initializer text.
    if pattern.findall(final_text)[i]!=pattern.findall(original)[i]: raise RuntimeError('legacy donor-frame prefix changed')

# Emit one report/manifest with hashes for frozen inputs. Build receipts are kept separate.
report={
 'schema':'numi.elbow_refinement_d17_local_frame_prolongation.v1',
 'candidate_id':'elbow-refinement-current-main-017-single-residual-prolongation-001',
 'candidate_identity_sha256':identity_sha,
 'status':'private candidate; unadmitted diagnostic only; source-only native init pending',
 'physical_qualification':False,'FEM_horizon_run':False,
 'parent_candidate':'D16-owner-orientation-core005','parent_manifest_sha256':parent_manifest_sha,
 'source_csv_sha256':sha(registered_csv),'pose_snapshot_sha256':sha(pose_json),
 'skin_payload_sha256':'d4353d5379f552909f8035da2603cfe9d2dc15cd02f4da78a367caab985f1a04',
 'prolongation_065':{'accepted_transfer_sha256':sha(D17/'apps/NumiHumanAcceptedFEMSkinTransfer.hpp'),
                     'baseline_helper_sha256':sha(D17/'apps/NumiHumanElbowBaselineProlongation065.hpp'),
                     'parent_rows':2332,'projected_rows':79,'identity_rows':2332,'interpolated_rows':79},
 'change_scope':{'changed_arrays':['projected donorLocalPoint rows 2332..2410','projected skin inner local rows 2332..2410','projected fat deep local rows 2332..2410','accepted transfer baseline prolongation input'],
                 'O_I_D_world_geometry_changed':False,'topology_changed':False,'weights_changed':False,'materials_changed':False,
                 'mass_debits_changed':False,'legacy_2332_prefix_changed':False,'admission_flag_changed':False,
                 'all_thresholds_and_physical_gates_changed':False},
 'formula':{'O_local':'fresh source C_local + M_reference * registered residual (O-C), once',
            'I_local':'O_local + M_reference * (I_world-O_reference_world)',
            'D_local':'O_local + M_reference * (D_world-O_reference_world)',
            'baseline_prolongation':'identity on existing rows; 0.5/0.5 mapped parent endpoints for each new midpoint'},
 'metrics':{'projected_rows':len(rows),'all_source_triangle_corners_share_donor':True,
            'max_O_local_vs_native_source_frame_m':max(x['csv_O_local_error_m'] for x in rows),
            'max_residual_O_minus_C_consistency_m':float(max(np.max(np.abs(refpts[i]-np.array([float(df[i][k]) for k in ('source_cx','source_cy','source_cz')])-residual[i])) for i in range(2332,2411))),
            'max_abs_I_D_local_relation_error_m':max(float(np.max(np.abs(new_i_local[i]-new_o_local[i]-np.asarray(fr[i][2]).reshape(3,3)@(np.array([float(nodes[i][k]) for k in ('ix','iy','iz')])-refpts[i])))) for i in range(2332,2411))},
 'environment_gate':{'flag':'NUMI_HUMAN_EXPERIMENTAL_UNADMITTED_LAYER_DIAGNOSTIC','candidate_id':'NUMI_HUMAN_EXPERIMENTAL_UNADMITTED_LAYER_CANDIDATE_ID','candidate_sha256':'NUMI_HUMAN_EXPERIMENTAL_UNADMITTED_LAYER_CANDIDATE_SHA256','requires_all_three_exact_values':True,'kWholeBodySkinLayersAdmitted_remains_false':True},
 'identity_inputs':identity_files,
 'file_hashes':{
   'apps/numilab_human_myosim_visual_probe.mm':sha(app),
   'apps/NumiHumanWholeBodySkinReduction.hpp':sha(header),
   'apps/NumiHumanAcceptedFEMSkinTransfer.hpp':sha(D17/'apps/NumiHumanAcceptedFEMSkinTransfer.hpp'),
   'apps/NumiHumanElbowBaselineProlongation065.hpp':sha(D17/'apps/NumiHumanElbowBaselineProlongation065.hpp'),
   'apps/NumiHumanAcceptedFEMSkinRenderPass.hpp':sha(D17/'apps/NumiHumanAcceptedFEMSkinRenderPass.hpp'),
   'apps/NumiHumanAcceptedFEMBoundaryRenderPass.hpp':sha(D17/'apps/NumiHumanAcceptedFEMBoundaryRenderPass.hpp'),
   'apps/NumiHumanRestingVisual.hpp':sha(D17/'apps/NumiHumanRestingVisual.hpp'),
 }
}
# Reduce huge embedded paths: list only files explicitly needed for candidate and reproduce via parent D16.
(D17/'candidate-manifest-017.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
(D17/'local-frame-generation-report-017.json').write_text(json.dumps({'schema':report['schema'],'candidate_id':report['candidate_id'],'candidate_identity_sha256':identity_sha,'rows':rows,'metrics':report['metrics']},indent=2,sort_keys=True)+'\n')
print(json.dumps({'candidate':report['candidate_id'],'identity_sha256':identity_sha,'manifest':str(D17/'candidate-manifest-017.json'),'manifest_sha256':sha(D17/'candidate-manifest-017.json'),'projected_rows':len(rows),'max_O_local_vs_native_source_frame_m':report['metrics']['max_O_local_vs_native_source_frame_m'],'app_sha256':sha(app),'reduction_header_sha256':sha(header)},indent=2))
