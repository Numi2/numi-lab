#!/usr/bin/env python3
"""CPU-only final scene assembly using viewer 018 and post-run 931 checks."""
import argparse,csv,hashlib,json,math,os,re,shlex,struct,subprocess,sys
from pathlib import Path
E=Path("/Users/n/numi-human-resting-evidence-20261005");H=Path("/Users/n/numi-human-resting-final-integration-001");L=Path("/Users/n/numi-human-performance-source-014");B=Path("/Users/n/numi-human-retired-alias-visibility-build-018-attempt2")
RUNTIME=Path("/Users/n/numi-human-performance-build-014")
V=Path("/Users/n/numi-human-retired-alias-visibility-018")
D=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004")
OLD=E/"common-atlas-skin-composition-907/resting-scene";REF=E/"native-terminal-cycle-931"
BASE003=E/"common-atlas-skin-registration-003/resting-anatomy-composition-001/resting-anatomy-receipt.json"
NHA=E/"lung-choroid-composition-924/choroid-v2/resting-thorax.nhanatomy";AR=E/"lung-choroid-composition-924/final-v2/resting-anatomy-receipt.json"
T=E/"tendon-semantic-foot-migration-834/numi-human-tendon-attachments.nhtendon";C=E/"reference-circulation-001/resting_reference_lv15.native.v3.json";R=E/"integrated-anatomy-engineering-lung-edge-collapse-761/resting-reference-respiration.json"
CAP="0,4991,5375,5759,6111,6495,7743,10000"
SKIN907=E/"common-atlas-skin-composition-907/bodyparts3d-myosim-skinned-shell.nhskin"
MANIFEST907=E/"common-atlas-skin-composition-907/common-atlas-skin-geometry-registration.manifest.json"
REF930=E/"native-terminal-candidate-930"; RUN932=E/"native-terminal-production-audits-932"
AGGREGATED=("min_contact_gap_m","peak_penetration_m","pre_projection_contact_residual_m_s",
 "pre_projection_limit_residual_generalized_s","pre_projection_equality_residual_generalized_s",
 "post_projection_contact_residual_m_s","post_projection_limit_residual_generalized_s",
 "post_projection_equality_residual_generalized_s","equality_position_projection_max_generalized",
 "equality_velocity_projection_max_generalized_s")
P={"human_commit":"935cc7d332b9118f41c719e2d2f9c7c568a3d41f","lab_commit":"d550d8ad88a76fdee5bb6e028286fd24e962571f","resting_anatomy":"350f6c42517408de2f8c73729f35dd26b80bdd2a54d77fbca4ae63ef66182551","resting_scene":"373cbc4777b9c4e9bb0c43254e66effdff69d8bf05686847c793294392363188","resting_run":"f6bc12635fcf41227a79b4d67e058b36f392a4147d083ce24366f8c3ef41c412","numi":"2e971e7c3680ec809c7e9ac71355b89502f710f835bc3c69725a91dbbdb01074","route":"77693c89dfb3bdc368b15d192d7e86f44ede696c516fa5a80e796f8ec8c3ce81","study":"b06bb5a1cd14d86be362a5d0603f68546ba590bf038f9586c4cf89946fc39c1e","build_pins":"cb883675ac1fc6735c80c49769ea6e0065181b427181bb99ea947e35d6952fdd","build_source_pins":"a2066150f3dedb1c40b64a74c8476bfa9b0bd76b5110a35f018a71859b008c2f","source_delta_patch":"e7acace57f332c0ae202f47cd8f8afe59242f7ebbab97499f5cc5cfbc58f52fa","viewer_commit":"41b254605e66035a4e3966e5a1ae790197431605","build_script":"f82d8646914ea94189cb767b87bbea30bdb3c0fbb9c921066de502a1bc253cee","binary":"486442abe2070168a902fb06669469e8b6cc612a9a6ef65ddb3d54957308d906","resp_metallib":"4b61361f513bf0996d687398498b85ba4e379edca91c36f134e1b0398f31c426","lib":"6bccfc4044d825423e66bc2f60ba3cf59eaa9a4936773ab08182f58a09927092","rigid":"2c78cb4150b97cea6e8169dad9e8f5dd59af667b247e07b56e48e857435560e4","body_order":"844d05330104a43f6c45867020f2abd493adec35d6e2f8f90fc636ee9bec04e7","nha":"3c444be7736c066a992988cc32b687917e1c4c5c3968a16b4d5f0106d5b5024e","receipt":"40652ad554565f27713d6023af3327994f9c226e5471eddbd3b77d1baaaace3e","tendon":"49daaf61421254cb18f3aa32f1d332b1810537afd4c4ef813abd2f0d118dcecc","circ":"43ff6b49daf1d42cf9e88d85d8577f70e112dfcc3756ec43954e544a3e4bc0dc","resp":"c518926bf47fba945cef52bb952ed6c559d604508988082c5b641b720bda503d","skin907":"898990d49a3f1dfa0cbe85b10765bf62fa0cc3c834a3371f5503c83facfacba2","manifest907":"f3aab90ecbd0751e984143b9e7a270312d880e09bde803f12e2a0d4bd9073d9c","skin_base_receipt":"a52e3184e77bca568ce603831cfd5d3b2bafdf46453634637a9c85fb6ffbed71","ref930csv":"dd2f3f975e9f01165ef9a23e6e399bbf0368abb8a95b745cc3ab72aae6c363a2","run932csv":"96cdfa7d392b316ce3a579721dcdce0993b1f00cdc6472b2624f542a05099485","run932meta":"6e2eabdd6f60ba3789d910b08e539ad922c8fab0d6c746016ee291f604d5f80c","old_scene":"8802260236813440c200c7723bd78a36112f7722b007f4160d0ac9c5aaf16f36","old_contact":"bcfece8e5da553b98694b724644234407fa4c38383b1e18d3c24d7caefa28927","refmeta":"d09f717e98e86208773afc63412db67eb59752766ab600f4d140dbc58d6489f0","refcsv":"aa3b6f12f482f09becbdf20185240d9bb27a945df21dd0771096aac0243faffd"}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4194304),b""):h.update(b)
 return h.hexdigest()
def hash_path(p):
 p=Path(p)
 if p.is_file():return sha(p)
 if p.is_dir():
  files=sorted(q for q in p.rglob("*") if q.is_file() and not q.is_symlink())
  if not files:raise RuntimeError("empty source directory "+str(p))
  manifest="".join(sha(q)+"  "+str(q)+"\n" for q in files).encode()
  return hashlib.sha256(manifest).hexdigest()
 raise RuntimeError("missing source input "+str(p))
def ck(p,d=None):
 p=Path(p)
 h=hash_path(p)
 if d and h!=d:raise RuntimeError("hash mismatch "+str(p))
 return h
def wj(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+chr(10))
def parse_contact(path):
 from numilab_human.resting_scene import NHCNT_HEADER,NHCNT_RECORD,NHCNT_MAGIC
 b=Path(path).read_bytes();m,abi,bc,n,res,src,*plane=NHCNT_HEADER.unpack_from(b)
 if m!=NHCNT_MAGIC or abi!=1 or res or not 1<=n<=32 or len(b)!=NHCNT_HEADER.size+n*NHCNT_RECORD.size:raise RuntimeError("invalid NHCNT")
 rows=[]
 for i in range(n):
  r=NHCNT_RECORD.unpack_from(b,NHCNT_HEADER.size+i*NHCNT_RECORD.size);v=r[2:]
  rows.append({"body_index":r[0],"geometry_id":r[1],"vertex_index":r[1]-1,"local_point_m":list(v[:3]),"bed_witness_m":list(v[3:6]),"friction":v[6]})
 return {"sha256":hashlib.sha256(b).hexdigest(),"body_count":bc,"source_archive_sha256":src.hex(),"plane_point":plane[:3],"plane_normal":plane[3:6],"friction":plane[6],"rows":rows}
def repo(p,h):
 head=subprocess.check_output(["git","-C",str(p),"rev-parse","HEAD"],text=True).strip();dirty=subprocess.check_output(["git","-C",str(p),"status","--porcelain","--untracked-files=all"],text=True).strip()
 if head!=h or dirty:raise RuntimeError("source checkout changed "+str(p))
 return {"path":str(p),"head":head,"clean":True}
def verify_viewer_build_lineage(build_pins,source_pins):
 base="b091d7dcead509a325194563ed38261319118a88"
 if build_pins.get("source_revision")!=base:
  raise RuntimeError("viewer018 build does not retain the pinned b091d7d build base")
 if source_pins.get("source_revision")!=base or source_pins.get("source_tree_clean") is not False:
  raise RuntimeError("viewer018 original build pins must disclose dirty-at-build source state")
 if source_pins.get("source_delta_patch_sha256")!=P["source_delta_patch"] or ck(B/"evidence/source-delta.patch",P["source_delta_patch"])!=P["source_delta_patch"]:
  raise RuntimeError("viewer018 original dirty source patch is not exact")
 expected={"build_script_sha256":P["build_script"],"compiled_binary_sha256":P["binary"],"human_respiration_metallib_sha256":P["resp_metallib"],"runtime_library_sha256":P["lib"]}
 if any(source_pins.get(k)!=v for k,v in expected.items()):raise RuntimeError("viewer018 build/source pin metadata mismatch")
 checkout=repo(V,P["viewer_commit"])
 pinned_files={}
 for path,digest in build_pins.get("source_files",{}).items():
  pinned_files[str(Path(path))]=ck(path,digest)
 relative_source_files={};source_pin_path_checks={}
 for pin,digest in source_pins.items():
  if Path(pin).is_absolute():
   source_pin_path_checks[pin]=ck(pin,digest)
   continue
  if "/" not in pin and pin!="CMakeLists.txt":continue
  path=V/pin
  got=ck(path,digest)
  absolute=str(path.resolve())
  if build_pins.get("source_files",{}).get(absolute)!=digest:
   raise RuntimeError("committed viewer source is not bound by the original build pin: "+pin)
  relative_source_files[pin]=got
  source_pin_path_checks[pin]=got
 if not relative_source_files:raise RuntimeError("viewer source pin manifest contains no tracked source paths")
 return {"current_source_checkout":checkout,"compiled_source_byte_equivalence":relative_source_files,
  "source_pin_path_checks":source_pin_path_checks,
  "build_source_file_and_directory_pins_verified":len(pinned_files),
  "directory_hash_algorithm":"SHA-256 over sorted regular-file rows '<sha256>  <absolute path>\n'; symlinks excluded",
  "build_source_provenance":{"base_revision":base,"source_tree_clean_at_build":False,
   "dirty_source_delta_sha256":P["source_delta_patch"],"binary_sha256":P["binary"]}}
def compose_for_scene(composer,fixture,skin,manifest,output,anatomy_receipt=AR):
 base_receipt=BASE003 if fixture else anatomy_receipt
 composition_dir=Path(output)/"anatomy"
 result=composer(base_receipt,skin,manifest,composition_dir)
 composed_receipt=composition_dir/"resting-anatomy-receipt.json"
 native_receipt=anatomy_receipt if fixture else composed_receipt
 return base_receipt,composition_dir,composed_receipt,native_receipt,result

def validate_compare_contract(invocation,metadata,assembly,expected_assets,expected_dyld,expected_runtime_path,expected_runtime_sha):
 for key in ("argv","asset_sha256","environment"):
  if metadata.get(key)!=invocation.get(key):raise RuntimeError("invocation/metadata differs: "+key)
 owner=assembly["owner_cli"]
 if invocation.get("argv")!=owner.get("resolved_native_argv") or invocation.get("asset_sha256")!=owner.get("asset_sha256"):
  raise RuntimeError("native invocation differs from exact assembled owner command/assets")
 request=owner.get("requested",{})
 if request.get("inspection_tour") is not True or request.get("inspection_period_s")!=2.5:
  raise RuntimeError("assembly did not request the admitted inspection tour")
 for path,digest in expected_assets.items():
  if invocation["asset_sha256"].get(path)!=digest:raise RuntimeError("critical runtime/anatomy dependency differs: "+path)
 env=invocation["environment"]
 if env.get("NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT")!="0" or env.get("NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS")!="8":
  raise RuntimeError("wrong q/COM diagnostic settings")
 if env.get("NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS")!="0,4991,5375,5759,6111,6495,7743,10000":
  raise RuntimeError("wrong accepted capture cadence")
 if env.get("NUMI_HUMAN_RESTING_INSPECTION_TOUR")!="1" or env.get("NUMI_HUMAN_RESTING_INSPECTION_PERIOD_SECONDS")!="2.5":
  raise RuntimeError("wrong inspection-tour environment")
 if env.get("DYLD_LIBRARY_PATH")!=expected_dyld or not expected_dyld.startswith("/Users/n/numi-human-performance-build-014/lib:"):
  raise RuntimeError("DYLD runtime path is not the pinned Lab014 library path")
 argv=invocation["argv"]
 try:
  steps=int(argv[argv.index("--muscle-step-count")+1]);dt=float(argv[argv.index("--muscle-step-seconds")+1])
 except (ValueError,IndexError) as exc:raise RuntimeError("native argv lacks physical step settings") from exc
 if steps!=10000 or dt!=.002:raise RuntimeError("not 20 s at 2 ms")
 runtime=metadata.get("loaded_metal_runtime",{})
 if (runtime.get("verified") is not True or runtime.get("expected_path")!=expected_runtime_path
     or runtime.get("expected_sha256")!=expected_runtime_sha or len(runtime.get("observed_images",[]))!=1
     or runtime["observed_images"][0].get("path")!=expected_runtime_path):
  raise RuntimeError("loaded native Metal runtime does not match pinned Lab014 library")
 return {"assembly_invocation_equal":True,"critical_assets_verified":len(expected_assets),"inspection_tour":True,
         "inspection_period_seconds":2.5,"dyld_library_path":expected_dyld,"loaded_runtime_verified":True}

def select_anatomy(a):
  values=[getattr(a,k,None) for k in ("anatomy","anatomy_sha","anatomy_receipt","anatomy_receipt_sha")]
  if any(v is not None for v in values) and not all(v is not None for v in values):
    raise RuntimeError("custom anatomy requires both paths and both exact hashes")
  if all(v is not None for v in values):
    if a.fixture_legacy_907:raise RuntimeError("legacy fixture cannot override pinned anatomy")
    nha,nha_sha,receipt,receipt_sha=values
    nha=Path(nha).resolve();receipt=Path(receipt).resolve()
  else:
    nha,nha_sha,receipt,receipt_sha=NHA.resolve(),P["nha"],AR.resolve(),P["receipt"]
  ck(nha,nha_sha);ck(receipt,receipt_sha)
  payload=json.loads(receipt.read_text()).get("payload",{})
  if payload.get("sha256")!=nha_sha or Path(payload.get("path","")).resolve()!=nha:
    raise RuntimeError("anatomy receipt does not bind exact selected payload path/hash")
  return nha,nha_sha,receipt,receipt_sha

def anatomy_assets_from_assembly(assembly):
  native=assembly["native_receipt"];receipt=Path(native["path"]).resolve()
  ck(receipt,native["sha256"])
  payload=json.loads(receipt.read_text()).get("payload",{})
  path=Path(payload.get("path","")).resolve();digest=payload.get("sha256")
  if digest!=native["NHA_sha256"] or digest!=assembly["base_nha_sha256"]:
    raise RuntimeError("assembly and composed receipt anatomy identities differ")
  if assembly["source_sha256"].get(str(path))!=digest:
    raise RuntimeError("assembled anatomy missing from immutable source identities")
  ck(path,digest)
  return {str(path):digest,str(receipt):native["sha256"]}

def assemble(a):
 sys.path[:0]=[str(H/"src"),str(L/"matter/tools")]
 from numilab_human import resting_scene,resting_anatomy,resting_run
 from resting_intervention_study import NATIVE_310S_REQUIRED_ENVIRONMENT,NATIVE_310S_DISABLED_EXPERIMENTS
 nha,nha_sha,ar,ar_sha=select_anatomy(a)
 files=[(D/"myosim-fullbody-core-reference.nhrigid","rigid"),(D/"myosim-fullbody-reference.manifest.json","body_order"),(BASE003,"skin_base_receipt"),(T,"tendon"),(C,"circ"),(R,"resp"),(B/"evidence/build-pins.json","build_pins"),(B/"evidence/source-pins.json","build_source_pins"),(B/"evidence/source-delta.patch","source_delta_patch"),(E/"native-retired-alias-visibility-018-attempt2/build-native-viewer-018-attempt2.sh","build_script"),(B/"bin/numi-human-native","binary"),(B/"matter/shaders/HumanRespiration.metallib","resp_metallib"),(Path("/Users/n/numi-human-performance-build-014/lib/libmetalrobo.dylib"),"lib"),(OLD/"resting-supine-scene.manifest.json","old_scene"),(OLD/"myosim-fullbody-resting-bed-support.nhcnt","old_contact"),(REF/"run-metadata.json","refmeta"),(REF/"resting-coupled.csv","refcsv")]
 for p,k in files:ck(p,P[k])
 checkouts=[repo(H,P["human_commit"]),repo(L,P["lab_commit"])]
 mods=[(H/"src/numilab_human/resting_anatomy.py","resting_anatomy"),(H/"src/numilab_human/resting_scene.py","resting_scene"),(H/"src/numilab_human/resting_run.py","resting_run"),(L/"tools/numi","numi"),(L/"numi/commands/human-resting","route"),(L/"matter/tools/resting_intervention_study.py","study")]
 for p,k in mods:ck(p,P[k])
 viewer_build_pins_path=B/"evidence/build-pins.json"
 viewer_source_pins_path=B/"evidence/source-pins.json"
 viewer_patch_path=B/"evidence/source-delta.patch"
 viewer_build_pins=json.loads(viewer_build_pins_path.read_text())
 viewer_source_pins=json.loads(viewer_source_pins_path.read_text())
 viewer_lineage=verify_viewer_build_lineage(viewer_build_pins,viewer_source_pins)
 checkouts.append(viewer_lineage["current_source_checkout"])
 expected_artifacts={str(B/"bin/numi-human-native"):P["binary"],str(B/"matter/shaders/HumanRespiration.metallib"):P["resp_metallib"],str(RUNTIME/"lib/libmetalrobo.dylib"):P["lib"]}
 if any(viewer_build_pins.get("artifacts",{}).get(path)!=digest for path,digest in expected_artifacts.items()):
  raise RuntimeError("viewer018 build artifact pins differ from launch paths")
 skin=a.skin.resolve();mf=a.manifest.resolve()
 if ck(skin)!=a.skin_sha or ck(mf)!=a.manifest_sha:raise RuntimeError("caller candidate hashes differ")
 fixture=bool(a.fixture_legacy_907)
 if fixture and (skin!=SKIN907.resolve() or mf!=MANIFEST907.resolve() or a.skin_sha!=P["skin907"] or a.manifest_sha!=P["manifest907"]):
  raise RuntimeError("legacy-fixture mode accepts only exact pinned 907 skin and manifest")
 if not fixture and (skin==SKIN907.resolve() or mf==MANIFEST907.resolve()):
  raise RuntimeError("907 inputs require explicit --fixture-legacy-907; they are not final corrected assets")
 md=json.loads(mf.read_text());op=md.get("output_payload",{})
 if md.get("schema")!="numi.human.common-atlas-skin-geometry-registration.v1" or Path(op.get("path","")).resolve()!=skin or op.get("sha256")!=a.skin_sha:raise RuntimeError("registration manifest does not bind exact NHSKIN")
 if json.loads(ar.read_text()).get("payload",{}).get("sha256")!=nha_sha:raise RuntimeError("selected receipt/NHA mismatch")
 out=a.out.resolve()
 if E not in out.parents or out.exists():raise RuntimeError("output must be fresh child of evidence")
 out.mkdir(parents=True)
 composition_base,composition_dir,composition_receipt_path,native_receipt,comp=compose_for_scene(
  resting_anatomy.compose_skin_binding_candidate,fixture,skin,mf,out,ar)
 composed_receipt=json.loads(composition_receipt_path.read_text())
 if comp.get("anatomy_payload_unchanged") is not True or comp.get("functional_bindings_unchanged") is not True:
  raise RuntimeError("skin composition changed the base anatomy payload or functional bindings")
 if composed_receipt["mass_geometry_accounting"]["skin_payload_sha256"]!=a.skin_sha:
  raise RuntimeError("skin composition did not bind the exact candidate skin")
 if fixture:
  current_receipt=json.loads(ar.read_text())
  if current_receipt["mass_geometry_accounting"]["skin_payload_sha256"]!=a.skin_sha:
   raise RuntimeError("current 924 receipt does not already bind exact fixture skin")
 else:
  if composed_receipt["payload"]["sha256"]!=nha_sha:
   raise RuntimeError("final skin composition changed the selected NHA")
 current_native_receipt=json.loads(native_receipt.read_text())
 if current_native_receipt["payload"]["sha256"]!=nha_sha:
  raise RuntimeError("selected native receipt does not bind the selected NHA")
 scene=resting_scene.prepare(D/"myosim-fullbody-core-reference.nhrigid",D/"myosim-fullbody-muscle-reference.nhmyo",skin,D/"myosim-fullbody-reference.manifest.json",out/"resting-scene",pitch_degrees=-89.,friction=1.)
 old=json.loads((OLD/"resting-supine-scene.manifest.json").read_text());oc=parse_contact(OLD/"myosim-fullbody-resting-bed-support.nhcnt");nc=parse_contact(scene["outputs"]["support_contact"]["path"])
 opose=old["pose"];npose=scene["pose"];ob={x["geometry_id"]:x for x in oc["rows"]};nb={x["geometry_id"]:x for x in nc["rows"]};common=set(ob)&set(nb)
 cmp={"old_root_xyz_m":opose["root_translation_xyz_m"],"new_root_xyz_m":npose["root_translation_xyz_m"],"root_translation_delta_m":[npose["root_translation_xyz_m"][i]-opose["root_translation_xyz_m"][i] for i in range(3)],"old_quaternion_xyzw":opose["root_delta_quaternion_xyzw"],"new_quaternion_xyzw":npose["root_delta_quaternion_xyzw"],"old_support_vertex_ids":sorted(x["vertex_index"] for x in oc["rows"]),"new_support_vertex_ids":sorted(x["vertex_index"] for x in nc["rows"]),"removed_support_vertex_ids":sorted(x["vertex_index"] for x in oc["rows"] if x["geometry_id"] not in nb),"added_support_vertex_ids":sorted(x["vertex_index"] for x in nc["rows"] if x["geometry_id"] not in ob),"same_vertex_body_changes":[i for i in sorted(common) if ob[i]["body_index"]!=nb[i]["body_index"]],"same_vertex_local_point_deltas_m":{str(i):[nb[i]["local_point_m"][j]-ob[i]["local_point_m"][j] for j in range(3)] for i in common if nb[i]["local_point_m"]!=ob[i]["local_point_m"]},"old_nhcnt_sha256":oc["sha256"],"new_nhcnt_sha256":nc["sha256"],"old_min_full_skin_gap_m":old["bed"]["minimum_source_skin_gap_m"],"new_min_full_skin_gap_m":scene["bed"]["minimum_source_skin_gap_m"],"new_nhcnt_support_rows":nc["rows"],"scope":"reduced 32-region point contact derived from the complete registered skin each step; NHCNT records initial region seeds, not triangle-mesh collision"}
 if fixture:
  if nc["sha256"]!=P["old_contact"] or scene["pose"]!=old["pose"] or scene["bed"]["support_witnesses"]!=old["bed"]["support_witnesses"]:
   raise RuntimeError("907 CPU fixture did not exactly reproduce pinned 907 root/NHCNT/support rows")
 receipt=native_receipt
 if json.loads(receipt.read_text())["payload"]["sha256"]!=nha_sha:raise RuntimeError("selected native receipt/NHA pin differs")
 nativeout=out/"native-run";ns=argparse.Namespace(body_scene=out/"resting-scene/resting-supine-scene.manifest.json",anatomy_receipt=receipt,tendon=T,lab=L,build=B,output=nativeout,circulation=C,respiration=R,seconds=20.,dt=.002,dimension=512,mechanics_only=False,postural_activation_cap=.01,release_initialization=True,upper_passive_joints=False,rigid_hands=True,contact_iterations=64,drive_intervention=None,inspection_tour=True,inspection_period_seconds=2.5)
 argv,assets=resting_run.command(ns)
 env={"HOME":"/Users/n","PATH":"/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin","TMPDIR":"/tmp","LANG":"en_US.UTF-8","NUMI_LAB_ROOT":str(L),"NUMI_BUILD_DIR":str(B),"NUMI_HUMAN_ROOT":str(H),"NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS":CAP,"NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT":"0","NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT":"1","NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS":"8","NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT":str(nativeout/"common-field-failure.json")}
 env.update(NATIVE_310S_REQUIRED_ENVIRONMENT);env.update(NATIVE_310S_DISABLED_EXPERIMENTS)
 cli=[str(L/"tools/numi"),"human-resting","--body-scene",str(ns.body_scene),"--anatomy-receipt",str(receipt),"--tendon",str(T),"--output",str(nativeout),"--circulation",str(C),"--respiration",str(R),"--seconds","20","--dt","0.002","--dimension","512","--postural-activation-cap","0.01","--release-initialization","--rigid-hands","--contact-iterations","64","--inspection-tour","--inspection-period-seconds","2.5"]
 dyld_lib=(B/"lib").resolve();dyld_matter=(B/"matter").resolve()
 if dyld_lib!=RUNTIME/"lib":raise RuntimeError("viewer018 DYLD lib resolution is not frozen physical014")
 cp=out/("review-only-owner-command.txt" if fixture else "launch-command.sh")
 cp.write_text("#!/bin/sh\nset -eu\nexec /usr/bin/env -i "+" ".join(k+"="+shlex.quote(v) for k,v in sorted(env.items()))+" "+shlex.join(cli)+"\n")
 if not fixture:cp.chmod(0o755)
 src={str(p):sha(p) for p in [skin,mf,SKIN907,MANIFEST907,nha,ar,BASE003,composition_base,composition_receipt_path,receipt,T,C,R,D/"myosim-fullbody-core-reference.nhrigid",D/"myosim-fullbody-reference.manifest.json",B/"evidence/build-pins.json",B/"evidence/source-pins.json",B/"evidence/source-delta.patch",E/"native-retired-alias-visibility-018-attempt2/build-native-viewer-018-attempt2.sh",B/"bin/numi-human-native",B/"matter/shaders/HumanRespiration.metallib",V/"CMakeLists.txt",V/"apps/NumiHumanRestingAnatomy.hpp",V/"apps/NumiHumanRestingVisual.hpp",V/"apps/NumiHumanRestingInspectionLayers.hpp",V/"apps/numilab_human_myosim_visual_probe.mm",V/"matter/src/human_respiration.metal",V/"tests/numi_human_retired_inspection_layers_test.cpp",Path(__file__).resolve(),Path("/Users/n/numi-human-resting-evidence-20261005/final-native-scene-preflight-936/v014/test_final_scene_936_v014.py"),Path("/Users/n/numi-human-performance-build-014/lib/libmetalrobo.dylib"),OLD/"resting-supine-scene.manifest.json",OLD/"myosim-fullbody-resting-bed-support.nhcnt",REF/"run-metadata.json",REF/"resting-coupled.csv"]+[p for p,k in mods]}
 for path,digest in viewer_build_pins.get("source_files",{}).items():
  src[str(Path(path))]=ck(path,digest)
 report={"schema":"numi.human.final-native-scene-preflight.v1","status":("fixture_907_legacy_unqualified_cpu_only" if fixture else "assembled_owner_cli_validated_native_not_run"),"asset_role":("pinned 907 fixture only; not corrected/final geometry" if fixture else "caller-pinned NHSKIN and exact receipt-bound NHA; assembly is not geometry qualification"),"owner_checkouts":checkouts,"source_sha256":src,"preflight_script_sha256":sha(Path(__file__).resolve()),"base_nha_sha256":nha_sha,"skin_composition":{"base_receipt_path":str(composition_base),"base_receipt_sha256":sha(composition_base),"composed_receipt_path":str(composition_receipt_path),"composed_receipt_sha256":sha(composition_receipt_path),"composed_skin_sha256":composed_receipt["mass_geometry_accounting"]["skin_payload_sha256"],"composition_base_nha_sha256":composed_receipt["payload"]["sha256"],"used_by_native_cli":receipt.resolve()==composition_receipt_path.resolve(),"owner_result":comp,"scope":("legacy fixture composed against its exact 003 source receipt; CLI uses already bound current924 receipt" if fixture else "final candidate composed onto selected anatomy receipt; composed receipt is used by scene CLI")},"native_receipt":{"path":str(receipt),"sha256":sha(receipt),"NHA_sha256":current_native_receipt["payload"]["sha256"],"source":"existing current924 receipt" if fixture else "owner-composed selected anatomy receipt"},"scene":{"manifest":str(out/"resting-scene/resting-supine-scene.manifest.json"),"manifest_sha256":sha(out/"resting-scene/resting-supine-scene.manifest.json"),"support_payload":str(scene["outputs"]["support_contact"]["path"]),"support_sha256":nc["sha256"],"root_pose":scene["pose"],"bed":scene["bed"],"comparison_to_907":cmp},"viewer_build_binding":{"build_base_revision":viewer_lineage["build_source_provenance"]["base_revision"],"source_tree_clean_at_build":False,"dirty_source_delta_sha256":P["source_delta_patch"],"current_source_commit":P["viewer_commit"],"current_source_tree_clean":True,"compiled_source_byte_equivalence":viewer_lineage["compiled_source_byte_equivalence"],"source_pin_path_checks":viewer_lineage["source_pin_path_checks"],"build_source_file_and_directory_pins_verified":viewer_lineage["build_source_file_and_directory_pins_verified"],"directory_hash_algorithm":viewer_lineage["directory_hash_algorithm"],"build_source_provenance":viewer_lineage["build_source_provenance"],"build_pins_path":str(viewer_build_pins_path),"source_pins_path":str(viewer_source_pins_path),"binary_sha256":P["binary"]},"owner_cli":{"resolved_native_argv":argv,"asset_sha256":assets,"asset_count":len(assets),"asset_bytes":sum(Path(p).stat().st_size for p in assets),"requested":{"nominal_seconds":20,"dt":.002,"steps":10000,"dimension":512,"activation_cap":.01,"release_init":True,"rigid_hands":True,"contact_iterations":64,"inspection_tour":True,"inspection_period_s":2.5,"q_audit":0,"COM_segment_steps":8,"captures":[int(x) for x in CAP.split(",")]},"expected_runtime_environment":{"DYLD_LIBRARY_PATH":str(dyld_lib)+":"+str(dyld_matter),"DYLD_PRINT_LIBRARIES":"1","inspection":"owner CLI sets tour=1, period=2.5s"},"validated_by":"resting_run.command"},"launch_command":str(cp),"native_command_executable":not fixture,"post_run_comparison":"Use --compare-run/--compare-out; candidate is bound to its adjacent assembly report and exact owner argv/assets; 931 one-root data is compared to q0/COM8 endpoint and window extrema. q0 omits generalized-q trace.","native_run":False,"limits":["20 s and sparse captures do not qualify 310 s anatomy or physiology","contact is reduced 32-region point contact derived from the complete registered skin each step; NHCNT rows seed those regions, not triangle-mesh collision"]}
 wj(out/"assembly-preflight.json",report)
 print(json.dumps({"status":report["status"],"out":str(out),"root_delta_xyz_m":cmp["root_translation_delta_m"],"support_counts":[len(oc["rows"]),len(nc["rows"])],"asset_count":len(assets),"native_run":False},indent=2))
def read_csv(path):
 with Path(path).open(newline="") as f:return list(csv.DictReader(f))
def compare_cadence(reference_path,candidate_path,roots,segment=8,require_exact=False):
 ref=read_csv(reference_path);cand=read_csv(candidate_path)
 if len(ref)!=roots or roots%segment:raise RuntimeError("reference trace length is not the declared root count")
 if not ref or not cand:raise RuntimeError("trace CSV is empty")
 headers=list(ref[0])
 if set(headers)!=set(cand[0]) or len(headers)!=60:raise RuntimeError("trace CSV missing/extra column or schema is not 60 fields")
 rsteps=[int(row["step"]) for row in ref];csteps=[int(row["step"]) for row in cand]
 if rsteps!=list(range(1,roots+1)):raise RuntimeError("reference is not one row per accepted root")
 if csteps!=list(range(segment,roots+1,segment)):raise RuntimeError("candidate is not one endpoint per exact segment")
 if not set(AGGREGATED)<=set(headers):raise RuntimeError("missing a declared window-extrema diagnostic")
 columns={};matched=0
 for row in cand:
  step=int(row["step"]);window=ref[step-segment:step];endpoint=ref[step-1]
  for key in headers:
   try:
    got=float(row[key])
    vals=[float(q[key]) for q in (window if key in AGGREGATED else [endpoint])]
   except (KeyError,ValueError,TypeError) as exc:raise RuntimeError(f"missing or nonnumeric trace value at step {step}, column {key}") from exc
   if not math.isfinite(got) or any(not math.isfinite(v) for v in vals):raise RuntimeError(f"nonfinite trace value at step {step}, column {key}")
   expected=(min(vals) if key=="min_contact_gap_m" else max(vals)) if key in AGGREGATED else vals[0]
   q=columns.setdefault(key,{"exact_equal":0,"max_abs_delta":0.0,"sum_signed_delta":0.0,"samples":0})
   delta=got-expected;q["exact_equal"]+=int(got==expected);q["max_abs_delta"]=max(q["max_abs_delta"],abs(delta));q["sum_signed_delta"]+=delta;q["samples"]+=1;matched+=1
 for q in columns.values():q["mean_signed_delta"]=q.pop("sum_signed_delta")/q["samples"]
 exact=all(q["exact_equal"]==q["samples"] for q in columns.values())
 if require_exact and not exact:raise RuntimeError("cadence fixture did not match endpoint/window-extrema semantics")
 return {"reference_rows":len(ref),"candidate_rows":len(cand),"reference_root_count":roots,"segment_steps":segment,"candidate_accepted_steps":csteps,"endpoint_field_count":len(headers)-len(AGGREGATED),"window_aggregate_fields":list(AGGREGATED),"aggregation":"min for min_contact_gap_m; max for the other nine; direct accepted endpoint for all remaining fields","compared_values":matched,"all_values_exact":exact,"columns":columns}
def cadence_self_test():
 ck(REF930/"resting-coupled.csv",P["ref930csv"]);ck(RUN932/"resting-coupled.csv",P["run932csv"]);ck(RUN932/"run-metadata.json",P["run932meta"])
 m=json.loads((RUN932/"run-metadata.json").read_text())
 if m.get("exit_code")!=0 or m.get("environment",{}).get("NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT")!="0" or m.get("environment",{}).get("NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS")!="8":raise RuntimeError("932 retained cadence fixture metadata mismatch")
 result=compare_cadence(REF930/"resting-coupled.csv",RUN932/"resting-coupled.csv",64,8,True)
 import tempfile
 with tempfile.TemporaryDirectory(prefix="numi-936-cadence-") as tmp:
  rows=read_csv(RUN932/"resting-coupled.csv");bad=Path(tmp)/"missing.csv"
  with bad.open("w",newline="") as f:
   w=csv.DictWriter(f,fieldnames=[x for x in rows[0] if x!="PaO2_mmhg"]);w.writeheader();w.writerows([{k:v for k,v in q.items() if k!="PaO2_mmhg"} for q in rows])
  try:compare_cadence(REF930/"resting-coupled.csv",bad,64,8)
  except RuntimeError:pass
  else:raise RuntimeError("cadence reader accepted a missing column")
  rows[0]["PaO2_mmhg"]="nan";bad2=Path(tmp)/"nonfinite.csv"
  with bad2.open("w",newline="") as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  try:compare_cadence(REF930/"resting-coupled.csv",bad2,64,8)
  except RuntimeError:pass
  else:raise RuntimeError("cadence reader accepted a nonfinite value")
 return result
def verify_terminal_log(run,n):
 text=(Path(run)/"native.log").read_text(errors="replace")
 done=re.findall(r"^resting_integrated_body=completed simulated_s=(\S+).*? physiology_body_clock=matched root_assistance=false(?: |$)",text,re.M)
 if len(done)!=1:raise RuntimeError("native log lacks unique successful matched-clock/unassisted completion")
 dt32=struct.unpack("<f",struct.pack("<f",.002))[0]
 if abs(float(done[0])-n*dt32)>1e-9:raise RuntimeError("native terminal simulated time does not match accepted count and Float32 dt")
 terminal=[q for q in text.splitlines() if "resting_terminal_presentation=accepted step="+str(n)+" " in q]
 identity=[q for q in text.splitlines() if "resting_terminal_capture_identity=accepted_step_"+str(n)+" " in q]
 if len(terminal)!=1 or "physical_steps_advanced=0" not in terminal[0] or "controller_steps_advanced=0" not in terminal[0]:
  raise RuntimeError("native log lacks no-advance accepted terminal presentation")
 if len(identity)!=1 or "terminal_physical_steps_advanced=0" not in identity[0]:
  raise RuntimeError("native log lacks exact accepted terminal identity")
 return {"accepted_steps":n,"actual_simulated_s":float(done[0]),"float32_dt_s":dt32,"terminal_presentations":len(terminal),"terminal_state_advanced_physics":False}
def verify_retired_alias_visibility_log(run):
 text=(Path(run)/"native.log").read_text(errors="replace")
 rows=[line for line in text.splitlines() if "resting_retired_inspection_source" in line]
 if len(rows)!=1:raise RuntimeError("native log lacks one retired-alias inspection-mask record")
 if re.match(r"^resting_retired_inspection_source semantic=51010(?:\s|$)",rows[0]) is None:
  raise RuntimeError("native log retired-alias record is not anchored to organ semantic 51010")
 required=(r"(?<!\S)stable_id=22(?!\S)",r"(?<!\S)layer_mask=0(?!\S)",
           r"(?<!\S)payload_retained=true(?!\S)",r"(?<!\S)geometry_retained=true(?!\S)")
 if any(re.search(token,rows[0]) is None for token in required):raise RuntimeError("native log does not prove retired organ semantic stable ID 22 is retained with a zero inspection mask")
 return {"retired_semantic":51010,"retired_organ_stable_ids":[22],"zero_inspection_mask":True,"payload_retained":True,"geometry_retained":True,"record":rows[0]}
def compare(run,out):
 run=run.resolve();out=out.resolve()
 if out.exists() or E not in out.parents:raise RuntimeError("comparison output must be fresh under evidence")
 for p,k in [(REF/"run-metadata.json","refmeta"),(REF/"resting-coupled.csv","refcsv")]:ck(p,P[k])
 inv=json.loads((run/"invocation.json").read_text());m=json.loads((run/"run-metadata.json").read_text())
 if m.get("exit_code")!=0 or m.get("source_files_changed_during_run")!=[] or m.get("loaded_metal_runtime",{}).get("verified") is not True:raise RuntimeError("owner run failed, changed, or runtime unverified")
 assembly_path=run.parent/"assembly-preflight.json"
 if not assembly_path.is_file():raise RuntimeError("candidate run lacks its exact CPU assembly preflight")
 assembly=json.loads(assembly_path.read_text())
 if assembly.get("native_run") is not False or assembly.get("status")!="assembled_owner_cli_validated_native_not_run":raise RuntimeError("candidate run is not bound to a non-fixture preflight")
 scene_manifest=Path(assembly["scene"]["manifest"]);contact_payload=Path(assembly["scene"]["support_payload"])
 if sha(scene_manifest)!=assembly["scene"]["manifest_sha256"] or sha(contact_payload)!=assembly["scene"]["support_sha256"]:raise RuntimeError("assembled scene/NHCNT changed after preflight")
 for path,digest in assembly["source_sha256"].items():
  if ck(path,digest)!=digest:raise RuntimeError("preflight input changed: "+path)
 repo(V,P["viewer_commit"])
 native_receipt=Path(assembly["native_receipt"]["path"]).resolve()
 if sha(native_receipt)!=assembly["native_receipt"]["sha256"]:raise RuntimeError("selected composed anatomy receipt changed")
 expected_runtime=Path("/Users/n/numi-human-performance-build-014/lib/libmetalrobo.dylib").resolve()
 expected_dyld=os.pathsep.join((str((B/"lib").resolve()),str((B/"matter").resolve())))
 expected_assets={**anatomy_assets_from_assembly(assembly),
  str((B/"bin/numi-human-native").resolve()):P["binary"],str((B/"matter/shaders/HumanRespiration.metallib").resolve()):P["resp_metallib"],str(expected_runtime):P["lib"]}
 compare_contract=validate_compare_contract(inv,m,assembly,expected_assets,expected_dyld,str(expected_runtime),P["lib"])
 av=inv["argv"]
 terminal_proof=verify_terminal_log(run,10000)
 retired_alias_proof=verify_retired_alias_visibility_log(run)
 trace=compare_cadence(REF/"resting-coupled.csv",run/"resting-coupled.csv",10000,8)
 poses={}
 for s in [0,4991,5375,5759,6111,6495,7743,10000]:
  a=json.loads((REF/"accepted-geometry"/f"step-{s}.receipt.json").read_text());b=json.loads((run/"accepted-geometry"/f"step-{s}.receipt.json").read_text())
  if a.get("accepted_step")!=s or b.get("accepted_step")!=s or b.get("physical_endpoint")!="accepted":raise RuntimeError("invalid capture receipt at "+str(s))
  aa={int(q["body_index"]):q for q in a["accepted_registered_body_poses"]};bb={int(q["body_index"]):q for q in b["accepted_registered_body_poses"]}
  if set(aa)!=set(bb) or len(bb)!=86:raise RuntimeError("body pose identities differ")
  pd=[abs(aa[i]["position_m"][j]-bb[i]["position_m"][j]) for i in aa for j in range(3)];qd=[abs(aa[i]["quaternion_xyzw"][j]-bb[i]["quaternion_xyzw"][j]) for i in aa for j in range(4)]
  poses[str(s)]={"body_count":86,"body_state_hash_equal":a.get("accepted_body_state_sha256")==b.get("accepted_body_state_sha256"),"max_position_delta_m":max(pd),"rms_position_delta_m":math.sqrt(sum(v*v for v in pd)/len(pd)),"max_quaternion_component_delta":max(qd)}
 out.mkdir(parents=True)
 result={"schema":"numi.human.final-20s-vs-931-comparison.v1","status":"measured_no_parity_assumed","reference":str(REF),"reference_csv_sha256":sha(REF/"resting-coupled.csv"),"candidate":str(run),"candidate_csv_sha256":sha(run/"resting-coupled.csv"),"reference_roots":10000,"candidate_observations":1250,"terminal_log_proof":terminal_proof,"retired_alias_inspection_policy":retired_alias_proof,"invocation_contract":compare_contract,"trace_cadence_comparison":trace,"captured_86_body_poses":poses,"q_trace_limit":"corrected q0 run omits full generalized-q history; only accepted registered poses at eight capture IDs are compared","qualification":"20 s regression only"}
 wj(out/"comparison.json",result)
 print(json.dumps({"status":result["status"],"out":str(out),"capture_steps":list(poses),"trace_observations":trace["candidate_rows"],"cadence_fields_exact":trace["all_values_exact"],"generalized_q":"not emitted under q0"},indent=2))
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--reference-self-test",action="store_true");ap.add_argument("--cadence-self-test",action="store_true");ap.add_argument("--fixture-legacy-907",action="store_true");ap.add_argument("--skin",type=Path);ap.add_argument("--skin-sha");ap.add_argument("--manifest",type=Path);ap.add_argument("--manifest-sha");ap.add_argument("--anatomy",type=Path);ap.add_argument("--anatomy-sha");ap.add_argument("--anatomy-receipt",type=Path);ap.add_argument("--anatomy-receipt-sha");ap.add_argument("--out",type=Path);ap.add_argument("--compare-run",type=Path);ap.add_argument("--compare-out",type=Path);a=ap.parse_args()
 if a.cadence_self_test:
  print(json.dumps({"status":"q0_com8_semantics_pass","comparison":cadence_self_test()},indent=2));return 0
 if a.reference_self_test:
  sys.path.insert(0,str(H/"src"))
  for p,k in [(OLD/"resting-supine-scene.manifest.json","old_scene"),(OLD/"myosim-fullbody-resting-bed-support.nhcnt","old_contact"),(REF/"run-metadata.json","refmeta"),(REF/"resting-coupled.csv","refcsv")]:ck(p,P[k])
  d=json.loads((OLD/"resting-supine-scene.manifest.json").read_text());c=parse_contact(OLD/"myosim-fullbody-resting-bed-support.nhcnt")
  print(json.dumps({"status":"pinned_907_931_inputs_pass","old_root":d["pose"],"old_support_count":len(c["rows"]),"old_support_vertex_ids":[r["vertex_index"] for r in c["rows"]]},indent=2));return 0
 if a.compare_run:
  if not a.compare_out:ap.error("--compare-out required")
  compare(a.compare_run,a.compare_out);return 0
 if not all([a.skin,a.skin_sha,a.manifest,a.manifest_sha,a.out]):ap.error("--skin --skin-sha --manifest --manifest-sha --out required")
 assemble(a);return 0
if __name__=="__main__":raise SystemExit(main())
