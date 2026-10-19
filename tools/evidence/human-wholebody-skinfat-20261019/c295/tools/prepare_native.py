import pathlib,json,hashlib,sys
root=pathlib.Path(__file__).resolve().parent
base=pathlib.Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/native-physical-root-001")
build=root/sys.argv[1]; target=root/sys.argv[2]
# Optional matched guarded runtime receipt; all shader aliases must agree.
runtime_receipt=pathlib.Path(sys.argv[4]) if len(sys.argv)>4 else None
assert not target.exists()
(target/"output").mkdir(parents=True)
inv=json.loads((base/"invocation.json").read_text())
b=json.loads((build/"build-receipt.json").read_text())
assert b["compile_exit"]==0 and b["link_exit"]==0
inv["argv"][0]=b["binary_path"];inv["binary_sha256"]=b["binary_sha256"]
inv["argv"][4]=str(target/"output")
if len(sys.argv)>3:inv["argv"][inv["argv"].index("--muscle-step-count")+1]=sys.argv[3]
inv["environment"]["NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM"]="1"
inv["cwd"]=str(target)
inv["environment"]["NUMI_HUMAN_SKIN_REFERENCE_DUMP"]=str(target/"registered-skin.csv")
inv["environment"]["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(target/"common-field-failure.json")
if runtime_receipt:
    runtime=json.loads(runtime_receipt.read_text())
    lib=runtime_receipt.parent/"libmetalrobo.dylib"
    msl=pathlib.Path(runtime["compile_argv"][next(i for i,x in enumerate(runtime["compile_argv"]) if x.startswith("-DNUMI_HUMAN_TENDON_FEM_TEST_SOURCE_PATH="))].split("=",1)[1].strip('"'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(lib)==runtime["library_sha256"] and sha(msl)==runtime["msl_sha256"]
    inv.update(library=str(lib),library_path=str(lib),library_sha256=sha(lib),msl_path=str(msl),msl_sha256=sha(msl),guarded_build_receipt=str(runtime_receipt))
    inv["environment"]["DYLD_LIBRARY_PATH"]=str(lib.parent)
    for name in ("NUMI_MATTER_TEST_SOURCE","NUMI_MATTER_SOURCE"):
        inv["environment"][name]=str(msl)
        inv["environment"][name+"_SHA256"]=sha(msl)
    inv["environment"]["NUMI_MATTER_METALLIB"]=str(msl)

inv["launcher_argv"]=inv["launcher_argv"][:inv["launcher_argv"].index("--")+1]+inv["argv"]
inv["base_invocation"]=str(base/"invocation.json")
# Historical metadata belongs to base_invocation, not to this new binary/runtime.
for key in ("candidate_runtime_source_sha256","candidate_runtime_object_sha256","scope","base","purpose"):
    inv.pop(key,None)
inv["qualification_scope"]="Whole-body layer native numerical/contact probe; rate-0 diagnostic unless explicitly changed; not sustained stability or performance qualification"
inv["purpose"]="Exercise the admitted whole-body layers and existing tissue/contact owners in one native scene."
inv["passive_fem_max_rate_exponent"]=int(inv["environment"]["NUMI_HUMAN_PASSIVE_FEM_MAX_RATE_EXPONENT"])
inv["simulation_dt_s"]=float(inv["argv"][inv["argv"].index("--muscle-step-seconds")+1])
inv["requested_steps"]=int(inv["argv"][inv["argv"].index("--muscle-step-count")+1])
inv["expected_registered_skin_csv_sha256"]="4dd63097a035f5f385712307f2336ef6fd46c964d1b93a2e85d84193ac12d327"
inv["source_revision"]="16c27b39a9f581ef2b01f18243b80bdee7635af1 plus pinned private wholebody build inputs"
(target/"invocation.json").write_text(json.dumps(inv,indent=2)+"\n")
(target/"run.py").write_text((base/"run.py").read_text())
print(target)
