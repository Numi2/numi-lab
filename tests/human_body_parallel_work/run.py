#!/usr/bin/env python3
"""Compare production GPU algorithms and native transaction controls on Apple Metal."""
import argparse,hashlib,json,os,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[2]
def sha(p):
    with pathlib.Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matter-metallib",type=pathlib.Path,required=True)
    ap.add_argument("--respiration-metallib",type=pathlib.Path,required=True)
    ap.add_argument("--output",type=pathlib.Path,required=True)
    ap.add_argument("--native-library",type=pathlib.Path)
    ap.add_argument("--material",type=pathlib.Path)
    args=ap.parse_args();out=args.output.resolve()
    if out==ROOT or ROOT in out.parents:ap.error("outputs must be outside source checkout")
    out.mkdir(parents=True,exist_ok=True)
    record={"status":"FAIL","scope":"GPU algorithm parity and isolated native transactions; no full-Human qualification","stages":[]}
    files=["matter/src/numi_human.mm","matter/src/runtime.mm","matter/tools/human_resting_runtime.hpp",
           "matter/src/human_respiration.metal","matter/src/metal/numi_human.metalinc",
           "matter/src/metal/fgmres.metalinc","matter/src/metal/vascular.metalinc","matter/src/metal/topology_mutation.metalinc",
           "tests/human_body_parallel_work/parallel_work_test.mm","tests/human_body_parallel_work/run.py"]
    record["source_sha256"]={s:sha(ROOT/s) for s in files}
    record["metallib_sha256"]={str(p.resolve()):sha(p) for p in (args.matter_metallib,args.respiration_metallib)}
    def run(name,argv,env=None):
        start=time.monotonic()
        with (out/(name+".stdout.log")).open("w") as o,(out/(name+".stderr.log")).open("w") as e:
            result=subprocess.run(list(map(str,argv)),cwd=ROOT,env=env,stdout=o,stderr=e,timeout=300)
        row={"name":name,"argv":list(map(str,argv)),"exit_code":result.returncode,"elapsed_s":time.monotonic()-start}
        if env:row["NUMI_overrides"]={k:v for k,v in env.items() if k.startswith("NUMI_")}
        record["stages"].append(row);print(json.dumps(row),flush=True)
        (out/"run-receipt.json").write_text(json.dumps(record,indent=2)+"\n")
        if result.returncode:raise RuntimeError(name+" failed; inspect "+str(out/(name+".stderr.log")))
    cpp=["/usr/bin/clang++","-std=c++23","-O3","-Wall","-Wextra","-Werror","-fobjc-arc",
         "-I",ROOT/"include","-I",ROOT/"matter/include","-I",ROOT/"matter/src"]
    try:
        run("device",["system_profiler","SPHardwareDataType","SPDisplaysDataType"])
        run("build",cpp+[ROOT/"tests/human_body_parallel_work/parallel_work_test.mm",
             "-framework","Foundation","-framework","Metal","-o",out/"parallel-work-test"])
        # Whole-command GPU timestamps, no encoder splitting/counter instrumentation.
        # Each pass executes both original and new algorithms on identical data.
        for pair in range(3):
            run("matched-"+str(pair),[out/"parallel-work-test",args.matter_metallib.resolve(),args.respiration_metallib.resolve()])
        if args.native_library:
            material=args.material or ROOT/"matter/materials/silicone.nmatter"
            run("native-build",cpp+[ROOT/"tests/numi_human_fem_surface_attachment/native_initialize_test.mm",
                args.native_library.resolve(),"-Wl,-rpath,"+str(args.native_library.resolve().parent),
                "-framework","Foundation","-framework","Metal","-o",out/"native-test"])
            env={k:v for k,v in os.environ.items() if not k.startswith("NUMI_")}
            env.update(NUMI_HUMAN_PARALLEL_NODE_WORK="1",NUMI_HUMAN_SPARSE_ANCHOR_PROJECTION="1",
                       NUMI_MATTER_PARALLEL_WORKING_SET_CLEAR="1",NUMI_MATTER_SPARSE_TOPOLOGY_INCIDENCE="1")
            for mode in ("inactive","active"):
                run("native-"+mode,[out/"native-test",material.resolve(),args.matter_metallib.resolve()]+(["active-contact"] if mode=="active" else []),env)
            record["native_library_sha256"]=sha(args.native_library)
        record["status"]="PASS"
    finally:
        record["binary_sha256"]={p.name:sha(p) for p in (out/"parallel-work-test",out/"native-test") if p.is_file()}
        (out/"run-receipt.json").write_text(json.dumps(record,indent=2)+"\n")
if __name__=="__main__":main()
