#!/usr/bin/env python3
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
evidence=root/"native-sustained-capacity-001"
run=evidence/"run-014"
run.mkdir(parents=True,exist_ok=True)
source=Path("/Users/n/numi-human-separation-20261018/C/deformable-contact-finalizer-233/source/matter/src/MatterSource.metal")
lib=Path("/Users/n/numi-human-separation-20261018/C/deformable-contact-finalizer-233/build/libmetalrobo.dylib")
probe=evidence/"sustained_capacity_probe"
wrapper=Path("/Users/n/numi-human-final-integration-publish-1397/tools/run_resting_job_exclusive.py")
runner=Path("/Users/n/numi-human-prep-venv-20261005/bin/python3")
reference=Path("/Users/n/numi-human-resting-evidence-20261005/reference-circulation-001/resting_reference_lv15.native.v3.json")
respiration=Path("/Users/n/numi-human-resting-evidence-20261005/native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-reference-respiration.json")
material=Path("/Users/n/numi-human-separation-20261018/C/bleeding-liquid-205/main-owner-206/matter/materials/blood_high_shear_newtonian_proxy.nmatter")
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cmd=[str(runner),str(wrapper),"--","/usr/bin/time","-l",str(probe),str(reference),str(respiration),str(material)]
env=os.environ.copy()
env["NUMI_MATTER_TEST_SOURCE"]=str(source)
env["NUMI_MATTER_TEST_SOURCE_SHA256"]=sha(source)
fixture={"NUMI_HUMAN_BLOOD_MPM_ENABLED":"1","NUMI_HUMAN_BLOOD_MPM_CONTACT_ENABLED":"0","NUMI_HUMAN_BLOOD_MPM_MAX_RATE_EXPONENT":"0","NUMI_HUMAN_RESTING_TIMESTEP_SENSITIVITY":"1","NUMI_SUSTAINED_CAPACITY_ONLY":"1","NUMI_SUSTAINED_FLOOR_ONLY":"0"}
env.update(fixture)
inputs=[reference,respiration,material,lib,source,probe,evidence/"HumanRespirationSustained.metal",evidence/"shader-flatten-receipt.json",root/"source/matter/src/human_respiration.metal",root/"source/matter/include/numi/matter/human_respiration.h",root/"source/matter/tools/human_resting_runtime.hpp",root/"source-pins.json"]
inv={"schema":"numi.sustained_blood_capacity_invocation.v1","command":cmd,"environment":{k:env[k] for k in ["NUMI_MATTER_TEST_SOURCE","NUMI_MATTER_TEST_SOURCE_SHA256"]},"fixture_settings":{**fixture,"frame_dt_s":0.00025,"microticks_per_command_buffer_max":1,"capacity_contact_enabled":False,"floor_contact_enabled":False,"capacity_frames":132,"capacity_only":True},"inputs_sha256":{str(p):sha(p) for p in inputs},"planned_runs":[{"duration_s":0.033,"frame_dt_s":0.00025,"frames":132,"offset_x_m":0.0,"contact_enabled":False}],"scope":"synthetic accepted external packets integrated by existing MPM; not physiological bleeding rate"}
(run/"invocation.json").write_text(json.dumps(inv,indent=2,sort_keys=True)+"\n")
for p,n in [(probe,"sustained_capacity_probe"),(lib,"libmetalrobo.dylib"),(source,"MatterSource.metal"),(evidence/"HumanRespirationSustained.metal","HumanRespirationSustained.metal"),(evidence/"shader-flatten-receipt.json","shader-flatten-receipt.json"),(root/"source/matter/src/human_respiration.metal","human_respiration.metal"),(root/"source/matter/include/numi/matter/human_respiration.h","human_respiration.h"),(root/"source/matter/tools/human_resting_runtime.hpp","human_resting_runtime.hpp"),(root/"source-pins.json","source-pins.json"),(material,"blood_high_shear_newtonian_proxy.nmatter")]:shutil.copy2(p,run/n)
timeout_seconds=300.0
start=time.monotonic()
timed_out=False
with (run/"stdout.log").open("w") as log:
    proc=subprocess.Popen(cmd,env=env,text=True,stdout=log,stderr=subprocess.STDOUT,
                          start_new_session=True)
    try:
        exit_code=proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out=True
        os.killpg(proc.pid,signal.SIGTERM)
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            proc.wait()
        exit_code=124
elapsed=time.monotonic()-start
output=(run/"stdout.log").read_text()
res={"schema":"numi.sustained_blood_capacity_result.v1","exit_code":exit_code,"timed_out":timed_out,"timeout_seconds":timeout_seconds,"elapsed_seconds":elapsed,"stdout_sha256":hashlib.sha256(output.encode()).hexdigest(),"inputs_sha256":inv["inputs_sha256"],"output":output}
(run/"result.json").write_text(json.dumps(res,indent=2,sort_keys=True)+"\n")
print(json.dumps({"exit_code":exit_code,"timed_out":timed_out,"elapsed_seconds":elapsed,"run":str(run),"tail":"\n".join(output.splitlines()[-35:])},indent=2))
raise SystemExit(exit_code)
