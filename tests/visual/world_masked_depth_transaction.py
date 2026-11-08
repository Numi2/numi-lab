#!/usr/bin/env python3
"""Regression for mixed-status MetalWorld tracker output and masked depth."""
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def run(probe, root, name, capture):
    env=os.environ.copy()
    for key in ("NUMI_TEST_NO_CAPTURE", "NUMI_TEST_REJECT_ACTION",
                "NUMI_TEST_REJECT_AFTER_STEP", "NUMI_TEST_RESIDENT_REJECT",
                "NUMI_TEST_WORLD_STEP_CALLBACK"):
        env.pop(key, None)
    env["NUMI_TEST_WORLD_TRANSACTION"]="1"
    env["NUMI_TEST_RESULT_PREFIX"]=str(root/name)
    if not capture:
        env["NUMI_TEST_NO_CAPTURE"]="1"
    p=subprocess.run([probe],env=env,text=True,capture_output=True,timeout=120)
    require(p.returncode==0, f"{name}: exit {p.returncode}\n{p.stdout}\n{p.stderr}")
    meta=re.search(r"depth_width=(\d+) depth_height=(\d+) depth_features=(\d+)",p.stdout)
    width=re.search(r"output_per_env=(\d+)",p.stdout)
    transitions=re.findall(r"transition\[(\d+)\] done=(\d+) timeout=(\d+) physics_error=(\d+) reason=(\d+) reward=([^\n]+)",p.stdout)
    require(meta is not None and width is not None and len(transitions)==60, f"{name}: missing metadata")
    return tuple(map(int,meta.groups())),int(width.group(1)),transitions

def verify(root, meta, width, transitions):
    w,h,features=meta
    pixels=2*w*h
    require(width==102+pixels+features, "unexpected observation layout")
    data=(root/"first.observation").read_bytes()
    require(len(data)==30*3*2*width*4, "wrong capture size")
    resets=struct.unpack("<60I",(root/"first.reset").read_bytes())
    errors=[int(item[3])!=0 for item in transitions]
    counts={key:0 for key in ("accepted","rejected_reset","rejected_nonreset","accepted_depth_changed",
        "rejected_depth_changed","accepted_visible","rejected_visible","mixed")}
    track={frame*34+i for frame in range(3) for i in range(7)}
    far=struct.pack("<f",1.)*pixels
    zero_features=bytes(features*4)
    def row(step,phase,env):
        start=((step*3+phase)*2+env)*width*4
        return memoryview(data)[start:start+width*4]
    for step in range(30):
        counts["mixed"]+=errors[step*2]!=errors[step*2+1]
        for env in range(2):
            before,candidate,after=(row(step,phase,env) for phase in range(3))
            idx=step*2+env
            fail=errors[idx]
            depth=candidate[102*4:(102+pixels)*4]
            visible=any(value[0]<1. for value in struct.iter_unpack("<f",depth))
            counts["rejected_visible" if fail else "accepted_visible"]+=visible
            changed=before[102*4:]!=candidate[102*4:]
            if not fail:
                counts["accepted"]+=1
                counts["accepted_depth_changed"]+=changed
                require(after==candidate,f"accepted output changed: {step}/{env}")
            elif not resets[idx]:
                counts["rejected_nonreset"]+=1
                counts["rejected_depth_changed"]+=changed
                require(after==before,f"rejected output changed: {step}/{env}")
            else:
                counts["rejected_reset"]+=1
                counts["rejected_depth_changed"]+=changed
                for i in range(102):
                    expected=bytes(4) if i in track else before[i*4:(i+1)*4]
                    require(after[i*4:(i+1)*4]==expected, f"reset prefix: {step}/{env}/{i}")
                require(after[102*4:(102+pixels)*4]==far, f"reset depth: {step}/{env}")
                require(after[(102+pixels)*4:]==zero_features, f"reset features: {step}/{env}")
    require(all(counts.values()),f"vacuous fixture: {counts}")
    return counts

def main():
    probe=sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="numi-world-depth-") as tmp:
        root=Path(tmp)
        first=run(probe,root,"first",True)
        repeat=run(probe,root,"repeat",True)
        control=run(probe,root,"control",False)
        require(first==repeat==control,"transition or layout mismatch")
        for extension in ("q","v","scene","actor","status","reset","history","observation"):
            require((root/("first."+extension)).read_bytes()==(root/("repeat."+extension)).read_bytes(),
                    "repeat mismatch: "+extension)
        for extension in ("q","v","scene","actor","status","reset"):
            require((root/("first."+extension)).read_bytes()==(root/("control."+extension)).read_bytes(),
                    "capture changed result: "+extension)
        print(verify(root,*first))

if __name__=="__main__":
    main()
