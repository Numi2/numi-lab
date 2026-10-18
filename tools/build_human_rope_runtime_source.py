#!/usr/bin/env python3
"""Assemble the exact minimal Metal source for runtime rope geometry."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def structure(header: str, name: str) -> str:
    end=header.find("} "+name+";")
    if end < 0:
        raise SystemExit(f"missing structure terminator: {name}")
    begin=header.rfind("typedef struct ",0,end)
    if begin < 0:
        raise SystemExit(f"missing structure declaration: {name}")
    return header[begin:end+len("} "+name+";")]

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--visual-kernel", type=Path)
    ap.add_argument("--visual-header", type=Path)
    args=ap.parse_args()
    repo=args.repo.resolve()
    members={
      "engine_types.h": repo/"include/metalrobo/engine_types.h",
      "numi_human_stand_gpu.h": repo/"include/metalrobo/numi_human_stand_gpu.h",
      "numi_human_resting_rope_gpu.h": repo/"include/metalrobo/numi_human_resting_rope_gpu.h",
      "numi_human_tendon_gpu.h": repo/"include/metalrobo/numi_human_tendon_gpu.h",
      "NumiHumanRestingRope.metalinc": repo/"matter/src/NumiHumanRestingRope.metalinc",
      "NumiHumanSeparation.metalinc": repo/"matter/src/NumiHumanSeparation.metalinc",
      "NumiHumanSeparationForce.metalinc": repo/"matter/src/NumiHumanSeparationForce.metalinc",
    }
    if bool(args.visual_kernel) != bool(args.visual_header):
        raise SystemExit("visual kernel and header must be supplied together")
    if args.visual_kernel:
        members["NumiHumanRestingRopeVisual.metalinc"]=args.visual_kernel.resolve()
        members["NumiHumanRestingRopeVisual.hpp"]=args.visual_header.resolve()
        members["numi_human_resting_rope_visual_gpu.h"]=repo/"include/metalrobo/numi_human_resting_rope_visual_gpu.h"
        members["visual_platform_types.h"]=repo/"include/metalrobo/visual_platform_types.h"
    for name,path in members.items():
        if not path.is_file():
            raise SystemExit(f"missing pinned source input: {path}")
    raw={name:path.read_bytes() for name,path in members.items()}
    engine=raw["engine_types.h"].decode("utf-8")
    stand=raw["numi_human_stand_gpu.h"].decode("utf-8")
    rope=raw["numi_human_resting_rope_gpu.h"].decode("utf-8")
    tendon=raw["numi_human_tendon_gpu.h"].decode("utf-8")
    pieces=[
      "#include <metal_stdlib>\nusing namespace metal;\n"
      "#define MR_ALIGN16 alignas(16)\n"
      "using mr_u32=uint;using mr_uint4=uint4;using mr_float4=float4;\n",
      structure(engine,"MRArticulatedBodyPoseGPU"),
      structure(engine,"MRArticulatedPointWorldGPU"),
      structure(stand,"MRNumiHumanStandContactGPU"),
      structure(rope,"MRHumanRestingRopeGPU"),
      "\n".join(line for line in raw["NumiHumanRestingRope.metalinc"].decode("utf-8").splitlines()
                     if not line.lstrip().startswith("#include")),
      structure(tendon,"MRNumiHumanTendonBindingGPU"),
      structure(tendon,"MRNumiHumanTendonTransferResultGPU"),
      "\n".join(line for line in raw["NumiHumanSeparation.metalinc"].decode("utf-8").splitlines()
                     if not line.lstrip().startswith("#include")),
      "\n".join(line for line in raw["NumiHumanSeparationForce.metalinc"].decode("utf-8").splitlines()
                     if not line.lstrip().startswith("#include")),
    ]
    if args.visual_kernel:
        visual_gpu=raw["numi_human_resting_rope_visual_gpu.h"].decode("utf-8")
        visual_platform=raw["visual_platform_types.h"].decode("utf-8")
        visual_kernel=raw["NumiHumanRestingRopeVisual.metalinc"].decode("utf-8")
        pieces[1:1]=[structure(engine,"MRBodyStateGPU"),
                     structure(visual_platform,"MRVisualVertexGPUV2"),
                     structure(visual_gpu,"MRHumanRestingRopeVisualGPU")]
        pieces.append("\n".join(line for line in visual_kernel.splitlines()
                                   if not line.lstrip().startswith("#include")))
    output="\n".join(pieces).encode("utf-8")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(output)
    manifest={
      "schema":"numi.human.rope-runtime-metal-source.v1",
      "output_path":str(args.output.resolve()),
      "output_sha256":sha(output),
      "output_bytes":len(output),
      "inputs":{name:{"path":str(path.resolve()),"sha256":sha(raw[name]),"bytes":len(raw[name])}
                for name,path in members.items()},
      "structs":["MRArticulatedBodyPoseGPU","MRArticulatedPointWorldGPU",
                 "MRNumiHumanStandContactGPU","MRHumanRestingRopeGPU",
                 "MRNumiHumanTendonBindingGPU","MRNumiHumanTendonTransferResultGPU",
                 "MRHumanSeparationGPU"]+
                (["MRBodyStateGPU","MRVisualVertexGPUV2","MRHumanRestingRopeVisualGPU"]
                 if args.visual_kernel else []),
      "kernels":["nm_human_resting_rope_support",
                 "nm_human_separation_constraints",
                 "mr_numi_human_separation_release_crossing_forces"]+
                (["nm_human_resting_rope_visual"] if args.visual_kernel else []),
      "metal_language":"4.0",
      "math_mode":"safe",
      "math_floating_point_functions":"precise",
    }
    manifest_path=args.output.with_suffix(args.output.suffix+".manifest.json")
    manifest_path.write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,sort_keys=True))
    return 0
if __name__=="__main__":
    raise SystemExit(main())
