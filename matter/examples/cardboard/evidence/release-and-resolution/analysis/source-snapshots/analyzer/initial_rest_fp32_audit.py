#!/usr/bin/env python3
"""CPU audit of the explicit strip's initial Ds * rounded-inverse FEM F path.

This is a precision-scale diagnostic only. It does not reproduce Metal's exact
instruction selection, invoke the runtime, or establish that roundoff caused a
solver rejection.
"""
from __future__ import annotations
import argparse
import gzip
import json
import hashlib
import math
import re
import struct
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def parameter_map(path: Path) -> dict[str, float]:
    text=path.read_text()
    return {name:float(value) for name,value in re.findall(
        r"parameter\s+(\w+)\s*:[^=;]+?=\s*([-+0-9.eE]+)\s*;", text)}


def read_json(path: Path) -> dict[str, Any]:
    opener=gzip.open if path.suffix==".gz" else open
    with opener(path,"rt",encoding="utf-8") as stream:
        return json.load(stream)


def inverse3(a: list[float]) -> list[float]:
    x0,x1,x2,x3,x4,x5,x6,x7,x8=a
    det=x0*(x4*x8-x5*x7)-x1*(x3*x8-x5*x6)+x2*(x3*x7-x4*x6)
    if not math.isfinite(det) or det==0.0: raise ValueError("singular reference tetrahedron")
    r=1.0/det
    return [(x4*x8-x5*x7)*r,(x2*x7-x1*x8)*r,(x1*x5-x2*x4)*r,
            (x5*x6-x3*x8)*r,(x0*x8-x2*x6)*r,(x2*x3-x0*x5)*r,
            (x3*x7-x4*x6)*r,(x1*x6-x0*x7)*r,(x0*x4-x1*x3)*r]


def matmul(a: list[float], b: list[float], mode: str) -> list[float]:
    out=[]
    for row in range(3):
        for col in range(3):
            terms=[(a[3*row+k],b[3*k+col]) for k in range(3)]
            if mode=="serial_fma":
                # Python binary64 exactly represents a product of two FP32
                # inputs; round once per fused multiply-add as FP32 does.
                acc=f32(terms[0][0]*terms[0][1])
                acc=f32(terms[1][0]*terms[1][1]+acc)
                acc=f32(terms[2][0]*terms[2][1]+acc)
            elif mode=="rounded_products_adds":
                acc=f32(f32(terms[0][0]*terms[0][1])+f32(terms[1][0]*terms[1][1]))
                acc=f32(acc+f32(terms[2][0]*terms[2][1]))
            else:
                acc=sum(x*y for x,y in terms)
            out.append(acc)
    return out


def transpose(a: list[float]) -> list[float]:
    return [a[3*c+r] for r in range(3) for c in range(3)]


def frame_matrix(quaternion: list[float]) -> list[float]:
    x,y,z,w=(f32(v) for v in quaternion)
    norm=f32(math.sqrt(f32(f32(x*x+y*y)+f32(z*z+w*w))))
    scale=f32(1.0/norm)
    x,y,z,w=(f32(v*scale) for v in (x,y,z,w))
    def twice(v: float) -> float: return f32(2.0*f32(v))
    return [f32(1.0-twice(y*y+z*z)),twice(x*y-z*w),twice(x*z+y*w),
            twice(x*y+z*w),f32(1.0-twice(x*x+z*z)),twice(y*z-x*w),
            twice(x*z-y*w),twice(y*z+x*w),f32(1.0-twice(x*x+y*y))]


def deformation_for_tet(points: list[list[float]], tet: list[int], mode: str) -> list[float]:
    # Runtime nodal coordinates are FP32. Compiler promotes those exact stored
    # values to FP64, constructs Dm and its inverse, then serializes inverse
    # coefficients back to FP32. Metal forms Ds from FP32 coordinate differences.
    p=[[f32(v) for v in points[i]] for i in tet]
    p0=p[0]
    ds_gpu=[f32(p[c+1][r]-p0[r]) for r in range(3) for c in range(3)]
    ds_cooker=[p[c+1][r]-p0[r] for r in range(3) for c in range(3)]
    inv_gpu=[f32(v) for v in inverse3(ds_cooker)]
    return matmul(ds_gpu,inv_gpu,mode)


def stress_norm_mpa(f: list[float], q: list[float], p: dict[str,float], model: str) -> tuple[float,float,float]:
    fm=matmul(f,frame_matrix(q),"serial_fma")
    c=matmul(transpose(fm),fm,"serial_fma")
    e=[0.5*(c[0]-1.0),0.5*(c[4]-1.0),0.5*(c[8]-1.0),0.5*c[5],0.5*c[2],0.5*c[1]]
    if model=="orthotropic_stvk":
        c11,c22,c33,c12,c13,c23=(p[k] for k in ("c11","c22","c33","c12","c13","c23"))
        g12,g13,g23=(p[k] for k in ("g12","g13","g23"))
        s=[c11*e[0]+c12*e[1]+c13*e[2],c12*e[0]+c22*e[1]+c23*e[2],
           c13*e[0]+c23*e[1]+c33*e[2],2*g23*e[3],2*g13*e[4],2*g12*e[5]]
    else:
        lam,mu=p["lambda"],p["mu"]
        tr=e[0]+e[1]+e[2]
        s=[lam*tr+2*mu*e[0],lam*tr+2*mu*e[1],lam*tr+2*mu*e[2],
           2*mu*e[3],2*mu*e[4],2*mu*e[5]]
    # P_local = F_local*S. Right rotation to world leaves its Frobenius norm unchanged.
    stress=[[fm[3*r]*s[0]+fm[3*r+1]*s[5]+fm[3*r+2]*s[4],
             fm[3*r]*s[5]+fm[3*r+1]*s[1]+fm[3*r+2]*s[3],
             fm[3*r]*s[4]+fm[3*r+1]*s[3]+fm[3*r+2]*s[2]] for r in range(3)]
    pnorm=math.sqrt(sum(v*v for row in stress for v in row))
    enorm=math.sqrt(e[0]**2+e[1]**2+e[2]**2+2*(e[3]**2+e[4]**2+e[5]**2))
    qhill=float("nan")
    if "sigma_ref" in p and all(k in p for k in ("hill_F","hill_G","hill_H","hill_L","hill_M","hill_N")):
        h=[p[f"hill_{k}"] for k in "FGHLMN"]
        s11,s22,s33,s23,s13,s12=s
        q2=h[0]*(s22-s33)**2+h[1]*(s33-s11)**2+h[2]*(s11-s22)**2+2*h[3]*s23*s23+2*h[4]*s13*s13+2*h[5]*s12*s12
        qhill=math.sqrt(max(0.0,q2))/p["sigma_ref"]
    return enorm,pnorm,qhill


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--mesh",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--materials",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    mesh=read_json(args.mesh); manifest=read_json(args.manifest)
    params={}
    for item in manifest["materials"]["cells"]:
        name=item["name"]
        src=args.materials/(name+".nmatter")
        p=parameter_map(src)
        kind="orthotropic_stvk" if name.startswith("hajali2009_") else "isotropic_small_strain_tangent"
        params[int(item["index"])]=(name,p,kind)
    modes=("serial_fma","rounded_products_adds","fp64_product_of_rounded_operands")
    stats={mode:{idx:{"name":name,"tetrahedra":0,"max_F_minus_I_frobenius":0.0,
                      "max_Green_strain_frobenius":0.0,"max_first_Piola_frobenius_MPa":0.0,
                      "max_first_Piola_frobenius_Pa":0.0,"max_Hill_ratio":0.0}
                 for idx,(name,_,_) in params.items()} for mode in modes}
    for ti,tet in enumerate(mesh["tetrahedra"]):
        idx=int(mesh["material_indices"][ti]); name,p,model=params[idx]
        for mode in modes:
            fmode="exact" if mode=="fp64_product_of_rounded_operands" else mode
            f=deformation_for_tet(mesh["nodes_m"],tet,fmode)
            delta=[f[i]-(1.0 if i in (0,4,8) else 0.0) for i in range(9)]
            ferr=math.sqrt(sum(v*v for v in delta))
            enorm,pnorm,qhill=stress_norm_mpa(f,mesh["material_frames_xyzw"][ti],p,model)
            row=stats[mode][idx];row["tetrahedra"]+=1
            row["max_F_minus_I_frobenius"]=max(row["max_F_minus_I_frobenius"],ferr)
            row["max_Green_strain_frobenius"]=max(row["max_Green_strain_frobenius"],enorm)
            row["max_first_Piola_frobenius_MPa"]=max(row["max_first_Piola_frobenius_MPa"],pnorm)
            row["max_first_Piola_frobenius_Pa"]=max(row["max_first_Piola_frobenius_Pa"],pnorm*1e6)
            if math.isfinite(qhill): row["max_Hill_ratio"]=max(row["max_Hill_ratio"],qhill)
    source_files={
        "compiler_impl.cpp":Path("matter/src/compiler_impl.cpp"),
        "fem.metalinc":Path("matter/src/metal/fem.metalinc"),
        "common.metalinc":Path("matter/src/metal/common.metalinc"),
        "mixed_fem.metalinc":Path("matter/src/metal/mixed_fem.metalinc"),
    }
    material_paths={name:args.materials/(name+".nmatter") for name,_,_ in params.values()}
    input_hashes={"mesh":sha256(args.mesh),"manifest":sha256(args.manifest),
                  "materials":{name:sha256(path) for name,path in material_paths.items()},
                  "source":{name:sha256(path) for name,path in source_files.items()},
                  "audit_script":sha256(Path(__file__))}
    report={"schema":"numi.cardboard.initial_rest_fp32_precision_audit.v1",
            "classification":"bounded CPU arithmetic sensitivity estimate; not an execution trace of Metal",
            "source_path":"compiler_impl.cpp cookedNodePosition reads runtime-cooked FP32 node coordinates, promotes them to FP64 for Dm inverse, then serializes inverse as FP32; fem.metalinc tetrahedronDeformation computes FP32 Ds * FP32 inverseRest",
            "input_sha256":input_hashes,
            "mesh":str(args.mesh),"mesh_resolution":manifest["mesh"].get("resolution"),
            "timestep_or_solver":None,
            "comparison":"At initial reference positions u=0, the mathematically equivalent F=I+grad(u) representation is exactly identity. The Ds*rounded-inverse route is evaluated under two plausible FP32 dot-product roundings plus an FP64 product of rounded operands.",
            "material_model":"Haj-Ali paper materials use the authored orthotropic StVK initial elastic tangent, before any plastic return. Glue uses the source mu/lambda small-strain tangent. Frame rotations are applied to F before stress evaluation.",
            "limitations":["Metal instruction/FMA/reduction order is not reproduced.","The estimate does not show that initial strain caused any failed step.","No solver tolerance, material, or runtime source was changed."],
            "stats_by_product_rounding_and_material":stats,
            "max_across_materials":{mode:{"max_F_minus_I_frobenius":max(r["max_F_minus_I_frobenius"] for r in rows.values()),
                "max_Green_strain_frobenius":max(r["max_Green_strain_frobenius"] for r in rows.values()),
                "max_first_Piola_frobenius_Pa":max(r["max_first_Piola_frobenius_Pa"] for r in rows.values())} for mode,rows in stats.items()},
            "physical_validation":False}
    payload=json.dumps(report,indent=2,sort_keys=True)+"\n"
    with args.output.open("x",encoding="utf-8") as stream: stream.write(payload)

if __name__=="__main__": main()
