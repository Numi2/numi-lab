#!/usr/bin/env python3
"""Audit the captured one-DOF mixed-fixture rigid residual identity."""
import json, math, struct, sys
from pathlib import Path

def first_rhs(path, iteration):
    for line in Path(path).open():
        if not line.startswith("fem_newton_iterate="): continue
        row=json.loads(line.split("=",1)[1])
        if row["iteration"]==iteration and row["stage"]=="fully_assembled_rhs" and row["state_arenas"]:
            return row
    raise ValueError("selected RHS not captured")

def floats(hexstring):
    raw=bytes.fromhex(hexstring)
    if len(raw)%4: raise ValueError("unaligned float arena")
    return struct.unpack("<"+str(len(raw)//4)+"f",raw)

def inspect(row):
    n=row["fem_node_count"]
    if n!=8: raise ValueError("unexpected fixture")
    arenas=row["state_arenas"]
    rhs=floats(arenas["rhs_workspace"])
    candidate=floats(arenas["rigid_candidate"])[0]
    raw=bytes.fromhex(arenas["contact_samples"])
    if len(raw)!=8*160: raise ValueError("unexpected contact layout")
    impulse=0.0
    for i in range(8):
        # impulseAndNormal is the fifth 16-byte lane; xyz is impulse on rigid.
        impulse+=struct.unpack_from("<f",raw,i*160+64+8)[0]
    rigid_rhs=rhs[8*n]
    norm=lambda v:math.sqrt(math.fsum(x*x for x in v))
    return {"rigid_impulse_Ns":impulse,"rigid_candidate_velocity_increment_m_s":candidate,
            "rigid_mass_kg":1.0,"contact_minus_mass_delta_v_Ns":impulse-candidate,
            "captured_rigid_rhs_Ns":rigid_rhs,
            "rhs_components":{"fem_velocity":norm(rhs[i*4+k] for i in range(n) for k in range(3)),
                 "fem_pressure":norm(rhs[i*4+3] for i in range(n)),
                 "fields":norm(rhs[4*n:8*n]),"rigid":norm(rhs[8*n:])}}

a=inspect(first_rhs(sys.argv[1],2)); b=inspect(first_rhs(sys.argv[2],2))
if a!=b: raise ValueError("default and tight traces differ before default acceptance")
if abs(a["contact_minus_mass_delta_v_Ns"]-a["captured_rigid_rhs_Ns"])>1e-12:
    raise ValueError("rigid residual identity failed")
late=inspect(first_rhs(sys.argv[2],15))
print(json.dumps({"default_stopping_configuration":a,"tight_iteration_15":late,
                  "scope":"runtime-assembled legacy mixed fixture, not independent FP64 or midpoint contact"},indent=2))
