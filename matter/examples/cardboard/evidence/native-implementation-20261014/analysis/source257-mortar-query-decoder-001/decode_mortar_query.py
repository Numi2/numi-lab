#!/usr/bin/env python3
"""Independent decoder/replay for source257 mortar complement audit records.

CPU-only. Does not invoke production kernels or change solver inputs.
"""
from __future__ import annotations
import argparse, hashlib, json, math, pathlib, struct, sys
from fractions import Fraction

Q = pathlib.Path("/Users/n/cardboard-implementation-20261014-host-macmini")
SRC = Q / "snapshots/source-mortar-first-uncertified-audit-257-006"
BUILDDIR = Q / "build-source257-mortar-audit-007"
NATIVE = Q / "native"
RECORD_NAME = "actual_mortar_complement_first_uncertified_query"
EPS32 = 1.1920928955078125e-7

def sha(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]
def bits32(x): return f"{struct.unpack('<I', struct.pack('<f', float(x)))[0]:08x}"
def vec_add(a,b): return tuple(x+y for x,y in zip(a,b))
def vec_sub(a,b): return tuple(x-y for x,y in zip(a,b))
def vec_scale(a,s): return tuple(x*s for x in a)
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def norm2(a): return dot(a,a)
def norm(a): return math.sqrt(norm2(a))
def cross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def maximum_abs(a): return max(abs(x) for x in a)

def closest_point_triangle(p,a,b,c):
    # Ericson Voronoi-region construction in binary64, no dimensional thresholds.
    ab=vec_sub(b,a); ac=vec_sub(c,a); ap=vec_sub(p,a)
    d1=dot(ab,ap); d2=dot(ac,ap)
    if d1<=0 and d2<=0: return a,(1.,0.,0.),"A"
    bp=vec_sub(p,b); d3=dot(ab,bp); d4=dot(ac,bp)
    if d3>=0 and d4<=d3: return b,(0.,1.,0.),"B"
    vc=d1*d4-d3*d2
    if vc<=0 and d1>=0 and d3<=0:
        den=d1-d3
        if den==0: raise ValueError("degenerate AB edge")
        v=d1/den
        q=vec_add(a,vec_scale(ab,v))
        return q,(1-v,v,0.),"AB"
    cp=vec_sub(p,c); d5=dot(ab,cp); d6=dot(ac,cp)
    if d6>=0 and d5<=d6: return c,(0.,0.,1.),"C"
    vb=d5*d2-d1*d6
    if vb<=0 and d2>=0 and d6<=0:
        den=d2-d6
        if den==0: raise ValueError("degenerate AC edge")
        w=d2/den
        return vec_add(a,vec_scale(ac,w)),(1-w,0.,w),"AC"
    va=d3*d6-d5*d4
    if va<=0 and d4-d3>=0 and d5-d6>=0:
        den=(d4-d3)+(d5-d6)
        if den==0: raise ValueError("degenerate BC edge")
        w=(d4-d3)/den
        bc=vec_sub(c,b)
        return vec_add(b,vec_scale(bc,w)),(0.,1-w,w),"BC"
    den=va+vb+vc
    if den<=0: raise ValueError("degenerate face")
    v=vb/den; w=vc/den
    q=vec_add(a,vec_add(vec_scale(ab,v),vec_scale(ac,w)))
    return q,(1-v-w,v,w),"face"

def closest_point_triangle_fraction(p,a,b,c):
    # Exact rational Voronoi classification over the captured binary32 input
    # coordinates. Returns exact nearest squared-distance ingredients.
    sub=lambda x,y: tuple(i-j for i,j in zip(x,y))
    add=lambda x,y: tuple(i+j for i,j in zip(x,y))
    scale=lambda x,k: tuple(i*k for i in x)
    dprod=lambda x,y: sum(i*j for i,j in zip(x,y))
    ab=sub(b,a); ac=sub(c,a); ap=sub(p,a)
    d1=dprod(ab,ap); d2=dprod(ac,ap)
    if d1<=0 and d2<=0: return a,(Fraction(1),Fraction(0),Fraction(0)),"A"
    bp=sub(p,b); d3=dprod(ab,bp); d4=dprod(ac,bp)
    if d3>=0 and d4<=d3: return b,(Fraction(0),Fraction(1),Fraction(0)),"B"
    vc=d1*d4-d3*d2
    if vc<=0 and d1>=0 and d3<=0:
        den=d1-d3
        if den==0: raise ValueError("exact degenerate AB")
        v=d1/den; return add(a,scale(ab,v)),(1-v,v,Fraction(0)),"AB"
    cp=sub(p,c); d5=dprod(ab,cp); d6=dprod(ac,cp)
    if d6>=0 and d5<=d6: return c,(Fraction(0),Fraction(0),Fraction(1)),"C"
    vb=d5*d2-d1*d6
    if vb<=0 and d2>=0 and d6<=0:
        den=d2-d6
        if den==0: raise ValueError("exact degenerate AC")
        w=d2/den; return add(a,scale(ac,w)),(1-w,Fraction(0),w),"AC"
    va=d3*d6-d5*d4
    if va<=0 and d4-d3>=0 and d5-d6>=0:
        den=(d4-d3)+(d5-d6)
        if den==0: raise ValueError("exact degenerate BC")
        w=(d4-d3)/den; return add(b,scale(sub(c,b),w)),(Fraction(0),1-w,w),"BC"
    den=va+vb+vc
    if den<=0: raise ValueError("exact degenerate face")
    v=vb/den; w=vc/den
    return add(a,add(scale(ab,v),scale(ac,w))),(1-v-w,v,w),"face"

def exact_endpoint_distance2(p,tri):
    exact=lambda v: tuple(Fraction.from_float(float(x)) for x in v)
    pp=exact(p); aa,bb,cc=(exact(v) for v in tri)
    cp,bary,feature=closest_point_triangle_fraction(pp,aa,bb,cc)
    delta=tuple(x-y for x,y in zip(pp,cp))
    return sum(x*x for x in delta),bary,feature

def fraction_record(v):
    return {"numerator":str(v.numerator),"denominator":str(v.denominator)}

def bary3(xy):
    u,v=xy
    return (u,v,1.0-u-v)

def make_subtri(parent, bary_rows):
    return [tuple(sum(w*parent[j][k] for j,w in enumerate(bary3(row[:2]))) for k in range(3)) for row in bary_rows]

def lerp(a,b,t):
    return tuple(x+(y-x)*t for x,y in zip(a,b))

def validate_record(payload: bytes, descriptor: dict):
    if descriptor.get("dtype") != "record:NMMortarComplementAuditGPU" or descriptor.get("count") != 1 or descriptor.get("element_bytes") != 384 or descriptor.get("byte_count") != 384:
        raise ValueError("record descriptor/layout mismatch")
    off=int(descriptor["offset"]); n=int(descriptor["byte_count"])
    if off<0 or off+n>len(payload): raise ValueError("record outside payload")
    record=payload[off:off+n]
    if sha(record)!=descriptor.get("sha256"): raise ValueError("record digest mismatch")
    return record

def parse_case(run):
    base=NATIVE/f"mortar-inline-observer-on-{run}/diagnostics/capture-coupled-trial-capture"
    jp=base/"capture.coupled.json"; bp=base/"capture.coupled.bin"
    jb=jp.read_bytes(); raw=bp.read_bytes(); d=json.loads(jb)
    desc=[a for a in d["arrays"] if a.get("name")==RECORD_NAME]
    if len(desc)!=1: raise ValueError(f"{run}: expected exactly one audit record")
    a=desc[0]; off=int(a["offset"])
    try: record=validate_record(raw,a)
    except (ValueError,KeyError) as e: raise ValueError(f"{run}: {e}")
    if len(raw)!=int(d["payload"]["byte_count"]) or sha(raw)!=d["payload"]["sha256"]:
        raise ValueError(f"{run}: whole payload digest/length mismatch")
    words=struct.unpack("<16I",record[:64])
    def f4row(i): return struct.unpack_from("<4f",record,64+16*i)
    start=[f4row(i)[:3] for i in range(6)]
    finish=[f4row(6+i)[:3] for i in range(6)]
    left_bary=[f4row(12+i) for i in range(3)]
    right_bary=[f4row(15+i) for i in range(3)]
    result=f4row(18); params=f4row(19)
    return {
        "run":run,"json_path":str(jp),"bin_path":str(bp),"json_sha256":sha(jb),
        "payload_sha256":sha(raw),"payload_bytes":len(raw),"record_sha256":sha(record),
        "capture":d,"descriptor":a,"words":words,"start":start,"finish":finish,
        "left_bary":left_bary,"right_bary":right_bary,"result":result,"params":params,
        "record_raw_hex":record.hex()
    }

def build_query(c):
    w=c["words"]; kind=w[3]; feat=w[12:16]
    if kind!=4 or feat[1]!=1: raise ValueError(f"unsupported query kind/feature {kind}/{feat}")
    li,ri=w[8],w[9]; vi=feat[0]
    if li==0xffffffff or ri==0xffffffff or vi>2: raise ValueError("invalid cell/vertex")
    # For query kinds 3/4, makeMortarComplementAudit receives the already
    # interpolated mortar subtriangle endpoints (subtriangleEndpoints=true).
    # start/finish are therefore direct CCD inputs; bary arrays are maps back
    # to parent coordinates and must not be applied a second time.
    left0=c["start"][:3]
    left1=c["finish"][:3]
    right0=c["start"][3:]
    right1=c["finish"][3:]
    return {"left_cell":li,"right_cell":ri,"vertex_index":vi,
            "point0":right0[vi],"point1":right1[vi],
            "tri0":left0,"tri1":left1,
            "parent_start":c["start"],"parent_finish":c["finish"]}

def effective_threshold(q, thickness):
    scale=max(maximum_abs(q["point0"]),maximum_abs(q["point1"]),
              *(maximum_abs(p) for p in q["tri0"]),
              *(maximum_abs(p) for p in q["tri1"]))
    scale32=f32(scale)
    base=f32(max(f32(thickness),scale32,1.e-12))
    roundoff=f32(f32(8.0*EPS32)*base)
    return f32(f32(thickness)+roundoff),scale32,roundoff

def geom_at(q,t):
    p=lerp(q["point0"],q["point1"],t)
    tri=[lerp(x,y,t) for x,y in zip(q["tri0"],q["tri1"])]
    cp,bary,feature=closest_point_triangle(p,*tri)
    return norm(vec_sub(p,cp)),bary,feature

def f32v(a): return tuple(f32(x) for x in a)
def f32add(a,b): return tuple(f32(x+y) for x,y in zip(a,b))
def f32sub(a,b): return tuple(f32(x-y) for x,y in zip(a,b))
def f32mul(a,s): return tuple(f32(x*s) for x in a)
def f32dot(a,b): return f32(f32(f32(a[0]*b[0])+f32(a[1]*b[1]))+f32(a[2]*b[2]))
def f32len(a): return f32(math.sqrt(max(0.0,f32dot(a,a))))
def f32mix_source_order(a,b,t):
    # Source MSL mix behavior is modeled as x + t*(y-x); this is a CPU
    # shadow, not a claim that Python reproduces Metal contraction semantics.
    return tuple(f32(x+f32(t*f32(y-x))) for x,y in zip(a,b))

def closest_triangle_f32(p,a,b,c):
    ab=f32sub(b,a); ac=f32sub(c,a); ap=f32sub(p,a)
    scale=f32(max(maximum_abs(ab),maximum_abs(ac),maximum_abs(ap)))
    if not math.isfinite(scale): raise ValueError("nonfinite scale")
    if not scale>0: return a,(1.,0.,0.)
    ab=tuple(f32(x/scale) for x in ab); ac=tuple(f32(x/scale) for x in ac); ap=tuple(f32(x/scale) for x in ap)
    d1=f32dot(ab,ap); d2=f32dot(ac,ap)
    if d1<=0 and d2<=0: return a,(1.,0.,0.)
    bp=tuple(f32(x/scale) for x in f32sub(p,b)); d3=f32dot(ab,bp); d4=f32dot(ac,bp)
    if d3>=0 and d4<=d3: return b,(0.,1.,0.)
    vc=f32(f32(d1*d4)-f32(d3*d2))
    if vc<=0 and d1>=0 and d3<=0:
        den=f32(d1-d3); v=f32(d1/den) if den>0 else 0.
        return f32add(a,f32mul(f32sub(b,a),v)),(f32(1-v),v,0.)
    cp=tuple(f32(x/scale) for x in f32sub(p,c)); d5=f32dot(ab,cp); d6=f32dot(ac,cp)
    if d6>=0 and d5<=d6: return c,(0.,0.,1.)
    vb=f32(f32(d5*d2)-f32(d1*d6))
    if vb<=0 and d2>=0 and d6<=0:
        den=f32(d2-d6); w=f32(d2/den) if den>0 else 0.
        return f32add(a,f32mul(f32sub(c,a),w)),(f32(1-w),0.,w)
    va=f32(f32(d3*d6)-f32(d5*d4))
    if va<=0 and f32(d4-d3)>=0 and f32(d5-d6)>=0:
        den=f32(f32(d4-d3)+f32(d5-d6)); w=f32(f32(d4-d3)/den) if den>0 else 0.
        return f32add(b,f32mul(f32sub(c,b),w)),(0.,f32(1-w),w)
    den=f32(f32(va+vb)+vc)
    if den>0:
        v=f32(vb/den); w=f32(vc/den)
        return f32add(a,f32add(f32mul(f32sub(b,a),v),f32mul(f32sub(c,a),w))),(f32(1-v-w),v,w)
    # Degenerate fallback omitted from emulator; these captured cells are nondegenerate.
    raise ValueError("f32 degenerate triangle fallback reached")

def source_like_ca(q, thickness):
    p0=f32v(q["point0"]); p1=f32v(q["point1"])
    tri0=[f32v(x) for x in q["tri0"]]; tri1=[f32v(x) for x in q["tri1"]]
    scale=max(maximum_abs(p0),maximum_abs(p1),*(maximum_abs(x) for x in tri0),*(maximum_abs(x) for x in tri1))
    eff=f32(f32(thickness)+f32(f32(8*EPS32)*f32(max(thickness,scale,1.e-12))))
    speed=f32(f32len(f32sub(p1,p0))+max(f32len(f32sub(x,y)) for x,y in zip(tri1,tri0)))
    t=f32(0.0); trace=[]; outcome=None
    for i in range(16):
        p=f32mix_source_order(p0,p1,t)
        tri=[f32mix_source_order(x,y,t) for x,y in zip(tri0,tri1)]
        cp,bary=closest_triangle_f32(p,*tri)
        delta=f32sub(p,cp); dist=f32len(delta); gap=f32(dist-eff)
        trace.append({"i":i,"time":t,"time_bits":bits32(t),"distance":dist,"distance_bits":bits32(dist),
                      "gap":gap,"gap_bits":bits32(gap),"bary":bary})
        if dist<=eff:
            outcome={"hit":True,"certified":True,"toi":t,"separation":dist,"termination":"distance_band"}
            break
        # This emulator records the advancement recurrence only; SAT proof branches
        # are not asserted. If they are needed on a sample, recurrence replay is inconclusive.
        if not speed>1.e-12:
            outcome={"hit":False,"certified":True,"toi":t,"separation":dist,"termination":"stationary"}; break
        step=f32(max(f32((dist-eff)/speed),1.e-6))
        t=f32(t+step)
        if t>f32(1+8*EPS32):
            outcome={"hit":False,"certified":True,"toi":trace[-1]["time"],"separation":dist,"termination":"past_end"}; break
        t=min(t,1.0)
    if outcome is None:
        z=trace[-1]; outcome={"hit":False,"certified":False,"toi":z["time"],"separation":z["distance"],"termination":"budget_exhaustion"}
    return {"effective_thickness":eff,"effective_thickness_bits":bits32(eff),
            "speed_bound":speed,"speed_bound_bits":bits32(speed),"trace":trace,"outcome":outcome,
            "limitations":"float32 source-order CPU emulation; no Metal bitwise claim; SAT early-separation branches omitted"}

def source_like_sample(q, thickness, t):
    p0=f32v(q["point0"]); p1=f32v(q["point1"])
    tri0=[f32v(x) for x in q["tri0"]]; tri1=[f32v(x) for x in q["tri1"]]
    scale=max(maximum_abs(p0),maximum_abs(p1),*(maximum_abs(x) for x in tri0),*(maximum_abs(x) for x in tri1))
    eff=f32(f32(thickness)+f32(f32(8*EPS32)*f32(max(thickness,scale,1.e-12))))
    p=f32mix_source_order(p0,p1,t)
    tri=[f32mix_source_order(x,y,t) for x,y in zip(tri0,tri1)]
    cp,bary=closest_triangle_f32(p,*tri)
    dist=f32len(f32sub(p,cp))
    return {"time":t,"distance":dist,"distance_bits":bits32(dist),"effective_thickness":eff,
            "effective_thickness_bits":bits32(eff),"gap":f32(dist-eff),"gap_bits":bits32(f32(dist-eff)),"barycentric":bary}

def analyze(c):
    q=build_query(c)
    thickness=c["params"][0]
    eff,scale,roundoff=effective_threshold(q,thickness)
    d0,b0,f0=geom_at(q,0.0); d1,b1,f1=geom_at(q,1.0)
    # Continuity and opposite strict endpoint signs prove at least one threshold crossing
    # for the affine endpoint interpolation and the independently reconstructed triangle.
    bracket=(d0>eff and d1<eff) or (d0<eff and d1>eff)
    d20,bary20,feature0=exact_endpoint_distance2(q["point0"],q["tri0"])
    d21,bary21,feature1=exact_endpoint_distance2(q["point1"],q["tri1"])
    band2=Fraction.from_float(float(eff))**2
    exact_bracket=(d20>band2 and d21<band2) or (d20<band2 and d21>band2)
    lo,hi=0.,1.
    flo=d0-eff
    for _ in range(80):
        mid=(lo+hi)*.5
        dm=geom_at(q,mid)[0]
        if (dm-eff)*flo>0: lo=mid
        else: hi=mid
    troot=(lo+hi)*.5
    dr,br,fr=geom_at(q,troot)
    tc=c["result"][2]; dc,brec,frec=geom_at(q,tc)
    native=c["result"]
    # exact rational endpoint comparisons for the CPU-reconstructed barycentric endpoints.
    # Distances themselves are evaluated in binary64; endpoint margins are reported explicitly.
    return {
        "run":c["run"],"capture_json_path":c["json_path"],"payload_path":c["bin_path"],"capture_json_sha256":c["json_sha256"],"payload_sha256":c["payload_sha256"],
        "record_sha256":c["record_sha256"],"record_raw_hex":c["record_raw_hex"],
        "capture_identity":{"root":c["capture"]["root"],"control_step":c["capture"]["control_step"],
            "physics_substep":c["capture"]["physics_substep"],"microtick":c["capture"]["microtick"],
            "iteration":c["capture"]["iteration"],"trial":c["capture"]["trial"],
            "selector":c["capture"]["mortar_complement_audit_selector"]},
        "decoded":{"attempt_words":c["words"][0:4],"candidate_identity":c["words"][4:8],
            "cells":c["words"][8:12],"feature":c["words"][12:16],
            "query_kind":"right_subtriangle_vertex_vs_left_subtriangle",
            "left_cell":q["left_cell"],"right_cell":q["right_cell"],"right_vertex":q["vertex_index"],
            "native_hit":bool(native[0]),"native_certified":bool(native[1]),
            "native_toi":native[2],"native_toi_bits":bits32(native[2]),
            "native_separation_m":native[3],"native_separation_bits":bits32(native[3]),
            "thickness_m":thickness,"thickness_bits":bits32(thickness)},
        "independent_geometry":{
            "endpoint_reconstruction":"For query kinds 3/4, the captured start/finish arrays are already the three left/right mortar-subtriangle CCD vertices (makeMortarComplementAudit is passed subtriangleEndpoints=true). They are promoted directly from stored binary32 words; bary arrays are retained as parent-map metadata and are not applied a second time.",
            "threshold_model":"Production roundoff formula 8*epsilon*max(thickness,query endpoint coordinate scale,1e-12), rounded to binary32 scalar and promoted.",
            "coordinate_scale_m":scale,"distance_roundoff_m":roundoff,"effective_thickness_m":eff,
            "start_distance_m":d0,"start_distance_minus_band_m":d0-eff,"start_closest_feature":f0,"start_barycentric":b0,
            "finish_distance_m":d1,"finish_distance_minus_band_m":d1-eff,"finish_closest_feature":f1,"finish_barycentric":b1,
            "threshold_crossing_bracketed_by_endpoint_signs":bool(bracket),
            "exact_binary_rational_endpoint_sign_test":{"start_distance_squared_minus_band_squared":fraction_record(d20-band2),"finish_distance_squared_minus_band_squared":fraction_record(d21-band2),"start_feature":feature0,"start_barycentric":[str(x) for x in bary20],"finish_feature":feature1,"finish_barycentric":[str(x) for x in bary21],"strict_opposite_signs":bool(exact_bracket),"scope":"exact comparison for the captured binary32 CCD endpoints and binary32 effective thickness; linear interpolation of point/triangle endpoints is continuous, so strict opposite signs establish at least one band crossing for that captured path"},
            "binary64_bisection_t":troot,"distance_at_bisection_m":dr,"bisection_barycentric":br,"bisection_closest_feature":fr,
            "distance_at_native_reported_t_m":dc,"distance_minus_effective_band_at_native_t_m":dc-eff,
            "native_t_barycentric":brec,"native_t_closest_feature":frec,
            "interpretation":"The opposite endpoint signs establish a finite-thickness crossing for the independently reconstructed affine subtriangle path. This supports a real-contact interpretation of the uncertified query; it is not a formal interval proof of production Metal rounding or a whole-candidate CCD certificate."},
        "source_like_fp32_sample_at_native_t":source_like_sample(q,thickness,native[2]),
        "source_like_fp32_ca":source_like_ca(q,thickness)
    }

def sha_path(p): return sha(pathlib.Path(p).read_bytes())
def run_self_tests():
    cp,bary,feature=closest_point_triangle((.2,.2,1),(0,0,0),(1,0,0),(0,1,0))
    assert abs(cp[2])<1e-15 and abs(sum(bary)-1)<1e-12 and feature=="face"
    assert f32(1.0)==1.0 and bits32(1.0)=="3f800000"
    good=bytes(384); desc={"dtype":"record:NMMortarComplementAuditGPU","count":1,"element_bytes":384,"offset":0,"byte_count":384,"sha256":sha(good)}
    assert validate_record(good,desc)==good
    failures=0
    for bad_payload,bad_desc in [
        (good[:-1],desc),
        (good,{**desc,"offset":1}),
        (good,{**desc,"dtype":"wrong"}),
        (good,{**desc,"sha256":"0"*64}),
        (good,{**desc,"count":2}),
        (good,{**desc,"element_bytes":380}),
    ]:
        try: validate_record(bad_payload,bad_desc)
        except (ValueError,KeyError): failures+=1
    assert failures==6
    exact,_,_=exact_endpoint_distance2((Fraction(1),Fraction(1),Fraction(1)),[(Fraction(0),Fraction(0),Fraction(0)),(Fraction(1),Fraction(0),Fraction(0)),(Fraction(0),Fraction(1),Fraction(0))])
    assert exact==Fraction(3,2)
    return {"passed":11,"cases":["closest-point face reference","barycentric closure","binary32 word roundtrip","384-byte record accepted","wrong payload length rejected","offset range rejected","dtype rejected","record digest rejected","count rejected","element width rejected","exact rational endpoint distance"],
            "scope":"synthetic CPU parser/geometry contract only; no production GPU evidence"}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=pathlib.Path)
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()
    if args.self_test:
        print(json.dumps(run_self_tests(),sort_keys=True,allow_nan=False)); return
    cases=[analyze(parse_case(x)) for x in ("605","610")]
    report={
      "schema":"numi.cardboard.mortar_uncertified_query_cpu_replay.v1",
      "source":{"snapshot":str(SRC),"frozen_source_sha256":sha_path(SRC/"frozen-source.json"),
        "contact_metalinc_sha256":sha_path(SRC/"matter/src/metal/contact.metalinc"),
        "common_metalinc_sha256":sha_path(SRC/"matter/src/metal/common.metalinc"),
        "build_record_sha256":sha_path(BUILDDIR/"source-build.json"),
        "checker_sha256":sha_path(BUILDDIR/"matter/numi-matter-cardboard-full-contact-armijo-check"),
        "NumiMatter_metallib_sha256":sha_path(BUILDDIR/"matter/shaders/NumiMatter.metallib")},
      "method":{"cpu_only":True,"solver_inputs_modified":False,"gpu_dispatched":False,
        "source_anchors":{"capture_layout":"contact.metalinc:2159-2201","mortar_query_direction":"contact.metalinc:2710-2733","vt_ccd_iteration":"contact.metalinc:3648-3777","vt_wrapper_budget":"contact.metalinc:3779-3801","roundoff_threshold":"common.metalinc:80-86"},
        "geometry":"Independent binary64 closest-point-on-triangle reference over captured binary32 parent endpoints and barycentric maps.",
        "ca_shadow":"Source-like float32 recurrence shadow used to inspect near-threshold plateaus. It omits SAT early-separation branches and does not claim Metal bitwise parity or certification.",
        "claim_boundary":"A threshold sign bracket indicates a contact on this reconstructed path; it never turns the production uncertified result into a certified clear/miss."},
      "self_tests":run_self_tests(),"cases":cases
    }
    encoded=json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if args.out:
        args.out.write_text(encoded)
    else: print(encoded,end="")
if __name__=="__main__": main()
