#!/usr/bin/env python3
"""Compare strict double clipping with exact dyadic clipping on FP32 maps."""
import hashlib, json, math, pathlib, struct
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parent
CANDIDATES = ROOT / 'candidates.log'
OUT = ROOT / 'comparison-report.json'
THRESHOLD = struct.unpack('<f', struct.pack('<f', 1.0e-10))[0]
THRESHOLD_Q = Fraction.from_float(THRESHOLD)

def fp32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]

def exact_fp32_hex(token):
    return Fraction.from_float(fp32(float.fromhex(token)))

def sub(a, b): return (a[0] - b[0], a[1] - b[1])
def cross(a, b): return a[0] * b[1] - a[1] * b[0]
def orient(a, b, c): return cross(sub(b, a), sub(c, a))

def clip_area(vertices_a, vertices_b, exact):
    # Vertices are already exact Fractions or doubles. No geometric epsilon.
    o = orient(vertices_b[0], vertices_b[1], vertices_b[2])
    if o == 0: return None
    sign = 1 if o > 0 else -1
    poly = list(vertices_a)
    for edge_i in range(3):
        e0, e1 = vertices_b[edge_i], vertices_b[(edge_i + 1) % 3]
        edge = sub(e1, e0)
        def side(p): return sign * cross(edge, sub(p, e0))
        output = []
        for i, p in enumerate(poly):
            q = poly[(i + 1) % len(poly)]
            sp, sq = side(p), side(q)
            p_inside, q_inside = sp >= 0, sq >= 0
            if p_inside: output.append(p)
            if p_inside != q_inside:
                denominator = sp - sq
                if denominator == 0: return None
                t = sp / denominator
                output.append((p[0] + t * (q[0] - p[0]),
                               p[1] + t * (q[1] - p[1])))
        poly = output
        if not poly: return Fraction(0) if exact else 0.0
    if len(poly) < 3: return Fraction(0) if exact else 0.0
    terms = [cross(poly[i], poly[(i + 1) % len(poly)])
             for i in range(len(poly))]
    twice = sum(terms, Fraction(0)) if exact else math.fsum(terms)
    return abs(twice) / 2

def parse_candidate(line):
    f = line.split(',')
    vals = [exact_fp32_hex(v) for v in f[5:]]
    a = [(vals[3*i], vals[3*i+1]) for i in range(3)]
    b = [(vals[9+3*i], vals[10+3*i]) for i in range(3)]
    return f[1], (int(f[2]), int(f[3])), a, b

def from_float_tri(t): return [(float(x), float(y)) for x, y in t]

def next_f32(bits): return struct.unpack('<f', struct.pack('<I', bits))[0]

# Independent analytic controls, all coordinates authored as binary32.
A = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)),
     (Fraction(0), Fraction(1))]
def qtri(items): return [(Fraction.from_float(fp32(x)), Fraction.from_float(fp32(y))) for x,y in items]
controls = {
    'exact_shared_edge_disjoint': (A, qtri([(1,0),(1,1),(0,1)]), 'below_floor'),
    'same_edge_same_side_overlap': (A, qtri([(0,0),(1,0),(.5,.5)]), 'above_floor'),
    'positive_overlap_no_shared_edge': (A, qtri([(.9998,0),(1,1),(0,1)]), 'above_floor'),
    'one_ulp_inward_positive_above_floor': (A, qtri([(next_f32(0x3f7fffff),0),(1,1),(0,1)]), 'above_floor'),
    'one_ulp_outward_gap': (A, qtri([(next_f32(0x3f800001),0),(1,1),(0,1)]), 'below_floor'),
}

def floor_class(area):
    return 'above_floor' if area is not None and area > THRESHOLD_Q else 'below_floor'

candidate_rows = []
for line in CANDIDATES.read_text().splitlines():
    if not line.startswith('UNPROVEN,'): continue
    side, ids, a_q, b_q = parse_candidate(line)
    exact_area = clip_area(a_q, b_q, True)
    double_area = clip_area(from_float_tri(a_q), from_float_tri(b_q), False)
    candidate_rows.append({
        'side': side, 'stable_ids': list(ids),
        'exact_area': str(exact_area) if exact_area is not None else 'undefined',
        'exact_area_float': float(exact_area) if exact_area is not None else None,
        'strict_double_area': double_area,
        'exact_floor_class': floor_class(exact_area),
        'double_floor_class': 'above_floor' if double_area is not None and double_area > THRESHOLD else 'below_floor',
        'class_agrees': floor_class(exact_area) == ('above_floor' if double_area is not None and double_area > THRESHOLD else 'below_floor'),
    })

control_rows = []
for name, (a, b, expected) in controls.items():
    exact_area = clip_area(a, b, True)
    double_area = clip_area(from_float_tri(a), from_float_tri(b), False)
    exact_class, double_class = floor_class(exact_area), ('above_floor' if double_area is not None and double_area > THRESHOLD else 'below_floor')
    control_rows.append({'name':name,'expected_floor_class':expected,'exact_area':str(exact_area),'strict_double_area':double_area,'exact_floor_class':exact_class,'double_floor_class':double_class,'pass':exact_class==expected and double_class==expected})

summary = {
    'schema':'numi.cardboard.exact-dyadic-vs-double-overlap.v1',
    'input':str(CANDIDATES), 'input_sha256':hashlib.sha256(CANDIDATES.read_bytes()).hexdigest(),
    'method':'Exact rational Sutherland-Hodgman uses Fraction representations of the exact IEEE-754 binary32 input coordinates. Comparison uses strict double clipping on the same promoted coordinates, with no epsilon. Both are compared against the existing threshold value rounded to binary32; the threshold is not relaxed.',
    'threshold_fp32':THRESHOLD, 'threshold_exact_rational':str(THRESHOLD_Q),
    'candidate_side_count':len(candidate_rows),
    'candidate_unique_stable_pairs':len({tuple(r['stable_ids']) for r in candidate_rows}),
    'candidate_exact_zero_area_sides':sum(r['exact_area']=='0' for r in candidate_rows),
    'candidate_exact_positive_area_sides':sum(r['exact_area'] not in ('0','undefined') for r in candidate_rows),
    'candidate_exact_above_floor_sides':sum(r['exact_floor_class']=='above_floor' for r in candidate_rows),
    'candidate_double_above_floor_sides':sum(r['double_floor_class']=='above_floor' for r in candidate_rows),
    'candidate_double_exact_floor_class_mismatches':sum(not r['class_agrees'] for r in candidate_rows),
    'controls':control_rows,
    'controls_all_pass':all(r['pass'] for r in control_rows),
    'candidate_results':candidate_rows,
}
OUT.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='candidate_results'},indent=2))
if not summary['controls_all_pass'] or summary['candidate_double_exact_floor_class_mismatches']:
    raise SystemExit(1)
