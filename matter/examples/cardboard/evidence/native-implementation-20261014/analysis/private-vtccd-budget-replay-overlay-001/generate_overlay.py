#!/usr/bin/env python3
"""Deterministically derive a budget-trace specialization from frozen source122."""
from __future__ import annotations
import hashlib
import pathlib
import sys

BASE = pathlib.Path('/Users/home/cardboard-implementation-20261014/snapshots/source-native-tet858-vtccd-traces-122/matter/src/metal/contact.metalinc')
EXPECTED_BASE_SHA256 = 'ea5a9caf6f47588258aeeb5e8be57168826acfb94a566ecbfea7e647ad50fb5c'
OUT = pathlib.Path(__file__).parent / 'overlay/matter/src/metal/contact.metalinc'

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def extract_function(text: str) -> tuple[int, int, str]:
    marker = 'inline bool vertexTriangleCCD(\n    const float3 vertex0,'
    start = text.rfind(marker)
    if start < 0:
        raise RuntimeError('source122 production VTCCD definition marker missing')
    brace = text.find('{', start)
    depth = 0
    end = -1
    for i in range(brace, len(text)):
        if text[i] == '{': depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0: raise RuntimeError('unbalanced source helper braces')
    return start, end, text[start:end]

def transform(source: str) -> str:
    start, end, original = extract_function(source)
    fn = original
    fn = fn.replace('inline bool vertexTriangleCCD(\n',
        'template <uint IterationBudget, bool CaptureIterationTrace>\n'
        'inline bool vertexTriangleCCDBudgetCore(\n', 1)
    needle = '    const float thickness,\n    thread float& toi,'
    replacement = ('    const float thickness,\n'
        '    device float4* iterationTrace,\n'
        '    const uint traceBase,\n'
        '    const uint traceCapacity,\n'
        '    thread uint& iterationsExecuted,\n'
        '    thread float& toi,')
    assert fn.count(needle) == 1
    fn = fn.replace(needle, replacement, 1)
    fn = fn.replace(') {\n    certified = false;\n',
        ') {\n    iterationsExecuted = 0u;\n    certified = false;\n', 1)
    fn = fn.replace('for (uint iteration = 0u; iteration < 16u; ++iteration) {',
        'for (uint iteration = 0u; iteration < IterationBudget; ++iteration) {', 1)
    fn = fn.replace('        const float3 vertexPoint = mix(vertex0, vertex1, time);',
        '        iterationsExecuted = iteration + 1u;\n'
        '        const float sampleTime = time;\n'
        '        const float3 vertexPoint = mix(vertex0, vertex1, time);', 1)
    hit_old = '            certified = true;\n            return finite3(normal);'
    hit_new = ('            certified = true;\n'
        '            const bool hit = finite3(normal);\n'
        '            if (CaptureIterationTrace && iteration < traceCapacity) {\n'
        '                const uint flags = (certified ? 1u : 0u) | (hit ? 2u : 0u);\n'
        '                iterationTrace[traceBase + iteration] = float4(\n'
        '                    sampleTime, distance - effectiveThickness, distance, float(flags));\n'
        '            }\n'
        '            return hit;')
    assert fn.count(hit_old) == 1
    fn = fn.replace(hit_old, hit_new, 1)
    # Each other certified return is a distinct early miss. Record its queried geometry.
    miss_old = '            certified = true;\n            return false;'
    miss_new = ('            certified = true;\n'
        '            if (CaptureIterationTrace && iteration < traceCapacity) {\n'
        '                iterationTrace[traceBase + iteration] = float4(\n'
        '                    sampleTime, distance - effectiveThickness, distance, 1.0f);\n'
        '            }\n'
        '            return false;')
    assert fn.count(miss_old) == 3, fn.count(miss_old)
    fn = fn.replace(miss_old, miss_new)
    nested_axis_miss = (
        '                    certified = true;\n'
        '                    return false;\n'
        '                }\n'
        '            }\n'
        '        }\n'
        '        if (!(speedBound > kGeometryFloor)) {')
    nested_axis_trace = (
        '                    certified = true;\n'
        '                    if (CaptureIterationTrace && iteration < traceCapacity) {\n'
        '                        iterationTrace[traceBase + iteration] = float4(\n'
        '                            sampleTime, distance - effectiveThickness, distance, 1.0f);\n'
        '                    }\n'
        '                    return false;\n'
        '                }\n'
        '            }\n'
        '        }\n'
        '        if (!(speedBound > kGeometryFloor)) {')
    assert fn.count(nested_axis_miss) == 1
    fn = fn.replace(nested_axis_miss, nested_axis_trace, 1)
    tail = '        time = min(time, 1.0f);\n'
    tail_new = ('        time = min(time, 1.0f);\n'
        '        if (CaptureIterationTrace && iteration < traceCapacity) {\n'
        '            iterationTrace[traceBase + iteration] = float4(\n'
        '                sampleTime, distance - effectiveThickness, distance, 0.0f);\n'
        '        }\n')
    assert fn.count(tail) == 1
    fn = fn.replace(tail, tail_new, 1)

    wrapper = '''inline bool vertexTriangleCCD(
    const float3 vertex0,
    const float3 vertex1,
    const float3 a0,
    const float3 a1,
    const float3 b0,
    const float3 b1,
    const float3 c0,
    const float3 c1,
    const float thickness,
    thread float& toi,
    thread float3& normal,
    thread float3& barycentric,
    thread float3& point,
    thread float& separation,
    thread bool& certified
) {
    uint iterationsExecuted = 0u;
    return vertexTriangleCCDBudgetCore<16u, false>(
        vertex0, vertex1, a0, a1, b0, b1, c0, c1, thickness,
        nullptr, 0u, 0u, iterationsExecuted,
        toi, normal, barycentric, point, separation, certified);
}

'''
    # Put the original-signature wrapper and native replay kernel immediately after the core.
    replay = '''template <uint IterationBudget>
inline void runVertexTriangleCCDBudgetReplay(
    device const NMDeformableVTCCDAuditGPU* captured,
    device const uint* budgets,
    device float4* summaries,
    device float4* trace,
    const uint budgetIndex
) {
    const NMDeformableVTCCDAuditGPU input = captured[0];
    const uint packedFeature = input.attempt.w;
    const uint direction = packedFeature / 3u;
    const uint vertexIndex = packedFeature % 3u;
    const bool valid = input.attempt.x == 1u && input.attempt.y == 0u &&
        input.attempt.z == 173u && packedFeature < 6u && direction < 2u &&
        input.parameters.x > 0.0f && isfinite(input.parameters.x) &&
        budgets[budgetIndex] == IterationBudget;
    if (!valid) {
        summaries[2u * budgetIndex] = float4(0.0f, 0.0f, NAN, NAN);
        summaries[2u * budgetIndex + 1u] = float4(0.0f, float(IterationBudget),
            float(packedFeature), 0.0f);
        return;
    }
    const uint vertexBase = direction == 0u ? 0u : 3u;
    const uint triangleBase = direction == 0u ? 3u : 0u;
    float toi = 0.0f, separation = INFINITY;
    float3 normal = float3(0.0f), barycentric = float3(0.0f), point = float3(0.0f);
    bool certified = false;
    uint iterationsExecuted = 0u;
    const bool hit = vertexTriangleCCDBudgetCore<IterationBudget, true>(
        input.start[vertexBase + vertexIndex].xyz,
        input.finish[vertexBase + vertexIndex].xyz,
        input.start[triangleBase].xyz, input.finish[triangleBase].xyz,
        input.start[triangleBase + 1u].xyz, input.finish[triangleBase + 1u].xyz,
        input.start[triangleBase + 2u].xyz, input.finish[triangleBase + 2u].xyz,
        input.parameters.x, trace, budgetIndex * 128u, 128u,
        iterationsExecuted, toi, normal, barycentric, point, separation, certified);
    summaries[2u * budgetIndex] = float4(hit ? 1.0f : 0.0f,
        certified ? 1.0f : 0.0f, toi, separation);
    summaries[2u * budgetIndex + 1u] = float4(float(iterationsExecuted),
        float(IterationBudget), float(packedFeature), 1.0f);
}

// Replays only the exact captured VT input and exports every conservative
// advancement query. Trace lanes are (time, gap-to-effective-thickness,
// separation, flags), where flags bit0=certified and bit1=hit.
kernel void nm_contact_vtccd_iteration_budget_replay(
    device const NMDeformableVTCCDAuditGPU* captured [[buffer(0)]],
    device const uint* budgets [[buffer(1)]],
    device float4* summaries [[buffer(2)]],
    device float4* trace [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= 4u) return;
    switch (budgets[index]) {
        case 16u: runVertexTriangleCCDBudgetReplay<16u>(captured, budgets, summaries, trace, index); break;
        case 32u: runVertexTriangleCCDBudgetReplay<32u>(captured, budgets, summaries, trace, index); break;
        case 64u: runVertexTriangleCCDBudgetReplay<64u>(captured, budgets, summaries, trace, index); break;
        case 128u: runVertexTriangleCCDBudgetReplay<128u>(captured, budgets, summaries, trace, index); break;
        default:
            summaries[2u * index] = float4(0.0f, 0.0f, NAN, NAN);
            summaries[2u * index + 1u] = float4(0.0f, float(budgets[index]), 0.0f, 0.0f);
            break;
    }
}

'''
    generated = source[:start] + fn + '\n\n' + wrapper + replay + source[end:]
    return generated

def main() -> int:
    base = BASE.read_bytes()
    actual = sha(base)
    if actual != EXPECTED_BASE_SHA256:
        raise SystemExit(f'base SHA mismatch: expected {EXPECTED_BASE_SHA256}, got {actual}')
    source = base.decode('utf-8')
    generated = transform(source).encode('utf-8')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(generated)
    print(f'base_sha256={actual}')
    print(f'candidate_sha256={sha(generated)}')
    print(f'candidate_bytes={len(generated)}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
