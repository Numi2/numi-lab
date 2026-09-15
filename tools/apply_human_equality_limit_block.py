"""Integrate a local equality/limit Schur block into the existing Human sweep."""
from __future__ import annotations
import hashlib
from pathlib import Path

BASE_BLOB = "d9a10cdd2bc7c1c0b5527489226a755ce00245d8"
MARKER = "Strong equality/limit pairs use the same mass-response Schur block."


def once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"nonunique Human block anchor: {old[:100]!r}")
    return text.replace(old, new, 1)


def transform(source: str) -> str:
    raw = source.encode()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if blob != BASE_BLOB:
        raise ValueError(f"standing owner changed: {blob}; expected {BASE_BLOB}")
    source = once(source, "    uint limitDofs[MR_NUMI_HUMAN_STAND_MAX_DOFS];",
                  "    uint limitDofs[MR_NUMI_HUMAN_STAND_MAX_DOFS];\n    uint limitEqualityIndices[MR_NUMI_HUMAN_STAND_MAX_DOFS];")
    source = once(source, "        ++limitCount;", '''        // Strong equality/limit pairs use the same mass-response Schur block.
        // Select once per step. No new factorization, compliance, or global
        // solver is introduced; other rows remain in the existing sweep.
        limitEqualityIndices[limitCount] = MR_INVALID_INDEX;
        float bestCorrelation = 0.95f;
        for (uint ei = 0u; ei < dispatch.jointEqualityCount; ++ei) {
            device const MRNumiHumanJointEqualityGPU& eq = jointEqualities[ei];
            if (eq.indices.y != dof && eq.indices.w != dof) continue;
            float target = 0.0f, derivative = 0.0f, error = 0.0f;
            if (!evaluateJointEquality(eq, qState, qBase, nq, nv,
                                       target, derivative, error)) {
                fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED, ei);
                return;
            }
            device const float* er = responseScratch + responseBase +
                (3u * dispatch.supportContactCount + ei) * nv;
            float a = er[eq.indices.y];
            float b = response[eq.indices.y];
            if (eq.indices.w != MR_INVALID_INDEX) {
                a -= derivative * er[eq.indices.w];
                b -= derivative * response[eq.indices.w];
            }
            const float c = er[dof], d = response[dof];
            if (!(a > 0.0f) || !(d > 0.0f)) continue;
            const float correlation = (b / a) * (c / d);
            const float projected = fma(-c / a, b, d);
            if (isfinite(correlation) && isfinite(projected) &&
                correlation > bestCorrelation && projected > 1.0e-7f * d) {
                bestCorrelation = correlation;
                limitEqualityIndices[limitCount] = ei;
            }
        }
        ++limitCount;''')
    source = once(source, '''            const float nextImpulse = mrNumiHumanProjectIntervalImpulse(
                limitAccumulatedImpulses[limit], candidateV[dof],
                lowerVelocity, upperVelocity, effectiveMass);
            const float impulse = nextImpulse - limitAccumulatedImpulses[limit];''', '''            float nextImpulse = mrNumiHumanProjectIntervalImpulse(
                limitAccumulatedImpulses[limit], candidateV[dof],
                lowerVelocity, upperVelocity, effectiveMass);
            float pairedEqualityDelta = 0.0f;
            const uint pairedEquality = limitEqualityIndices[limit];
            if (pairedEquality != MR_INVALID_INDEX) {
                device const MRNumiHumanJointEqualityGPU& eq =
                    jointEqualities[pairedEquality];
                float target = 0.0f, derivative = 0.0f, error = 0.0f;
                if (!evaluateJointEquality(eq, qState, qBase, nq, nv,
                                           target, derivative, error)) {
                    fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED,
                         pairedEquality);
                    return;
                }
                device const float* er = responseScratch + responseBase +
                    (3u * dispatch.supportContactCount + pairedEquality) * nv;
                float a = er[eq.indices.y];
                float b = response[eq.indices.y];
                float velocity = candidateV[eq.indices.y];
                if (eq.indices.w != MR_INVALID_INDEX) {
                    a -= derivative * er[eq.indices.w];
                    b -= derivative * response[eq.indices.w];
                    velocity -= derivative * candidateV[eq.indices.w];
                }
                const auto block = mrNumiHumanProjectEqualityLimitBlock(
                    limitAccumulatedImpulses[limit], velocity, candidateV[dof],
                    clamp(-0.2f * error / timestep, -4.0f, 4.0f),
                    lowerVelocity, upperVelocity, a, b, er[dof], response[dof]);
                if (!block.valid || !isfinite(block.equalityDelta) ||
                    !isfinite(block.limitImpulse)) {
                    fail(status, MR_NUMI_HUMAN_STAND_FACTORIZATION_FAILED, dof);
                    return;
                }
                nextImpulse = block.limitImpulse;
                pairedEqualityDelta = block.equalityDelta;
                equalityLambdas[pairedEquality] += pairedEqualityDelta;
            }
            const float impulse = nextImpulse - limitAccumulatedImpulses[limit];''')
    source = once(source, '''            limitAccumulatedImpulses[limit] = nextImpulse;
            for (uint index = 0u; index < nv; ++index) {
                candidateV[index] += impulse * response[index];
            }''', '''            limitAccumulatedImpulses[limit] = nextImpulse;
            for (uint index = 0u; index < nv; ++index) {
                float delta = impulse * response[index];
                if (pairedEquality != MR_INVALID_INDEX) {
                    device const float* er = responseScratch + responseBase +
                        (3u * dispatch.supportContactCount + pairedEquality) * nv;
                    delta = fma(pairedEqualityDelta, er[index], delta);
                }
                candidateV[index] += delta;
            }''')
    # Equality impulses now also change in the limit block. Sample the final
    # coupled state, not the earlier equality-only intermediate state.
    start = source.index("        if (coupledSweep + 1u == coupledSweepCount) {\n        for (uint equalityIndex = 0u;")
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    block = source[start:end]
    source = source[:start] + source[end:]
    block = block.replace("if (coupledSweep + 1u == coupledSweepCount)",
                          "if ((dispatch.flags & MR_NUMI_HUMAN_STAND_HAS_JOINT_EQUALITIES) != 0u)", 1)
    source = once(source, "    maximumAcceleration = 0.0f;",
                  "    // Final equality evidence includes paired limit corrections.\n" + block + "\n\n    maximumAcceleration = 0.0f;")
    return source


def main() -> int:
    path = Path(__file__).resolve().parents[1] / "src/metal/NumiHumanStand.metal"
    source = path.read_text()
    if MARKER in source:
        if "mrNumiHumanProjectEqualityLimitBlock(" not in source:
            raise ValueError("paired block marker exists without its native call")
        print("Human equality/limit block already integrated")
        return 0
    revised = transform(source)
    temporary = path.with_suffix(".metal.tmp")
    temporary.write_text(revised)
    temporary.replace(path)
    print("Integrated local equality/limit Schur blocks with final-state diagnostics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
