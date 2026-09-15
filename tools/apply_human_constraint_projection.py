"""Apply the reviewed constraint correction to the exact published native owner.

One-shot source migration, not a runtime adapter. The source blob identity is
checked before any write; a mismatching checkout is never heuristically patched.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import re

BASE_BLOB = "61fe7fb51e75e5cf7c19efc9d0e532735879356a"
HEADER = '#include "metalrobo/numi_human_constraint_projection.h"'


def once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected one exact source anchor: {old[:90]!r}")
    return text.replace(old, new, 1)


def transform(source: str) -> str:
    data = source.encode("utf-8")
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if blob != BASE_BLOB:
        raise ValueError(f"native source changed: {blob}; expected {BASE_BLOB}")
    source = once(source, '#include "metalrobo/numi_human_stand_gpu.h"',
                  '#include "metalrobo/numi_human_stand_gpu.h"\n' + HEADER)
    source = once(source, '''    // Source static support reactions are explicit contact-owner loads. They
    // are converted to generalized force through the same point Jacobians
    // used by the runtime contact solver; they are not root assistance.
''', '''    // Support belongs exclusively to the unilateral impulse solve below.
    // Static support is a retractable warm start, never an additional force.
''')
    source = once(source, '''        if ((dispatch.flags & MR_NUMI_HUMAN_STAND_ENABLE_CONTACT) != 0u) {
            for (uint contact = 0u;
                 contact < dispatch.supportContactCount;
                 ++contact) {
                const float supportForce =
                    contacts[contact].frictionSlopAndStabilization.w;
                if (supportForce <= 0.0f) continue;
                effort += supportForce * pointJacobianAxis(
                    pointJacobians, pointJacobianBase,
                    contacts[contact].pointQueryIndex, nv, dof,
                    dispatch.groundNormal.xyz
                );
            }
        }
''', '')
    source = once(source, '''                    matrix[3u * row + column] = value;
                }
            }
        }
        }
        for (uint iteration = 0u;''', '''                    matrix[3u * row + column] = value;
                }
            }
            // Every step starts with a new free velocity. Initialize the
            // matching total impulse and apply it exactly once; retaining an
            // unapplied previous-step lambda would corrupt complementarity.
            const float seed = mrNumiHumanSupportSeedImpulse(
                support.frictionSlopAndStabilization.w, timestep, gap,
                support.frictionSlopAndStabilization.y);
            if (!isfinite(seed)) {
                fail(status, MR_NUMI_HUMAN_STAND_NONFINITE_RESULT, contact);
                return;
            }
            lambdas[3u * contact + 0u] = seed;
            lambdas[3u * contact + 1u] = 0.0f;
            lambdas[3u * contact + 2u] = 0.0f;
            device const float* normalResponse = responseScratch +
                responseBase + (3u * contact) * nv;
            for (uint dof = 0u; dof < nv; ++dof) {
                candidateV[dof] += seed * normalResponse[dof];
            }
        }
        }
        for (uint iteration = 0u;''')
    source = once(source, '''                const float targetNormalVelocity = max(
                    0.0f,
                    -support.frictionSlopAndStabilization.z * min(gap, 0.0f) /
                        timestep
                );''', '''                const float targetNormalVelocity =
                    mrNumiHumanContactVelocityTarget(gap, timestep,
                        support.frictionSlopAndStabilization.z);''')
    source = once(source, '''        const float tolerance = 1.0e-7f;
        const bool lowerNear = position <= properties.limits.x + tolerance;
        const bool upperNear = position >= properties.limits.y - tolerance;
        const bool lowerActive = lowerNear &&
            (position < properties.limits.x || candidateV[dof] < 0.0f);
        const bool upperActive = upperNear &&
            (position > properties.limits.y || candidateV[dof] > 0.0f);
        if (!lowerActive && !upperActive) continue;
        if (lowerActive && upperActive) {
            fail(status, MR_NUMI_HUMAN_STAND_INVALID_MODEL, dof);
            return;
        }
''', '''        // Prepare every authored scalar interval. Contact/equality updates
        // can activate a row that was inactive in the first free velocity.
        if (!isfinite(position)) {
            fail(status, MR_NUMI_HUMAN_STAND_NONFINITE_INPUT, dof);
            return;
        }
''')
    source = once(source, '''            const bool lowerActive =
                position <= properties.limits.x + 1.0e-7f &&
                (position < properties.limits.x || candidateV[dof] < 0.0f);
            const bool upperActive =
                position >= properties.limits.y - 1.0e-7f &&
                (position > properties.limits.y || candidateV[dof] > 0.0f);
''', '')
    pattern = r"            const float velocity = candidateV\[dof\];\n.*?(?=            if \(!isfinite\(impulse\)\))"
    replacement = '''            const float lowerVelocity = mrNumiHumanLowerLimitVelocityTarget(
                position, properties.limits.x, timestep);
            const float upperVelocity = mrNumiHumanUpperLimitVelocityTarget(
                position, properties.limits.y, timestep);
            if (!isfinite(lowerVelocity) || !isfinite(upperVelocity) ||
                lowerVelocity > upperVelocity) {
                fail(status, MR_NUMI_HUMAN_STAND_NONFINITE_RESULT, dof);
                return;
            }
            const float nextImpulse = mrNumiHumanProjectIntervalImpulse(
                limitAccumulatedImpulses[limit], candidateV[dof],
                lowerVelocity, upperVelocity, effectiveMass);
            const float impulse = nextImpulse - limitAccumulatedImpulses[limit];
'''
    source, count = re.subn(pattern, lambda _: replacement, source, flags=re.S)
    if count != 1:
        raise ValueError("source limit solve anchor is not unique")
    source = once(source, '            limitAccumulatedImpulses[limit] += impulse;',
                  '            limitAccumulatedImpulses[limit] = nextImpulse;')
    # Keep pre/post-projection evidence on the SAME interval as the solver.
    # In particular, record a newly activated row and any projection-induced
    # violation instead of restricting the audit to the original active set.
    pattern = (r"                const bool lowerActive =\n.*?"
               r"(?=\n            }\n        }\n        for \(uint equalityIndex)")
    def diagnostic(match: re.Match[str]) -> str:
        stage = "Pre" if "maximumPreProjectionLimitResidual" in match[0] else "Post"
        metric = f"maximum{stage}ProjectionLimitResidual"
        return f'''                const float lowerVelocity = mrNumiHumanLowerLimitVelocityTarget(
                    position, properties.limits.x, timestep);
                const float upperVelocity = mrNumiHumanUpperLimitVelocityTarget(
                    position, properties.limits.y, timestep);
                {metric} = max({metric},
                    max(max(0.0f, lowerVelocity - candidateV[dof]),
                        max(0.0f, candidateV[dof] - upperVelocity)));'''
    source, count = re.subn(pattern, diagnostic, source, flags=re.S)
    if count != 2:
        raise ValueError("expected both pre/post projection interval audits")
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    path = args.root / "src/metal/NumiHumanStand.metal"
    source = path.read_text(encoding="utf-8")
    if HEADER in source:
        print("Human constraint projection migration already applied")
        return 0
    revised = transform(source)
    temporary = path.with_suffix(".metal.tmp")
    temporary.write_text(revised, encoding="utf-8")
    temporary.replace(path)
    print("Applied support impulse ownership and releasable source-limit solve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
