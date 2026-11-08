from pathlib import Path
s=Path("/Users/numi/numi-lab-tracker-step-rollback-20261108/docs/evidence/rod-axis-rotation-20261108/candidate.metal").read_text()
def rep(old,new):
 global s
 assert s.count(old)==1, (old[:80],s.count(old))
 s=s.replace(old,new)
rep("    threadgroup uint axialPinIndex[MR_ROD_GPU_MAX_NODES];", "    threadgroup uint axialPinIndex[MR_ROD_GPU_MAX_NODES];\n    threadgroup float axialCoordinate[MR_ROD_GPU_MAX_NODES];")
rep("""        dispatch.flags == 0u && dispatch.dampingDerivativeTolerance.x == 0.0f &&
        dispatch.dampingDerivativeTolerance.y == 0.0f &&
        dispatch.gravityAndTimestep.y == 0.0f &&
        dispatch.gravityAndTimestep.z == 0.0f &&""","""        dispatch.flags == 0u && dispatch.dampingDerivativeTolerance.x == 0.0f &&
        dispatch.dampingDerivativeTolerance.y == 0.0f &&""")
rep("""        const uint end = dispatch.nodeCount - 1u;
        for (uint i = 0u; i < dispatch.nodeCount; ++i) axialPinIndex[i] = 0xffffffffu;
        bool eligible = true;""","""        const uint end = dispatch.nodeCount - 1u;
        const float3 origin = originalPositions[0];
        const float3 delta = originalPositions[end] - origin;
        const float extent = length(delta);
        const float3 axis = extent > kGeometryFloor && isfinite(extent) ? delta / extent : float3(0.0f);
        // Admit only deviations at the scale of FP32 rotation arithmetic.
        // This is research-only and does not prove the neglected bending force.
        const float relativeGuard = 32.0f * 1.1920928955078125e-7f;
        const auto axial = [&](float3 a) { return dot(a, axis); };
        const auto aligned = [&](float3 a) {
            const float magnitude = length(a);
            return all(isfinite(a)) && isfinite(magnitude) &&
                length(a - axis * axial(a)) <= relativeGuard * max(magnitude, 1.0e-12f);
        };
        for (uint i = 0u; i < dispatch.nodeCount; ++i) axialPinIndex[i] = 0xffffffffu;
        bool eligible = extent > kGeometryFloor && isfinite(extent) &&
            all(isfinite(axis)) && aligned(dispatch.gravityAndTimestep.xyz);""")
rep("""                all(isfinite(anchor.velocity)) &&
                anchor.targetAndCompliance.y == originalPositions[0].y &&
                anchor.targetAndCompliance.z == originalPositions[0].z &&
                anchor.velocity.y == 0.0f && anchor.velocity.z == 0.0f;""","""                all(isfinite(anchor.velocity)) &&
                aligned(anchor.targetAndCompliance.xyz - origin) &&
                aligned(anchor.velocity.xyz);""")
rep("""            eligible = originalPositions[i].y == originalPositions[0].y &&
                originalPositions[i].z == originalPositions[0].z &&
                velocities[i].y == 0.0f && velocities[i].z == 0.0f &&
                inverseMasses[i] > 0.0f && isfinite(inverseMasses[i]);
            if (i > 0u) {
                eligible = eligible && originalPositions[i].x > originalPositions[i - 1u].x;
            }""","""            eligible = aligned(originalPositions[i] - origin) && aligned(velocities[i]) &&
                inverseMasses[i] > 0.0f && isfinite(inverseMasses[i]);
            if (i > 0u) {
                eligible = eligible && axial(originalPositions[i] - originalPositions[i - 1u]) > kGeometryFloor;
            }""")
rep("""            for (uint i = 0u; i < dispatch.nodeCount; ++i) {
                if (axialPinIndex[i] != 0xffffffffu) {
                    positions[i].x = attachments[base + axialPinIndex[i]].targetAndCompliance.x;
                }
            }""","""            for (uint i = 0u; i < dispatch.nodeCount; ++i) {
                axialCoordinate[i] = axial(positions[i] - origin);
                if (axialPinIndex[i] != 0xffffffffu) {
                    axialCoordinate[i] = axial(attachments[base + axialPinIndex[i]].targetAndCompliance.xyz - origin);
                }
            }""")
rep("""                float rhs = kl * (positions[i - 1u].x + restLengths[i - 1u] - positions[i].x) +
                    kr * (positions[i + 1u].x - restLengths[i] - positions[i].x);""","""                float rhs = kl * (axialCoordinate[i - 1u] + restLengths[i - 1u] - axialCoordinate[i]) +
                    kr * (axialCoordinate[i + 1u] - restLengths[i] - axialCoordinate[i]);""")
rep("""                    if (axialPinIndex[i] == 0xffffffffu) positions[i].x += iterationPositions[i].x;""","""                    if (axialPinIndex[i] == 0xffffffffu) axialCoordinate[i] += iterationPositions[i].x;""")
rep("""            for (uint edge = 0u; edge < dispatch.edgeCount && valid; ++edge) {
                const float lengthX = positions[edge + 1u].x - positions[edge].x;
                valid = isfinite(lengthX) && lengthX > kGeometryFloor;
                if (valid) recordPositiveMaximum(maximumErrorBits, abs(lengthX - restLengths[edge]));
            }""","""            for (uint i = 0u; i < dispatch.nodeCount; ++i) {
                if (axialPinIndex[i] == 0xffffffffu) {
                    positions[i] += axis * (axialCoordinate[i] - axial(positions[i] - origin));
                } else {
                    positions[i] = attachments[base + axialPinIndex[i]].targetAndCompliance.xyz;
                }
            }
            for (uint edge = 0u; edge < dispatch.edgeCount && valid; ++edge) {
                const float lengthX = axial(positions[edge + 1u] - positions[edge]);
                valid = isfinite(lengthX) && lengthX > kGeometryFloor;
                if (valid) recordPositiveMaximum(maximumErrorBits, abs(lengthX - restLengths[edge]));
            }""")
rep("""                        (positions[node].x - positions[node - 1u].x - restLengths[edge]);""","""                        (axial(positions[node] - positions[node - 1u]) - restLengths[edge]);""")
rep("""                        (positions[node + 1u].x - positions[node].x - restLengths[edge]);""","""                        (axial(positions[node + 1u] - positions[node]) - restLengths[edge]);""")
rep("""                targetImpulses[j] = float3(force * h, 0.0f, 0.0f) +""","""                targetImpulses[j] = axis * (force * h) +""")
Path("/Users/numi/rod-vector-20261108/candidate.metal").write_text(s)
