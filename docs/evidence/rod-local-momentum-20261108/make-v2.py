from pathlib import Path
s=Path("/Users/numi/rod-vector-20261108/guarded.metal").read_text()
needle="""            for (uint edge = 0u; edge < dispatch.edgeCount && valid; ++edge) {
                const float lengthX = axial(positions[edge + 1u] - positions[edge]);"""
insert="""            // Research-only local mechanical certificate. This checks
            // stretch momentum; bend/twist and coupled forces remain open.
            for (uint i = 0u; i < dispatch.nodeCount && valid; ++i) {
                if (axialPinIndex[i] != 0xffffffffu) continue;
                float3 gradient = float3(0.0f);
                if (i > 0u) {
                    const uint edge = i - 1u;
                    const float3 d = positions[i] - positions[i - 1u];
                    const float l = length(d);
                    valid = all(isfinite(d)) && isfinite(l) && l > kGeometryFloor;
                    if (valid) gradient += (stretchStiffness[edge] / restLengths[edge]) *
                        (l - restLengths[edge]) * (d / l);
                }
                if (i < end && valid) {
                    const uint edge = i;
                    const float3 d = positions[i] - positions[i + 1u];
                    const float l = length(d);
                    valid = all(isfinite(d)) && isfinite(l) && l > kGeometryFloor;
                    if (valid) gradient += (stretchStiffness[edge] / restLengths[edge]) *
                        (l - restLengths[edge]) * (d / l);
                }
                if (valid) {
                    const float3 residual = (positions[i] - originalPositions[i]) / h -
                        velocities[i] + h * inverseMasses[i] * gradient;
                    const float limit = dispatch.dampingDerivativeTolerance.w / h;
                    if (!all(isfinite(residual)) || !isfinite(limit) || length(residual) > limit) {
                        recordFailure(failure, MR_ROD_GPU_DID_NOT_CONVERGE);
                        valid = false;
                    }
                }
            }
"""+needle
assert s.count(needle)==1
s=s.replace(needle,insert)
# Do not replace a specific nonconvergence with a generic geometry failure.
old="""            } else {
                recordFailure(failure, MR_ROD_GPU_DEGENERATE_GEOMETRY);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint iteration"""
new="""            } else if (atomic_load_explicit(&failure, memory_order_relaxed) == MR_ROD_GPU_SUCCESS) {
                recordFailure(failure, MR_ROD_GPU_DEGENERATE_GEOMETRY);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint iteration"""
assert s.count(old)==1
Path("residual.metal").write_text(s.replace(old,new))
