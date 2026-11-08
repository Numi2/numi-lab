from pathlib import Path
import gzip
here=Path(__file__).resolve().parent
old=here.parent/"mixed-failing-row-witness-20261108"
s=gzip.decompress((old/"baseline.metal.gz").read_bytes()).decode()
start=s.index("    float laneNormalResidual = 0.0f;", s.index("const float groupError = simd_max(max(finalNormal, finalTangent));"))
needle="""        const float limitU =
            cone.effectiveFrictionU * contact.impulses.x;"""
pos=s.index(needle,start)
instrument=r"""
        // Diagnostic only: a 3-contact island with at most one 3-node rod.
        // This post-solve capture never changes the solver or physical state.
        if (work.constraintCount == 3u && work.tileCount == 1u &&
            dispatch.rodNodeCount == 3u && dispatch.rodEdgeCount == 2u) {
            float m[9]; float ids[3];
            bool ok = true;
            const uint targetAxis = AXIS_VALUE;
            const float3 targetDirection = localRows[targetAxis].direction.xyz;
            const MRContactTileGPU sourceTile = tiles[tileBase + work.firstTile];
            for (uint slot = 0u; slot < 3u; ++slot) {
                const uint si = tileConstraintIndices[constraintBase + sourceTile.partialOffset + slot];
                ids[slot] = float(si);
                device const MRContactConstraintGPU& source = contacts[constraintBase + si];
                device const MREvaluatedConstraintIRRowGPU* sr = evaluatedRows + rowBase + 3u*si;
                for (uint axis = 0u; axis < 3u; ++axis) {
                    const float3 direction = sr[axis].direction.xyz;
                    float value = typedNormalCrossContactResponse(
                        localConstraint, si, axis, contact, source,
                        targetDirection, direction, bodies, pointJacobians,
                        pointJacobianBase, responseColumns, responseBase, dispatch.nv,
                        dispatch.articulationIndex, rodNodes, inverseRodMasses,
                        inverseRodTwistInertias, rodColliders, rodWitnesses,
                        constraintWitnessIndices, constraintBase);
                    if (typedRodConstraint(contact) && typedRodConstraint(source)) {
                        const uint ti = constraintWitnessIndices[constraintBase+localConstraint];
                        const uint sj = constraintWitnessIndices[constraintBase+si];
                        if (ti == MR_INVALID_INDEX || sj == MR_INVALID_INDEX) { ok=false; }
                        else {
                            const MRRodToolWitnessGPU tw = rodWitnesses[ti];
                            const MRRodToolWitnessGPU sw = rodWitnesses[sj];
                            const MRRodColliderGPU tc = rodColliders[tw.identity.z];
                            const MRRodColliderGPU sc = rodColliders[sw.identity.z];
                            if (tc.rodIndex == sc.rodIndex) {
                                float3 exact = float3(0.0f);
                                const bool valid = factorizedRodCrossResponse(
                                    dispatch, pass, environment, tc, tw, sc, sw, direction,
                                    rodNodes, rodFactorCaches, rodOperatorArena, exact);
                                ok = ok && valid;
                                value += dot(targetDirection,exact) - rodCrossContactResponse(
                                    tc,tw,sc,sw,rodNodes,inverseRodMasses,
                                    inverseRodTwistInertias,targetDirection,direction);
                            }
                        }
                    }
                    if (si == localConstraint && axis == targetAxis) value += localRows[axis].regularization;
                    m[3u*slot+axis]=value;
                    ok = ok && isfinite(value);
                }
            }
            if (!ok) { for (uint i=0u;i<9u;++i) m[i]=-INFINITY; }
            contact.friction=float4(m[0],m[1],m[2],m[3]);
            contact.response=float4(m[4],m[5],m[6],m[7]);
            contact.targetVelocityAndPreSolveNormal=float4(m[8],ids[0],ids[1],ids[2]);
        }
"""
(here/"baseline.metal").write_text(s)
for axis in range(3):
 (here/("axis%d.metal"%axis)).write_text(s[:pos]+instrument.replace("AXIS_VALUE",str(axis)+"u")+s[pos:])
(here/"probe.mm").write_bytes((old/"probe.mm").read_bytes())
