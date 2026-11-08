import pathlib,runpy
p=pathlib.Path(__file__).resolve().parent
g=runpy.run_path(str(p/"prepare.py"))
s=g["s"];pos=g["pos"];instrument=g["instrument"]
old="value += dot(targetDirection,exact) - rodCrossContactResponse("
assert instrument.count(old)==1
for axis in range(3):
 text=instrument.replace(old,"value -= rodCrossContactResponse(").replace("AXIS_VALUE",str(axis)+"u")
 (p/("rigid%d.metal"%axis)).write_text(s[:pos]+text+s[pos:])
factor=r"""
        if (work.constraintCount == 3u && work.tileCount == 1u &&
            dispatch.rodNodeCount == 3u && dispatch.rodEdgeCount == 2u &&
            dispatch.rodCount == 1u && localConstraint < 3u) {
            const MRRodFactorCacheGPU cache = rodFactorCaches[environment];
            const bool valid = cache.environment == environment && cache.rodIndex == 0u &&
                cache.velocityCount == 3u && cache.blockWidth == 2u &&
                cache.code == MR_ROD_GPU_SUCCESS && (cache.flags & MR_ROD_FACTOR_CACHE_VALID) != 0u;
            float v[12];
            for (uint k=0u; k<12u; ++k) {
                const uint idx = PART_VALUE * 36u + localConstraint * 12u + k;
                v[k] = !valid ? -INFINITY : idx < 85u ? rodOperatorArena[cache.firstBlock + idx] : 0.0f;
            }
            contact.friction=float4(v[0],v[1],v[2],v[3]);
            contact.response=float4(v[4],v[5],v[6],v[7]);
            contact.targetVelocityAndPreSolveNormal=float4(v[8],v[9],v[10],v[11]);
        }
"""
for part in range(3):
 (p/("factor%d.metal"%part)).write_text(s[:pos]+factor.replace("PART_VALUE",str(part)+"u")+s[pos:])
