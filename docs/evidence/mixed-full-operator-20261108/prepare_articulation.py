import pathlib,runpy
p=pathlib.Path(__file__).resolve().parent
g=runpy.run_path(str(p/"prepare.py"));s=g["s"];pos=g["pos"]
capture=r"""
        if (work.constraintCount == 3u && work.tileCount == 1u &&
            dispatch.rodNodeCount == 3u && dispatch.rodEdgeCount == 2u && dispatch.nv == 6u) {
            float v[12];
            for(uint k=0u;k<12u;++k) {
                const uint i=PART_VALUE*12u+k;
                if(i<18u) {
                    const uint a=i/6u, dof=i%6u;
                    v[k]=dot(localRows[a].direction.xyz,typedArticulationJacobianColumn(
                        localConstraint,contact,bodies,pointJacobians,pointJacobianBase,dof,dispatch.nv,
                        rodWitnesses,constraintWitnessIndices,constraintBase));
                } else if(i<21u) v[k]=localRows[i-18u].regularization;
                else v[k]=0.0f;
            }
            contact.friction=float4(v[0],v[1],v[2],v[3]);
            contact.response=float4(v[4],v[5],v[6],v[7]);
            contact.targetVelocityAndPreSolveNormal=float4(v[8],v[9],v[10],v[11]);
        }
"""
for i in range(2):(p/("jac%d.metal"%i)).write_text(s[:pos]+capture.replace("PART_VALUE",str(i)+"u")+s[pos:])
