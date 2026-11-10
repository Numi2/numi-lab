from pathlib import Path
import subprocess,difflib
R=Path.cwd();A=R/'artifacts/crow-v15-hybrid-ccd-failure-20261110';B=R/'build/crow-v15-hybrid-ccd-failure-20261110';old=(R/'src/metal/Collision.metal').read_text();needle='''    if ((shapeA.type == MR_SHAPE_SPHERE &&
         shapeB.type == MR_SHAPE_BOX) ||''';assert old.count(needle)==1
insert='''    if (shapeA.type == MR_SHAPE_BOX && shapeB.type == MR_SHAPE_BOX) {
        // Research-only SAT lower bound for conservative advancement.
        float3 axes[15]; uint count=0u;
        for (uint i=0u;i<3u;++i) axes[count++]=boxAxis(shapeA,i);
        for (uint i=0u;i<3u;++i) axes[count++]=boxAxis(shapeB,i);
        for (uint i=0u;i<3u;++i) for(uint j=0u;j<3u;++j) {
            const float3 c=cross(boxAxis(shapeA,i),boxAxis(shapeB,j));
            const float n=dot(c,c); if(n>1.0e-12f) axes[count++]=c*rsqrt(n);
        }
        const float3 centerDelta=shapeB.center-shapeA.center;
        float best=-INFINITY;float3 normal=coincidentNormal(colliderA,colliderB);
        for(uint i=0u;i<count;++i){
            float3 axis=axes[i];const float p=dot(centerDelta,axis);
            if(p<0.0f || (p==0.0f && dot(axis,normal)<0.0f)) axis=-axis;
            const float separation=abs(p)-boxProjectionRadius(shapeA,axis)-boxProjectionRadius(shapeB,axis);
            if(separation>best){best=separation;normal=axis;}
        }
        query={};query.status=MR_STEP_SUCCESS;query.normal=normal;
        query.separation=best-pairQueryPadding(shapeA,shapeB);
        query.pointA=boxSupport(shapeA,normal);query.pointB=boxSupport(shapeB,-normal);
        query.featureA=featureKey(MR_SHAPE_BOX,0u);query.featureB=featureKey(MR_SHAPE_BOX,0u);
        return true;
    }
'''
s=old.replace(needle,insert+needle);(A/'box-sat.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/Collision.metal',tofile='b/Collision.metal')));p=B/'collision-sat.metal';p.write_text(s)
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(p),'-o',str(B/'collision-sat.air')],check=True);other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name!='MetalRoboCollision.air');subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(x) for x in other],str(B/'collision-sat.air'),'-o',str(B/'collision-sat.metallib')],check=True)
sweep=(R/'apps/ccd_sweep_probe.metal').read_text().replace('#include "../src/metal/Collision.metal"','#include "collision-sat.metal"');q=B/'sweep-sat.metal';q.write_text(sweep);subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(B),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(q),'-o',str(B/'sweep-sat.air')],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',str(B/'sweep-sat.air'),'-o',str(B/'sweep-sat.metallib')],check=True)
