from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/"artifacts/crow-v15-perch-ankle-refine-20261110";B=R/"build/crow-v15-perch-ankle-refine-20261110";base=(R/"build/crow-v15-landing-toe-ramp-20261110/toe-ramp.metal").read_text()
needle="""            state.status.w=packed;
"""
insert="""            // Accepted contact resultant and articulated COM supply a bounded
            // proprioceptive support error; no force is applied by this observer.
            if (state.navigation.z>=5.0f && state.threatTeacher.w>=1.5f) {
                float3 weightedCOM=float3(0.0f);float totalMass=0.0f;
                const float masses[25]={0.3506562412f,0.00001f,0.00001f,0.00001f,0.00001f,0.0001406249939f,0.0001406249939f,0.01406249963f,0.009843749925f,0.007031249814f,0.005225000437f,0.009843749925f,0.007031249814f,0.005225000437f,0.001000000047f,0.03500000015f,0.00400000019f,0.0001f,0.0001f,0.0001f,0.0001f,0.0001f,0.0001f,0.0001f,0.0001f};
                if(robotEnd-robotFirst==25u) for(uint bi=robotFirst;bi<robotEnd;++bi) {
                    const float mass=masses[bi-robotFirst];
                    weightedCOM+=mass*bodyStates[bodyBase+bi].position.xyz;totalMass+=mass;
                }
                float3 support=float3(0.0f);float totalUp=0.0f;
                for(uint ci=0u;ci<activeContacts;++ci) {
                    const MRContactConstraintGPU c=contacts[contactBase+ci];
                    const bool ra=c.bodyA>=robotFirst && c.bodyA<robotEnd;
                    const bool rb=c.bodyB>=robotFirst && c.bodyB<robotEnd;
                    if(ra==rb)continue;
                    const uint body=ra ? c.bodyA : c.bodyB;
                    if(!bodyMember(body,contactGroups[0],contactMembers) && !bodyMember(body,contactGroups[1],contactMembers))continue;
                    const float up=max((ra ? -1.0f : 1.0f)*c.normal.z*c.impulses.x,0.0f);
                    support+=up*c.pointAndSeparation.xyz;totalUp+=up;
                }
                if(totalMass>0.0f && totalUp>1.0e-7f) {
                    const float3 delta=support/totalUp-weightedCOM/totalMass;
                    const float along=dot(delta.xy,yawBasis);
                    const uint encoded=uint(clamp(round(along*500.0f)+64.0f,0.0f,127.0f));
                    packed |= encoded<<25u;
                }
            }
            state.status.w=packed;
"""
assert base.count(needle)==1;s0=base.replace(needle,insert)
needle="""        if (fixedFlightController && state.threatTeacher.w >= 1.5f && action < 15u)
            navigationTeacherRequested = 0.0f;
"""
assert s0.count(needle)==1
for name,kp,kd in (("ankle12","12.0","1.0"),("ankle18","18.0","1.5")):
 extra="""        if (fixedFlightController && state.episode.z==10u && state.navigation.z>=5.0f &&
            state.threatTeacher.w>=1.5f && (state.status.w&63u)>=28u && ((state.status.w>>6u)&63u)>=28u && (action==7u || action==9u || action==10u || action==12u) &&
            ((action==7u || action==9u) ? state.nutrition.x : state.nutrition.y)>0.05f) {
            const uint encoded=(state.status.w>>25u)&127u;
            if(encoded!=0u) {
                const float supportError=(float(encoded)-64.0f)*0.002f;
                const float correction=clamp("""+kp+"""f*supportError-"""+kd+"""f*state.commandExtension.z,-0.60f,0.60f);
                navigationTeacherRequested=(action==9u || action==12u) ? -1.077f*correction : correction;
            }
        }
"""
 s=s0.replace(needle,needle+extra);src=B/(name+".metal");src.write_text(s);(A/(name+".patch")).write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="a/toe-ramp.metal",tofile="b/"+name+".metal")));air=B/(name+".air");lib=B/(name+".metallib");subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(src),"-o",str(air)],check=True);subprocess.run(["xcrun","-sdk","macosx","metallib",*[str(p) for p in sorted((R/"build/shaders").glob("MetalRobo*.air")) if p.name!="MetalRoboLocomotionTask.air"],str(air),"-o",str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
