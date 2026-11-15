from pathlib import Path
import hashlib,json,subprocess
r=Path('/Users/n/Developer/numi-crow-rl-20261107');a=r/'artifacts/crow-v15-recorded-restart-20261114';b=r/'build/crow-v15-recorded-restart-20261114'
s=(a/'restart-round-neural-balance-masked.metal').read_text()
assert s.count('4991u)-0.5f)*0.004f')==1 and s.count('4992u)-0.5f)*0.004f')==1 and s.count('4993u)-0.5f)*0.04f')==1
s=s.replace('4991u)-0.5f)*0.004f','4991u)-0.5f)*0.012f').replace('4992u)-0.5f)*0.004f','4992u)-0.5f)*0.012f').replace('4993u)-0.5f)*0.04f','4993u)-0.5f)*0.10f')
old='''        const bool bilateral=(state.status.w & 63u)>=12u && ((state.status.w>>6u)&63u)>=12u;
        const float support=bilateral ? 1.0f : 0.0f;
        reward=dispatch.timing.x*(2.0f*support - 2.0f*min(tilt*tilt,4.0f)
              -0.4f*min(dot(rootLinearVelocity,rootLinearVelocity),4.0f)
              -0.05f*min(dot(baseAngular,baseAngular),25.0f));'''
new='''        // Last-substep contact is a training signal; evaluation sums all substeps.
        uint perch=MR_INVALID_INDEX;
        for (uint ri=0u;ri<program.counts1.w;++ri) {
            const MRTaskRewardOperatorGPU op=rewards[ri];
            if (op.source.x==MR_TASK_REWARD_NAVIGATION_WAYPOINT_PROGRESS || op.source.x==MR_TASK_REWARD_NAVIGATION_WAYPOINT_REACH) { perch=robotEnd+op.source.z+4u; break; }
        }
        float up[2]={0.0f,0.0f},front[2]={0.0f,0.0f},rear[2]={0.0f,0.0f};
        for (uint ci=0u;ci<activeContacts && perch!=MR_INVALID_INDEX;++ci) {
            const MRContactConstraintGPU cc=contacts[contactBase+ci];
            if (cc.impulses.x<=0.0f || (cc.bodyA!=perch && cc.bodyB!=perch)) continue;
            const uint b=cc.bodyA==perch ? cc.bodyB : cc.bodyA;
            const int side=(b==robotFirst+11u || (b>=robotFirst+19u && b<=robotFirst+26u)) ? 0 : (b==robotFirst+15u || (b>=robotFirst+27u && b<=robotFirst+34u)) ? 1 : -1;
            if(side<0) continue;
            const float sign=cc.bodyA==perch ? 1.0f : -1.0f;
            const float3 normal=cc.normal.xyz,tangent=cc.tangent.xyz,other=cross(normal,tangent);
            up[side]+=sign*(normal.z*cc.impulses.x+tangent.z*cc.impulses.y+other.z*cc.impulses.z);
            if(b>=robotFirst+19u) {const uint digit=((b-robotFirst-19u)%8u)/2u; if(digit==3u) rear[side]+=sign*normal.x*cc.impulses.x; else front[side]+=sign*normal.x*cc.impulses.x;}
        }
        const float expected=.5f*.45006f*9.81f*dispatch.timing.x/16.0f;
        float support=0.0f,opposed=0.0f;
        for(uint side=0u;side<2u;++side) {
            support+=clamp(up[side]/max(expected*.5f,1.0e-8f),0.0f,1.0f)*0.5f;
            opposed+=clamp(front[side]/1.0e-5f,0.0f,1.0f)*clamp(-rear[side]/1.0e-5f,0.0f,1.0f)*0.5f;
        }
        reward=dispatch.timing.x*(1.5f*support + 1.0f*opposed - 4.0f*min(tilt*tilt,4.0f)
              -1.0f*min(dot(rootLinearVelocity,rootLinearVelocity),4.0f)
              -0.1f*min(dot(baseAngular,baseAngular),25.0f));'''
assert s.count(old)==1;s=s.replace(old,new)
src=a/'restart-round-neural-balance-v2.metal';src.write_text(s)
build=json.loads((a/'shader-build.json').read_text());cc=build['compile'];cc[cc.index('-c')+1]=str(src);air=b/'restart-round-neural-balance-v2.air';lib=b/'restart-round-neural-balance-v2.metallib';cc[cc.index('-o')+1]=str(air);subprocess.run(cc,check=True)
link=build['link'];link[link.index(str(b/'restart.air'))]=str(air);link[link.index('-o')+1]=str(lib);subprocess.run(link,check=True)
(a/'neural-v2-build.json').write_text(json.dumps({'compile':cc,'link':link,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (src,air,lib)}},indent=2)+'\n')
