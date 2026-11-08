from pathlib import Path
import hashlib,json
p=Path('/Users/numi/frictionless-mixed-rod-20261108');out=Path('.')
source=(p/'split.metal').read_text();needle='''        laneNormalResidual = max(
            laneNormalResidual,
            contact.impulses.x > kConeEpsilon
            ? abs(normalEquation)
            : max(-normalEquation, 0.0f)
        );'''
assert source.count(needle)==1
source=source.replace(needle,needle+'''
        // Diagnostic-only: preserve the physical final row velocities.
        contact.friction.xyz = float3(
            dot(localRows[0].direction.xyz, relative),
            dot(localRows[1].direction.xyz, relative),
            dot(localRows[2].direction.xyz, relative));''')
(out/'slip-trace.metal').write_text(source)
s=(p/'sweep_probe.cpp').read_text()
a='for(int iterations:{1,2,4,8,16,32,64}) for(float mu:{.5f,0.f}) for(float vn:{-1.f,0.f,1.f}){'
b='''for(int iterations:{2,64}) for(float mu:{.5f,0.f}) for(float vn:{-1.f,0.f,1.f}) for(int dir=0;dir<5;++dir){
  const float vx[5]={1.f,-1.f,0.f,0.f,0.f},vz[5]={0.f,0.f,1.f,-1.f,0.f};'''
assert s.count(a)==1;s=s.replace(a,b)
a='m.defaultV[0]=1.f;m.defaultV[1]=vn;';b='m.defaultV[0]=vx[dir];m.defaultV[1]=vn;m.defaultV[2]=vz[dir];'
assert s.count(a)==1;s=s.replace(a,b)
a='<<",\\\"initial_vn\\\":"<<vn';b='<<",\\\"dir\\\":"<<dir<<",\\\"initial_vn\\\":"<<vn'
assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
a='<<"],\\\"target\\\":"<<r.targetVelocity';b='<<"],\\\"post_relative\\\":["<<c.friction.x<<","<<c.friction.y<<","<<c.friction.z<<"],\\\"target\\\":"<<r.targetVelocity'
assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
a='<<"],\\\"anchor_a\\\":["';b='<<"],\\\"tangent\\\":["<<c.tangent.x<<","<<c.tangent.y<<","<<c.tangent.z<<"],\\\"anchor_a\\\":["'
assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
(out/'probe.cpp').write_text(s)
for filename in ['PLAN.md','prepare.py','probe.cpp','slip-trace.metal']:
 data=Path(filename).read_bytes();print(hashlib.sha256(data).hexdigest(),filename)
