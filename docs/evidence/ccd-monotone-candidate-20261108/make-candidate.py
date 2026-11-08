from pathlib import Path
p=Path('Collision.before.metal');s=p.read_text()
needle='''    clear=true;lower=1;upper=1;work=0;
    for(uint corner=0;corner<8;++corner) {'''
assert s.count(needle)==1
insert='''    clear=true;lower=1;upper=1;work=0;
    // Research-only whole-sweep certificate. Each corner must be clear by a
    // direct Bernstein bound or monotonically separating from a guarded start.
    // Otherwise the original per-vertex subdivision remains authoritative.
    CCDI endpointDot={0,0};
    for(uint i=0;i<4;++i)endpointDot=ccdQuatAdd(endpointDot,ccdQuatMul(q0[i],q1[i]));
    const CCDI db[3]={D0,ccdQuatAdd(D0,ccdScale(D1,.5f)),ccdQuatAdd(ccdQuatAdd(D0,D1),D2)};
    bool whole=tolerance>=0.f&&centerDelta.lo>0.f&&endpointDot.lo>0.f;
    for(uint k=0;k<3;++k)whole=whole&&db[k].lo>0.f&&isfinite(db[k].lo);
    if(whole) {
        const CCDI kLower=ccdQuatMul(ccdScalar(centerDelta.lo),ccdScalar(endpointDot.lo));
        for(uint corner=0;corner<8&&whole;++corner) {
            const float3 v=box.dimensions.xyz*float3((corner&1)?1:-1,(corner&2)?1:-1,(corner&4)?1:-1);
            const CCDI r0=ccdRotatedProjection(q0,v,n),r2=ccdRotatedProjection(delta,v,n);
            const CCDI r1=ccdQuatAdd(ccdQuatAdd(ccdRotatedProjection(q1,v,n),ccdQuatNeg(r0)),ccdQuatNeg(r2));
            const CCDI rb[3]={r0,ccdQuatAdd(r0,ccdScale(r1,.5f)),ccdQuatAdd(ccdQuatAdd(r0,r1),r2)};
            bool direct=true;
            for(uint k=0;k<3;++k)direct=direct&&ccdQuatAdd(rb[k],ccdQuatMul(center0,db[k])).lo>0.f;
            if(direct)continue;
            const bool start=ccdQuatAdd(r0,ccdQuatMul(center0,D0)).lo>0.f;
            const CCDI p0=ccdQuatAdd(ccdQuatMul(r1,D0),ccdQuatNeg(ccdQuatMul(r0,D1)));
            const CCDI p1=ccdScale(ccdQuatAdd(ccdQuatMul(r2,D0),ccdQuatNeg(ccdQuatMul(r0,D2))),2.f);
            const CCDI p2=ccdQuatAdd(ccdQuatMul(r2,D1),ccdQuatNeg(ccdQuatMul(r1,D2)));
            const CCDI pb[3]={p0,ccdQuatAdd(p0,ccdScale(p1,.5f)),ccdQuatAdd(ccdQuatAdd(p0,p1),p2)};
            bool monotone=start;
            for(uint k=0;k<3;++k)monotone=monotone&&ccdQuatAdd(pb[k],ccdQuatMul(kLower,db[k])).lo>0.f;
            whole=monotone;
        }
        if(whole)return true;
    }
    for(uint corner=0;corner<8;++corner) {'''
s=s.replace(needle,insert);Path('Collision.candidate.metal').write_text(s)
