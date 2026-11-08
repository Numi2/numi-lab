// Research-only nine-variable projected contact solve. All quantities are
// thread-private; the caller publishes only a fully certified solution.
inline bool mrResearchJointResidual(
    thread const float* W, thread const float* j0, thread const float* r0,
    thread const float* js, thread const float* vs, thread const float* sn,
    thread const float* st, thread const float* mu, thread const float* y,
    thread float* f, thread float& error
) {
    float j[9], r[9];
    for (uint i=0u;i<9u;++i) {
        j[i]=j0[i]+js[i]*y[i];
        float d=0.0f;
        for (uint k=0u;k<9u;++k) d+=W[9u*i+k]*js[k]*y[k];
        r[i]=r0[i]-d;
        if (!isfinite(j[i]) || !isfinite(r[i])) return false;
    }
    error=0.0f;
    for (uint i=0u;i<3u;++i) {
        const uint k=3u*i;
        const float n=(j[k]-max(0.0f,j[k]+sn[i]*r[k]))/sn[i];
        const float2 q=float2(j[k+1u]+st[i]*r[k+1u],j[k+2u]+st[i]*r[k+2u]);
        const float radius=mu[i]*max(0.0f,j[k]);
        const float norm=length(q);
        const float2 projected=norm>radius && norm>0.0f ? q*(radius/norm) : q;
        const float2 t=(float2(j[k+1u],j[k+2u])-projected)/st[i];
        f[k]=n/vs[k]; f[k+1u]=t.x/vs[k+1u]; f[k+2u]=t.y/vs[k+2u];
        error=max(error,max(abs(n),length(t)));
        if (!isfinite(f[k]) || !isfinite(f[k+1u]) || !isfinite(f[k+2u])) return false;
    }
    return isfinite(error);
}

inline bool mrResearchJointSolve(
    thread const float* W, thread const float* j0, thread const float* r0,
    thread const float* sn, thread const float* st, thread const float* mu,
    thread float* result
) {
    float js[9],vs[9],y[9],f[9];
    for (uint i=0u;i<9u;++i) {
        js[i]=max(abs(j0[i]),1e-5f);
        vs[i]=max(abs(W[9u*i+i])*js[i],0.001f);
        y[i]=0.0f;
    }
    float error=INFINITY;
    if (!mrResearchJointResidual(W,j0,r0,js,vs,sn,st,mu,y,f,error)) return false;
    for (uint iteration=0u;iteration<8u && error>1e-5f;++iteration) {
        float augmented[9][10];
        for (uint c=0u;c<9u;++c) {
            float yp[9],ym[9],fp[9],fm[9],ep,em;
            for (uint i=0u;i<9u;++i) { yp[i]=y[i]; ym[i]=y[i]; }
            yp[c]+=0.002f; ym[c]-=0.002f;
            if (!mrResearchJointResidual(W,j0,r0,js,vs,sn,st,mu,yp,fp,ep) ||
                !mrResearchJointResidual(W,j0,r0,js,vs,sn,st,mu,ym,fm,em)) return false;
            for (uint r=0u;r<9u;++r) augmented[r][c]=(fp[r]-fm[r])/0.004f;
        }
        for (uint r=0u;r<9u;++r) augmented[r][9]=-f[r];
        for (uint c=0u;c<9u;++c) {
            uint pivot=c; float biggest=abs(augmented[c][c]);
            for (uint r=c+1u;r<9u;++r) if (abs(augmented[r][c])>biggest) {
                biggest=abs(augmented[r][c]); pivot=r;
            }
            if (!(biggest>1e-8f) || !isfinite(biggest)) return false;
            if (pivot!=c) for (uint k=c;k<10u;++k) {
                const float tmp=augmented[c][k]; augmented[c][k]=augmented[pivot][k]; augmented[pivot][k]=tmp;
            }
            const float inverse=1.0f/augmented[c][c];
            for (uint k=c;k<10u;++k) augmented[c][k]*=inverse;
            for (uint r=0u;r<9u;++r) if (r!=c) {
                const float factor=augmented[r][c];
                for (uint k=c;k<10u;++k) augmented[r][k]-=factor*augmented[c][k];
            }
        }
        float merit=0.0f;
        for (uint i=0u;i<9u;++i) merit+=f[i]*f[i];
        bool accepted=false;
        float alpha=1.0f;
        for (uint line=0u;line<12u;++line,alpha*=0.5f) {
            float trial[9],ft[9],et;
            for (uint i=0u;i<9u;++i) trial[i]=y[i]+alpha*augmented[i][9];
            if (!mrResearchJointResidual(W,j0,r0,js,vs,sn,st,mu,trial,ft,et)) continue;
            float trialMerit=0.0f;
            for (uint i=0u;i<9u;++i) trialMerit+=ft[i]*ft[i];
            if (trialMerit<merit) {
                for (uint i=0u;i<9u;++i) { y[i]=trial[i]; f[i]=ft[i]; }
                error=et; accepted=true; break;
            }
        }
        if (!accepted) return false;
    }
    if (!(error<=1e-5f)) return false;
    // Enforce admissibility exactly, then recertify after projection.
    for (uint i=0u;i<9u;++i) result[i]=j0[i]+js[i]*y[i];
    for (uint i=0u;i<3u;++i) {
        const uint k=3u*i;
        result[k]=max(result[k],0.0f);
        float2 t=float2(result[k+1u],result[k+2u]);
        const float radius=mu[i]*result[k]; const float norm=length(t);
        if (norm>radius && norm>0.0f) t*=radius/norm;
        result[k+1u]=t.x; result[k+2u]=t.y;
    }
    // A small projected normal residual alone can conceal a strictly
    // positive impulse on a separating row. Set that row to the inactive
    // branch, then re-evaluate the entire coupled block before publication.
    for (uint pass=0u;pass<3u;++pass) {
        bool changed=false;
        for (uint i=0u;i<3u;++i) {
            const uint k=3u*i;
            float rhs=r0[k];
            for (uint c=0u;c<9u;++c) rhs-=W[9u*k+c]*(result[c]-j0[c]);
            if (!isfinite(rhs)) return false;
            if (result[k]>0.0f && rhs < -1e-5f) {
                result[k]=0.0f; result[k+1u]=0.0f; result[k+2u]=0.0f;
                changed=true;
            }
        }
        if (!changed) break;
    }
    for (uint i=0u;i<9u;++i) y[i]=(result[i]-j0[i])/js[i];
    if (!mrResearchJointResidual(W,j0,r0,js,vs,sn,st,mu,y,f,error) || error>1e-5f) return false;
    for (uint i=0u;i<3u;++i) {
        const uint k=3u*i; float rhs=r0[k];
        for (uint c=0u;c<9u;++c) rhs-=W[9u*k+c]*(result[c]-j0[c]);
        const float normal=result[k]>0.0f ? abs(rhs) : max(rhs,0.0f);
        if (!isfinite(normal) || normal>1e-5f) return false;
    }
    return true;
}
