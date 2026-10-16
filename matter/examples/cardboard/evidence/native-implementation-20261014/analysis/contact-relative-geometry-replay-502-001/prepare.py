from pathlib import Path
import hashlib,json
B=Path('/Users/home/cardboard-implementation-20261014');P=B/'analysis/contact-armijo-energy-replay-499-001';O=B/'analysis/contact-relative-geometry-replay-502-001'
assert not O.exists();O.mkdir()
k=(P/'contact_barrier_trial_replay.metal').read_text()
helper='''
// Diagnostic only: evaluate the same contact energy after translating the
// nearest-point problem to the query origin. No production source is changed.
inline bool localBarrierEnergyAt(const NMDeformableContactGPU contact,
    thread const float3* position, thread float& energy, thread float& separation) {
    float m=1.0f;
    if (contact.identity.z==NM_DEFORMABLE_CONTACT_POINT_POINT) {
        separation=length(position[0]-position[1]);
    } else if (contact.identity.z==NM_DEFORMABLE_CONTACT_VERTEX_TRIANGLE) {
        float3 bary;
        const float3 a=position[1]-position[0];
        const float3 b=position[2]-position[0];
        const float3 c=position[3]-position[0];
        const float3 q=closestPointOnTriangle(float3(0.0f),a,b,c,bary);
        separation=length(q);
        float3 g1,g2;
        m=ipcFeatureMollifierDifferential(position[2]-position[1],position[3]-position[1],g1,g2);
    } else if(contact.identity.z==NM_DEFORMABLE_CONTACT_EDGE_EDGE) {
        float s,t;float3 q1,q2;
        closestPointsOnSegments(float3(0.0f),position[1]-position[0],
            position[2]-position[0],position[3]-position[0],s,t,q1,q2);
        separation=length(q1-q2);
        float3 g1,g2;
        m=ipcFeatureMollifierDifferential(position[1]-position[0],position[3]-position[2],g1,g2);
    } else {energy=INFINITY;separation=-INFINITY;return false;}
    energy=m*ipcBarrierEnergy(separation,contact.barrier.z,contact.mollifier.w);
    return isfinite(separation)&&isfinite(energy);
}
'''
k=k.replace('namespace numi_matter_metal {','namespace numi_matter_metal {\n'+helper,1)
k=k.replace('deformableBarrierEnergyAt(', 'localBarrierEnergyAt(')
(O/'contact_barrier_trial_replay.metal').write_text(k)
(O/'contact_barrier_trial_replay.mm').write_bytes((P/'contact_barrier_trial_replay.mm').read_bytes())
(O/'prepare.py').write_bytes(Path(__file__).read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(O/'prediction.json').write_text(json.dumps({'schema':'numi.cardboard.relative-contact-geometry-replay-prediction.v1','base_energy_replay_output_sha256':'cb1f014b70d5bd21e08f385e4fbea2c6d857328b2fd4a9c0506d64cb18ff0237','source176_closure_sha256':'8427a7f247282ac8488061975174830b9ccfaaba4918dff626a2b9673b03ca41','case_manifest_sha256':'dc62eb0462fbf4af9f495363c63bb5ef692213643da0f318602eba3b91c08370','kernel_sha256':sha(O/'contact_barrier_trial_replay.metal'),'host_source_sha256':sha(O/'contact_barrier_trial_replay.mm'),'purpose':'Evaluate nearest-point geometry in query-relative coordinates; unchanged IPC energy law. Compare with exact same captured positions and independent FP64/MP metric. Diagnostic only; no production/contact/CCD qualification.'},indent=2)+'\n')
print(sha(O/'prediction.json'))
