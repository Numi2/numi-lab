"""One-shot source-hashed contact/equality integration. Retire after testing."""
from pathlib import Path
import hashlib
import os

root = Path(__file__).resolve().parents[1]
os.chdir(root)
hashes = {
    'include/metalrobo/numi_human_stand_gpu.h': ('e617e7f8806a666a530d6a8494b1045e9621a2b54358ac712746295a89d67ee1', '01a456af8a1897ce39b015446bb755e206b159d8956dbcabf8dab732b82f81ae'),
    'src/metal/MetalArticulatedOperator.mm': ('b36e57f3b928309bbc9c3dde52223052c8196a0df0979180cc6b826d03d504cd', '9acc7a2ace4616448a47987e8b65fbac4a5f759632fce447d476fc9a0d7447fc'),
    'src/metal/NumiHumanStand.metal': ('91c2e186dde892b8f9fdd925d8133bf6497be745a6ac579a37bd23eb256efdf9', 'e4893837e19aa0d7cde7969106a9d3d4f893c7b13b342d10bb5ce1c85a17adfc'),
    'tests/numi_human_bilateral_metal_test.mm': ('c75148c3e1746d0837629f0bd3a6a16a27199015bf24ceca9b875c5016712683', '9bd466ca739f03e01b3041ef798d027689a6c0cd237b06c8217c698a0c99da69'),
    'tools/check_human_constraint_owner.py': ('a435c0999eb3982fef3387878ee77aa0d7b84707ae489d6fac42dd1c09896df9', '25c69c4d36a21ba064179e2cb60002d2a625a50cce7d7f08c40c64554a915547'),
}
for name, (before, _) in hashes.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == before, name

p=Path('src/metal/NumiHumanStand.metal')
s=p.read_text()
helper='''// Apply the existing full bilateral Schur projector to a response column.
// Coefficients retain the equality impulse accompanying each unit unilateral
// impulse. Two residual corrections use the same factor, without compliance.
inline bool conditionBilateralResponse(
    device float* response,
    device float* equalityCorrection,
    device const float* equalityFactor,
    device const float* equalityScale,
    device const float* equalityPivots,
    device float* equalityRhs,
    device const float* responseScratch,
    const uint responseBase,
    const uint supportCount,
    device const MRNumiHumanJointEqualityGPU* jointEqualities,
    device const float* q,
    const uint qBase, const uint nq, const uint nv, const uint equalityCount
) {
    for (uint ei = 0u; ei < equalityCount; ++ei)
        equalityCorrection[ei] = 0.0f;
    if (equalityCount == 0u) return true;
    for (uint refinement = 0u; refinement < 2u; ++refinement) {
        for (uint ei = 0u; ei < equalityCount; ++ei) {
            device const MRNumiHumanJointEqualityGPU& eq = jointEqualities[ei];
            float target = 0.0f, derivative = 0.0f, error = 0.0f;
            if (!evaluateJointEquality(eq, q, qBase, nq, nv,
                                       target, derivative, error)) return false;
            float residual = response[eq.indices.y];
            if (eq.indices.w != MR_INVALID_INDEX)
                residual = fma(-derivative, response[eq.indices.w], residual);
            equalityRhs[ei] = residual;
        }
        if (!mrNumiHumanBilateralSolve(equalityFactor, equalityScale,
                equalityPivots, equalityRhs, equalityCount)) return false;
        for (uint ei = 0u; ei < equalityCount; ++ei) {
            equalityCorrection[ei] -= equalityRhs[ei];
            if (!isfinite(equalityCorrection[ei])) return false;
        }
        for (uint index = 0u; index < nv; ++index) {
            float correction = 0.0f;
            for (uint ei = 0u; ei < equalityCount; ++ei) {
                device const float* er = responseScratch + responseBase +
                    (3u * supportCount + ei) * nv;
                correction = fma(equalityRhs[ei], er[index], correction);
            }
            response[index] -= correction;
            if (!isfinite(response[index])) return false;
        }
    }
    return true;
}

'''
assert s.count('inline void fail(')==1
s=s.replace('inline void fail(',helper+'inline void fail(',1)
s=s.replace('nv * equalityCount;', '(nv + 3u * dispatch.supportContactCount) * equalityCount;',1)
s=s.replace('limitEqualityCorrections = equalityRhs + equalityCount;', '''limitEqualityCorrections = equalityRhs + equalityCount;
    device float* contactEqualityCorrections = limitEqualityCorrections + nv * equalityCount;''',1)
start=s.index('    // Equality rows use the same factored mass matrix as contact.')
end=s.index('    if ((dispatch.flags & MR_NUMI_HUMAN_STAND_ENABLE_CONTACT) != 0u) {\n    // Source-authored scalar position limits',start)
eq=s[start:end]
s=s[:start]+s[end:]
eq=eq.replace('''    // Equality rows use the same factored mass matrix as contact. Solving
    // them in the coupled sweep keeps the authored anatomical manifold and
    // active unilateral rows mutually consistent without a hidden motor.''','''    // Establish the bilateral tangent first. Both unilateral response
    // families preserve it using this same factor and equality reactions.''')
needle='''         ++coupledSweep) {

    if ((dispatch.flags & MR_NUMI_HUMAN_STAND_ENABLE_CONTACT) != 0u) {'''
assert s.count(needle)==1
s=s.replace(needle,'''         ++coupledSweep) {

'''+eq+'''    if ((dispatch.flags & MR_NUMI_HUMAN_STAND_ENABLE_CONTACT) != 0u) {''',1)
needle='''                if (!solveFactor(factor, workspace, response, nv)) {
                    fail(status, MR_NUMI_HUMAN_STAND_CONTACT_FAILED, contact);
                    return;
                }
'''
assert s.count(needle)==1
s=s.replace(needle,needle+'''                device float* equalityCorrection = contactEqualityCorrections +
                    (3u * contact + axis) * equalityCount;
                if (!conditionBilateralResponse(response, equalityCorrection,
                        equalityFactor, equalityScale, equalityPivots, equalityRhs,
                        responseScratch, responseBase, dispatch.supportContactCount,
                        jointEqualities, qState, qBase, nq, nv, equalityCount)) {
                    fail(status, MR_NUMI_HUMAN_STAND_CONTACT_FAILED, contact);
                    return;
                }
''',1)
needle='''            for (uint dof = 0u; dof < nv; ++dof) {
                candidateV[dof] += seed * normalResponse[dof];
            }
'''
assert s.count(needle)==1
s=s.replace(needle,needle+'''            device const float* seedEqualityCorrection = contactEqualityCorrections +
                (3u * contact) * equalityCount;
            for (uint ei = 0u; ei < equalityCount; ++ei)
                equalityLambdas[ei] = fma(seed, seedEqualityCorrection[ei], equalityLambdas[ei]);
''',1)
needle='''                for (uint axis = 0u; axis < 3u; ++axis) {
                    device const float* response = responseScratch + responseBase +
                        (3u * contact + axis) * nv;
                    for (uint dof = 0u; dof < nv; ++dof) {
                        candidateV[dof] += applied[axis] * response[dof];
                    }
                }
'''
assert s.count(needle)==1
s=s.replace(needle,needle.replace('''                    for (uint dof''','''                    device const float* equalityCorrection = contactEqualityCorrections +
                        (3u * contact + axis) * equalityCount;
                    for (uint ei = 0u; ei < equalityCount; ++ei)
                        equalityLambdas[ei] = fma(applied[axis], equalityCorrection[ei], equalityLambdas[ei]);
                    for (uint dof'''),1)
start=s.index('            for (uint refinement = 0u; refinement < 2u; ++refinement) {', s.index('// Eliminate ALL bilateral rows'))
end=s.index('            // Cancellation in a direction already fixed by E', start)
s=s[:start]+'''            if (!conditionBilateralResponse(response, equalityCorrection,
                    equalityFactor, equalityScale, equalityPivots, equalityRhs,
                    responseScratch, responseBase, dispatch.supportContactCount,
                    jointEqualities, qState, qBase, nq, nv, equalityCount)) {
                fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED, dof);
                return;
            }
'''+s[end:]
s=s.replace('// Final equality evidence includes full-block limit corrections.', '// Final equality evidence includes full-block contact and limit corrections.')
p.write_text(s)
p=Path('src/metal/MetalArticulatedOperator.mm');s=p.read_text()
s=s.replace('std::size_t bilateralScratchElements = 0u;', 'std::size_t bilateralScratchElements = 0u;\n        std::size_t contactEqualityColumns = 0u;',1)
needle='''            !checkedMultiply(input.stand.jointEqualities.size(),
                             bilateralScratchElements, bilateralScratchElements) ||'''
assert s.count(needle)==1
s=s.replace(needle,'''            !checkedMultiply(input.stand.contacts.size(), 3u,
                             contactEqualityColumns) ||
            !checkedAdd(bilateralScratchElements, contactEqualityColumns,
                        bilateralScratchElements) ||
'''+needle,1)
s=s.replace('vectors and one equality correction per possible limit column.', 'vectors and equality corrections per contact/limit response column.')
p.write_text(s)
p=Path('include/metalrobo/numi_human_stand_gpu.h')
p.write_text(p.read_text().replace('#define MR_NUMI_HUMAN_STAND_ABI_VERSION 11u','#define MR_NUMI_HUMAN_STAND_ABI_VERSION 12u'))
p=Path('tools/check_human_constraint_owner.py');s=p.read_text()
s=s.replace('Final equality evidence includes full-block limit corrections.','Final equality evidence includes full-block contact and limit corrections.')
s=s.replace('require("nv * equalityCount" in kernel,', 'require("(nv + 3u * dispatch.supportContactCount) * equalityCount" in kernel,')
s += '''\nrequire("contactEqualityCorrections" in kernel and\n        kernel.count("conditionBilateralResponse(response, equalityCorrection,") == 2,\n        "contact and limit families must share complete bilateral conditioning")\nrequire(kernel.index("mrNumiHumanBilateralFactor(") <\n        kernel.index("mrNumiHumanSupportSeedImpulse("),\n        "bilateral factor must precede the support warm start")\n'''
p.write_text(s)
p=Path('tests/numi_human_bilateral_metal_test.mm');s=p.read_text()
s=s.replace('bool loaded,unsigned limitedDof) {','''bool loaded,unsigned limitedDof,
                  unsigned contactBody=MR_INVALID_INDEX,float seedImpulse=0.0f,bool closing=true) {
    const unsigned nc = contactBody == MR_INVALID_INDEX ? 0u : 1u;
    require(nc == 0u || limitedDof == MR_INVALID_INDEX,
            "contact oracle excludes a simultaneous finite stop");''',1)
s=s.replace('envs*(4*nv+neq)*sizeof(float)', 'envs*(4*nv+neq+12*nc)*sizeof(float)',1)
s=s.replace('envs*((neq+nv)*nv+neq*(neq+3+nv))*sizeof(float)', 'envs*((3*nc+neq+nv)*nv+neq*(neq+3+nv+3*nc))*sizeof(float)',1)
s=s.replace('    if (limitedDof != MR_INVALID_INDEX)\n        d->flags |=', '    if (limitedDof != MR_INVALID_INDEX || nc != 0u)\n        d->flags |=',1)
needle='''    d->groundPointAndTimestep={0,0,0,h};d->groundNormal={0,0,1,0};d->targetRootOrientation={0,0,0,1};'''
assert s.count(needle)==1
s=s.replace(needle,needle+'''
    if (nc != 0u) {
        d->supportContactCount=nc;
        d->groundNormal={0,1,0,0};
        auto* contact=static_cast<MRNumiHumanStandContactGPU*>(b[11].contents);
        contact->bodyIndex=contactBody;contact->pointQueryIndex=4*contactBody+1;
        // A y-normal impulse at x=1 changes both linear and angular momentum.
        // Zero and excessive seeds must produce the same cold-start solution.
        contact->frictionSlopAndStabilization={0,1e-5f,0.2f,seedImpulse/h};
    }''',1)
s=s.replace('    std::array<std::array<double,4>,envs> expected{},old{};', '''    std::array<std::array<double,4>,envs> expected{},old{};
    std::array<double,envs> expectedNormal{},oldLinear{},expectedLinear{};''',1)
s=s.replace('        q[e*nq+6]=1;', '''        q[e*nq+6]=1;
        oldLinear[e]=nc ? (closing ? -2.0 : 2.0) : 0.0;
        v[e*nv+1]=static_cast<float>(oldLinear[e]);
        expectedLinear[e]=oldLinear[e];''',1)
needle='''        expected[e][2]=c1*expected[e][1];expected[e][3]=c2*expected[e][1];'''
assert s.count(needle)==1
s=s.replace(needle,'''        if (nc != 0u) {
            // Reduce the equality-constrained model analytically to root
            // translation plus a 2x2 angular inertia, then impose one normal
            // contact. This oracle shares no factor/projector with production.
            const double multiplier=contactBody==0u ? 0.0 : t[contactBody-1];
            const double angularRootResponse=(c-bb*multiplier)/determinant;
            const double jointResponse=(a*multiplier-bb)/determinant;
            const double inverseContactMass=0.25+angularRootResponse+multiplier*jointResponse;
            const double freeNormal=oldLinear[e]+expected[e][0]+multiplier*expected[e][1];
            require(inverseContactMass>0.0,"independent contact inertia is invalid");
            expectedNormal[e]=std::max(0.0,-freeNormal/inverseContactMass);
            expectedLinear[e]+=0.25*expectedNormal[e];
            expected[e][0]+=angularRootResponse*expectedNormal[e];
            expected[e][1]+=jointResponse*expectedNormal[e];
        }
'''+needle,1)
s=s.replace('<<" limitedDof="<<limitedDof<<" got="', '<<" limitedDof="<<limitedDof<<" contactBody="<<contactBody<<" got="')
s=s.replace('''        require(std::abs(newMomentum-oldMomentum)<1e-5*(1+std::abs(oldMomentum)),"internal equality created angular momentum");++checks;
        if(!loaded) {require(newEnergy<=oldEnergy+1e-5,"unforced equality projection created energy");++checks;}''','''        require(std::abs(newMomentum-oldMomentum-expectedNormal[e])<1e-5*(1+std::abs(oldMomentum)),
                "constraint solve violated angular impulse balance");++checks;
        oldEnergy+=2.0*oldLinear[e]*oldLinear[e];
        newEnergy+=2.0*double(v[e*nv+1])*v[e*nv+1];
        if(!loaded) {require(newEnergy<=oldEnergy+1e-5,"unforced equality projection created energy");++checks;}
        if(nc != 0u) {
            require(std::abs(v[e*nv+1]-expectedLinear[e])<2e-5,
                    "contact solve differs from reduced linear momentum oracle");++checks;
            const unsigned contactJoint=contactBody==0u ? MR_INVALID_INDEX : 5u+contactBody;
            const double normalVelocity=v[e*nv+1]+v[e*nv+5]+(contactJoint==MR_INVALID_INDEX?0:v[e*nv+contactJoint]);
            require(normalVelocity>=-2e-6,"equality correction invalidated unilateral contact");++checks;
            require(std::abs(4.0*(v[e*nv+1]-oldLinear[e])-expectedNormal[e])<1e-5,
                    "contact solve violated linear impulse balance");++checks;
            require(expectedNormal[e]==0.0 || std::abs(normalVelocity)<2e-6,
                    "loaded contact has nonzero terminal normal velocity");++checks;
        }''',1)
needle='''            std::cout<<"Human production bilateral block: "<<checks'''
assert s.count(needle)==1
s=s.replace(needle,'''            for(float h:{1e-4f,5e-5f,1.25e-5f})for(unsigned sweeps:{1u,4u,64u})
                for(bool reverse:{false,true})for(bool loaded:{false,true})
                    for(unsigned contactBody:{0u,1u,2u,3u})for(float seed:{0.0f,20.0f})
                        for(bool closing:{false,true})
                            checks+=exercise(device,pipeline,queue,h,sweeps,reverse,loaded,
                                MR_INVALID_INDEX,contactBody,seed,closing);
'''+needle,1)
p.write_text(s)
for name, (_, after) in hashes.items():
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    assert actual == after, (name, actual, after)
print('Five exact contact/equality source files verified')
