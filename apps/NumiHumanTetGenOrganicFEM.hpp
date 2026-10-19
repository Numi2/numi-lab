#pragma once
// In-memory adapter from an offline, source-audited TetGen PLC result to
// Matter's existing ObjectSource. This does not define a serialized format,
// a new mesher, a constitutive law, or a runtime solver.
#include <NumiHumanCoarseTissuePreparation.hpp>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace metalrobo::human::coarse_tissue {

struct RegisteredTetMaterial {
    std::uint32_t materialIndex=NM_INVALID_INDEX;
    std::string materialName, provenance;
};
struct RegisteredTetrahedralVolume {
    std::string name, provider, hierarchy, sourceMember;
    std::string anatomicalConceptIdentifier; // e.g. FMA50878; never cast to an invented uint ID.
    std::string sourceMeshSHA256, sourcePayloadSHA256;
    std::string geometryProvenance, intersectionAuditProvenance;
    std::string referenceConfigurationProvenance;
    // Caller must bind this to the exact source registration/body-pose transform
    // and check every world/reference/donor-local point through that transform.
    std::string frameTransformProvenance;
    std::string boundaryReductionProvenance;
    std::uint32_t stableIdentifier=0, sourceBodyIndex=NM_INVALID_INDEX, sourceLayer=0;
    std::uint32_t sourceVertexCount=0, sourceFaceCount=0;
    bool aggregateIdentity=false;
    std::array<std::uint64_t,4> materialSourceIdentity{};
    std::array<std::uint64_t,4> referenceSourceIdentity{};
    // TetGen output points in current world, supplied reference, and original
    // donor-local coordinates. UINT32_MAX in nodeSourceVertex means a mesher
    // Steiner node (or an explicitly documented reduced-boundary vertex).
    std::vector<V3> worldPoints, referenceWorldPoints, donorLocalPoints;
    std::vector<std::uint32_t> donorBodyIndices, nodeSourceVertex;
    std::vector<std::array<std::uint32_t,4>> tetrahedra;
    // Interior TetGen cells normally have no single source triangle parent;
    // UINT32_MAX states that explicitly. Boundary faces retain per-face source
    // triangle ancestry, including many-to-one ancestry after a documented
    // surface reduction or refinement.
    std::vector<std::uint32_t> tetrahedronSourceFaceIndices;
    std::vector<Face> boundaryFaces;
    std::vector<std::uint32_t> boundarySourceFaceIndices;
    std::vector<RegisteredTetMaterial> materials;
    std::vector<std::uint32_t> tetrahedronMaterialBindingIndices;
};
struct PreparedTetrahedralAncestry {
    std::string anatomicalConceptIdentifier, hierarchy, geometryProvenance;
    std::string intersectionAuditProvenance, referenceConfigurationProvenance, frameTransformProvenance;
    std::vector<Face> boundaryFaces;
    std::vector<std::uint32_t> boundarySourceFaceIndices;
    std::vector<std::uint32_t> nodeSourceVertexIndices;
};

inline bool appendRegisteredTetrahedralFEM(numi::matter::WorldSource& world,
        const RegisteredTetrahedralVolume& source,
        std::string_view expectedName,std::string_view expectedMember,
        std::string_view expectedAnatomicalConcept,std::uint32_t expectedStableIdentifier,
        std::uint32_t expectedSourceLayer,std::string objectName,
        double existingDonorMassKg,PreparedOrganicFEM& prepared,
        PreparedTetrahedralAncestry& ancestry,std::string& error,
        std::uint32_t maximumTetrahedra=250000u) {
    auto fail=[&](const char* message){error=message;return false;};
    auto hashOK=[](const std::string& h){return h.size()==64&&std::all_of(h.begin(),h.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});};
    auto identityOK=[](const std::array<std::uint64_t,4>& x){return std::any_of(x.begin(),x.end(),[](auto v){return v!=0u;});};
    const std::size_t n=source.worldPoints.size(),nt=source.tetrahedra.size(),nb=source.boundaryFaces.size();
    if(source.name!=expectedName||source.sourceMember!=expectedMember||source.anatomicalConceptIdentifier!=expectedAnatomicalConcept||source.stableIdentifier!=expectedStableIdentifier||source.sourceLayer!=expectedSourceLayer)
        return fail("registered tet source differs from the explicit typed anatomical identity");
    if(source.name.empty()||source.provider.empty()||source.hierarchy.empty()||source.sourceMember.empty()||
       (source.stableIdentifier==0&&source.anatomicalConceptIdentifier.empty()&&!source.aggregateIdentity)||
       !hashOK(source.sourceMeshSHA256)||!hashOK(source.sourcePayloadSHA256)||
       source.geometryProvenance.empty()||source.intersectionAuditProvenance.empty()||source.referenceConfigurationProvenance.empty()||source.frameTransformProvenance.empty()||source.sourceVertexCount==0||source.sourceFaceCount==0)
        return fail("registered tet source identity, hashes, or geometry audit provenance is incomplete");
    if(!identityOK(source.materialSourceIdentity)||!identityOK(source.referenceSourceIdentity))
        return fail("regional material and reference-coordinate source identities must be explicit");
    if(n<4||nt==0||nt>maximumTetrahedra||nb<4||
       source.referenceWorldPoints.size()!=n||source.donorLocalPoints.size()!=n||source.nodeSourceVertex.size()!=n||
       source.boundarySourceFaceIndices.size()!=nb||
       (!source.tetrahedronSourceFaceIndices.empty()&&source.tetrahedronSourceFaceIndices.size()!=nt)||
       (!source.donorBodyIndices.empty()&&source.donorBodyIndices.size()!=n)||
       source.materials.empty()||(!source.tetrahedronMaterialBindingIndices.empty()&&source.tetrahedronMaterialBindingIndices.size()!=nt))
        return fail("registered tet volume arrays have inconsistent sizes or exceed the coarse mesh budget");
    if(source.sourceBodyIndex==NM_INVALID_INDEX&&source.donorBodyIndices.size()!=n)
        return fail("tet volume needs one registered donor body or a complete per-node donor map");
    if(!std::isfinite(existingDonorMassKg)||existingDonorMassKg<=0.0||objectName.empty())
        return fail("existing Human donor mass debit and Matter object name must be positive and explicit");
    if(std::any_of(source.nodeSourceVertex.begin(),source.nodeSourceVertex.end(),[&](auto v){return v!=NM_INVALID_INDEX&&v>=source.sourceVertexCount;}))return fail("node source-vertex ancestry is out of range");
    if(!source.tetrahedronSourceFaceIndices.empty()&&std::any_of(source.tetrahedronSourceFaceIndices.begin(),source.tetrahedronSourceFaceIndices.end(),[&](auto f){return f!=NM_INVALID_INDEX&&f>=source.sourceFaceCount;}))return fail("tetrahedron source-face ancestry is out of range");
    if(source.boundaryReductionProvenance.empty()){
        if(std::any_of(source.boundarySourceFaceIndices.begin(),source.boundarySourceFaceIndices.end(),[&](auto f){return f==NM_INVALID_INDEX||f>=source.sourceFaceCount;}))return fail("unreduced boundary faces must retain valid source-face ancestry");
        for(const auto& f:source.boundaryFaces)for(auto node:f)if(node>=n||source.nodeSourceVertex[node]==NM_INVALID_INDEX)
            return fail("boundary Steiner vertices require explicit source-boundary reduction provenance");
    } else if(std::any_of(source.boundarySourceFaceIndices.begin(),source.boundarySourceFaceIndices.end(),[&](auto f){return f!=NM_INVALID_INDEX&&f>=source.sourceFaceCount;}))return fail("reduced boundary ancestry points outside the registered source mesh");

    std::vector<std::uint32_t> materialForTet(nt),donorForNode(n);
    std::vector<double> density(source.materials.size());
    for(std::size_t bi=0;bi<source.materials.size();bi++){
        const auto& binding=source.materials[bi];
        if(binding.materialIndex>=world.materials.size()||binding.materialName.empty()||binding.provenance.empty())return fail("each tet material requires an existing indexed Matter law and provenance");
        const auto& material=world.materials[binding.materialIndex];
        if(material.name!=binding.materialName||std::find(material.supportedRepresentations.begin(),material.supportedRepresentations.end(),numi::matter::Representation::fem)==material.supportedRepresentations.end())return fail("existing Matter material identity or FEM support differs from the explicit assignment");
        auto it=std::find_if(material.parameters.begin(),material.parameters.end(),[](const auto& p){return p.name=="density";});
        if(it==material.parameters.end()||it->identifiable||!std::isfinite(it->defaultValue)||it->defaultValue<=0||it->defaultValue<it->lower||it->defaultValue>it->upper)return fail("each passive tissue material needs a fixed finite density within its existing range");
        density[bi]=it->defaultValue;
        if(bi>0){const auto& owner=world.materials[source.materials.front().materialIndex];if(material.staticFriction!=owner.staticFriction||material.dynamicFriction!=owner.dynamicFriction||material.restitution!=owner.restitution||material.adhesion!=owner.adhesion)return fail("regional material interface response must match the uniform external-contact owner");}
    }
    for(std::size_t ti=0;ti<nt;ti++){
        const auto bi=source.tetrahedronMaterialBindingIndices.empty()?0u:source.tetrahedronMaterialBindingIndices[ti];
        if(bi>=source.materials.size())return fail("tetrahedron material binding index is out of range");materialForTet[ti]=bi;
    }
    for(std::size_t i=0;i<n;i++){
        if(!std::all_of(source.worldPoints[i].begin(),source.worldPoints[i].end(),[](double x){return std::isfinite(x);})||
           !std::all_of(source.referenceWorldPoints[i].begin(),source.referenceWorldPoints[i].end(),[](double x){return std::isfinite(x);})||
           !std::all_of(source.donorLocalPoints[i].begin(),source.donorLocalPoints[i].end(),[](double x){return std::isfinite(x);}))return fail("registered tet node coordinates must be finite in every frame");
        donorForNode[i]=source.donorBodyIndices.empty()?source.sourceBodyIndex:source.donorBodyIndices[i];
        if(donorForNode[i]==NM_INVALID_INDEX)return fail("registered tet node has no existing donor-body identity");
    }

    using FaceKey=Face;
    auto key3=[](Face f){std::sort(f.begin(),f.end());return f;};
    std::set<FaceKey> expectedBoundary;
    std::map<FaceKey,int> expectedOrientation;
    auto faceOrientation=[](Face f){unsigned inversions=0;for(unsigned i=0;i<3;i++)for(unsigned j=i+1;j<3;j++)if(f[i]>f[j])inversions++;return (inversions%2)?-1:1;};
    std::map<std::array<std::uint32_t,2>,std::pair<unsigned,int>> sourceEdges;
    for(std::size_t i=0;i<nb;i++){
        const auto f=source.boundaryFaces[i];if(f[0]>=n||f[1]>=n||f[2]>=n||f[0]==f[1]||f[1]==f[2]||f[2]==f[0])return fail("tet boundary face index is invalid");
        if(!expectedBoundary.insert(key3(f)).second)return fail("tet boundary contains duplicate triangles");expectedOrientation[key3(f)]=faceOrientation(f);
        for(unsigned e=0;e<3;e++){auto a=f[e],b=f[(e+1)%3];std::array<std::uint32_t,2> k{std::min(a,b),std::max(a,b)};auto& x=sourceEdges[k];x.first++;x.second+=(a<b)?1:-1;}
    }
    if(std::any_of(sourceEdges.begin(),sourceEdges.end(),[](const auto& x){return x.second.first!=2||x.second.second!=0;}))return fail("registered tet boundary is not a closed consistently oriented manifold");

    struct FaceUse { unsigned count=0; int orientation=0; };
    std::map<FaceKey,FaceUse> faceUse;std::vector<double> nodalMass(n);double currentTetVolume=0.0,currentFloat32TetVolume=0.0,referenceTetVolume=0.0,floatReferenceTetVolume=0.0,minTetVolume=std::numeric_limits<double>::infinity();
    auto f32=[](V3 p){return V3{static_cast<float>(p[0]),static_cast<float>(p[1]),static_cast<float>(p[2])};};
    std::set<std::array<std::uint32_t,4>> uniqueTets;
    for(std::size_t ti=0;ti<nt;ti++){
        const auto t=source.tetrahedra[ti];auto canonicalTet=t;std::sort(canonicalTet.begin(),canonicalTet.end());if(!uniqueTets.insert(canonicalTet).second)return fail("registered tet complex contains duplicate tetrahedra");std::set<std::uint32_t> unique;
        for(auto node:t)if(node>=n||!unique.insert(node).second)return fail("tet element contains an invalid or repeated node");
        auto signedV=tetVolume(source.worldPoints[t[0]],source.worldPoints[t[1]],source.worldPoints[t[2]],source.worldPoints[t[3]]);
        auto refV=tetVolume(source.referenceWorldPoints[t[0]],source.referenceWorldPoints[t[1]],source.referenceWorldPoints[t[2]],source.referenceWorldPoints[t[3]]);
        const auto a=f32(source.referenceWorldPoints[t[0]]),b=f32(source.referenceWorldPoints[t[1]]),c=f32(source.referenceWorldPoints[t[2]]),d=f32(source.referenceWorldPoints[t[3]]);
        const auto refV32=tetVolume(a,b,c,d);
        const auto wa=f32(source.worldPoints[t[0]]),wb=f32(source.worldPoints[t[1]]),wc=f32(source.worldPoints[t[2]]),wd=f32(source.worldPoints[t[3]]);
        const auto currentV32=tetVolume(wa,wb,wc,wd);
        if(!std::isfinite(signedV)||signedV<=0||!std::isfinite(refV)||refV<=0||!std::isfinite(refV32)||refV32<=0||!std::isfinite(currentV32)||currentV32<=0)return fail("tet element must have finite positive signed volume in double and runtime Float32 current and reference coordinates");
        for(const auto& p:{a,b,c,d})if(!std::all_of(p.begin(),p.end(),[](double x){return std::isfinite(x);}))return fail("tet element overflows runtime Float32 coordinates");
        for(unsigned omit=0;omit<4;omit++){
            Face face{};unsigned k=0;for(unsigned j=0;j<4;j++)if(j!=omit)face[k++]=t[j];auto& use=faceUse[key3(face)];use.count++;use.orientation+=((omit%2)?-1:1)*faceOrientation(face);
        }
        for(unsigned j=1;j<4;j++)if(donorForNode[t[j]]!=donorForNode[t[0]])return fail("one tet crosses donor bodies; partition the volume before nodal mass assignment");
        const double rho=density[materialForTet[ti]];for(auto node:t)nodalMass[node]+=rho*refV/4.0;
        currentTetVolume+=signedV;currentFloat32TetVolume+=currentV32;referenceTetVolume+=refV;floatReferenceTetVolume+=refV32;minTetVolume=std::min(minTetVolume,refV);
    }
    std::set<FaceKey> actualBoundary;
    std::map<std::array<std::uint32_t,2>,unsigned> boundaryEdgeUse;
    for(const auto& [face,use]:faceUse){
        if(use.count==1){actualBoundary.insert(face);if(!expectedOrientation.contains(face)||use.orientation!=expectedOrientation[face])return fail("tet boundary orientation differs from its source boundary");for(unsigned e=0;e<3;e++){auto a=face[e],b=face[(e+1)%3];boundaryEdgeUse[{std::min(a,b),std::max(a,b)}]++;}}
        else if(use.count==2){if(use.orientation!=0)return fail("shared tetrahedron face orientation is inconsistent");}
        else return fail("tetrahedral complex contains a nonmanifold face");
    }
    if(actualBoundary!=expectedBoundary||std::any_of(boundaryEdgeUse.begin(),boundaryEdgeUse.end(),[](const auto& e){return e.second!=2;}))return fail("tetrahedral boundary differs from the source-marked closed boundary");

    auto signedBoundaryVolume=[&](const std::vector<V3>& points){V3 origin{};for(const auto& f:source.boundaryFaces)for(auto i:f)origin=add(origin,mul(points[i],1.0/(3.0*nb)));double volume=0;for(const auto& f:source.boundaryFaces){auto a=points[f[0]],b=points[f[1]],c=points[f[2]];volume+=dot(sub(a,origin),cross(sub(b,origin),sub(c,origin)))/6.0;}return volume;};
    std::vector<V3> currentFloat32Points;currentFloat32Points.reserve(n);for(const auto& p:source.worldPoints)currentFloat32Points.push_back(f32(p));
    const double currentBoundaryVolume=signedBoundaryVolume(source.worldPoints),currentFloat32BoundaryVolume=signedBoundaryVolume(currentFloat32Points),referenceBoundaryVolume=signedBoundaryVolume(source.referenceWorldPoints);
    auto close=[](double a,double b,double r){return std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=r*std::max(std::abs(a),std::abs(b));};
    if(!(currentBoundaryVolume>0&&currentFloat32BoundaryVolume>0&&referenceBoundaryVolume>0)||!close(currentTetVolume,currentBoundaryVolume,1e-6)||!close(currentFloat32TetVolume,currentFloat32BoundaryVolume,1e-5)||!close(referenceTetVolume,referenceBoundaryVolume,1e-6)||!close(floatReferenceTetVolume,referenceBoundaryVolume,1e-5))return fail("TetGen current/reference/Float32 volumes fail source-boundary closure");
    double integratedMass=0;for(auto m:nodalMass)integratedMass+=m;
    if(!(referenceTetVolume>0&&integratedMass>0)||!close(integratedMass,existingDonorMassKg,1e-7))return fail("fixed material densities do not close to the existing donor-mass debit");

    std::uint32_t first=0;for(const auto& old:world.objects){if(old.femNodes.size()>std::numeric_limits<std::uint32_t>::max()-first)return fail("global FEM node index overflow");first+=static_cast<std::uint32_t>(old.femNodes.size());}if(n>std::numeric_limits<std::uint32_t>::max()-first)return fail("appended FEM node range overflows global uint32 indices");
    numi::matter::ObjectSource object;object.name=objectName;object.materialIndex=source.materials.front().materialIndex;object.representation=numi::matter::Representation::fem;object.twoWayCoupling=true;object.adaptive=false;object.mixedFEM=false;object.characteristicLength=std::cbrt(referenceBoundaryVolume);
    object.femNodes=source.worldPoints;object.femReferenceNodes=source.referenceWorldPoints;object.femReferenceSourceIdentity=source.referenceSourceIdentity;object.femReferenceDisplacementGradient=true;
    object.tetrahedra.reserve(nt);object.femMaterialIndices.reserve(nt);object.femMaterialSourceIdentity=source.materialSourceIdentity;
    for(std::size_t ti=0;ti<nt;ti++){object.tetrahedra.push_back({source.tetrahedra[ti]});object.femMaterialIndices.push_back(source.materials[materialForTet[ti]].materialIndex);}
    std::set<std::uint32_t> contactNodes;for(const auto& f:source.boundaryFaces)for(auto node:f)contactNodes.insert(node);object.femContactNodes.assign(contactNodes.begin(),contactNodes.end());
    if(object.femContactNodes.empty()||object.femReferenceNodes.size()!=object.femNodes.size()||object.femMaterialIndices.size()!=nt)return fail("ObjectSource FEM reference, material, or external-contact boundary is incomplete");

    PreparedOrganicFEM result;result.sourceName=source.name;result.sourceMember=source.sourceMember;result.meshSHA256=source.sourceMeshSHA256;result.payloadSHA256=source.sourcePayloadSHA256;result.sourceStableIdentifier=source.stableIdentifier;result.sourceBodyIndex=source.sourceBodyIndex;result.aggregateIdentity=source.aggregateIdentity;result.materialIndex=object.materialIndex;result.materialName.clear();result.materialProvenance.clear();
    for(std::size_t bi=0;bi<source.materials.size();bi++){if(bi){result.materialName+=" + ";result.materialProvenance+="; ";}result.materialName+=source.materials[bi].materialName;result.materialProvenance+=source.materials[bi].provenance;}
    result.firstGlobalFEMNode=first;result.volumeM3=referenceBoundaryVolume;result.massDebitKg=existingDonorMassKg;result.densityKgM3=integratedMass/referenceTetVolume;result.massResidualKg=integratedMass-existingDonorMassKg;result.audit.volumeM3=referenceBoundaryVolume;result.audit.minFaceAreaM2=std::numeric_limits<double>::infinity();
    result.nodeSourceVertex=source.nodeSourceVertex;result.tetSourceFace=source.tetrahedronSourceFaceIndices.empty()?std::vector<std::uint32_t>(nt,NM_INVALID_INDEX):source.tetrahedronSourceFaceIndices;
    for(const auto& face:source.boundaryFaces){const auto a=source.referenceWorldPoints[face[0]],b=source.referenceWorldPoints[face[1]],c=source.referenceWorldPoints[face[2]];result.audit.minFaceAreaM2=std::min(result.audit.minFaceAreaM2,norm(cross(sub(b,a),sub(c,a)))*0.5);}
    result.donorMassNodes.reserve(n);for(std::uint32_t i=0;i<n;i++){if(!std::isfinite(nodalMass[i])||nodalMass[i]<=0)return fail("tet mesh contains a node without positive lumped donor mass");result.donorMassNodes.push_back({.nodeIndex=first+i,.donorBody=donorForNode[i],.massKg=nodalMass[i],.localPosition=source.donorLocalPoints[i]});}
    ancestry.anatomicalConceptIdentifier=source.anatomicalConceptIdentifier;ancestry.hierarchy=source.hierarchy;ancestry.geometryProvenance=source.geometryProvenance;ancestry.intersectionAuditProvenance=source.intersectionAuditProvenance;ancestry.referenceConfigurationProvenance=source.referenceConfigurationProvenance;ancestry.frameTransformProvenance=source.frameTransformProvenance;ancestry.boundaryFaces=source.boundaryFaces;ancestry.boundarySourceFaceIndices=source.boundarySourceFaceIndices;ancestry.nodeSourceVertexIndices=source.nodeSourceVertex;
    prepared=std::move(result);world.objects.push_back(std::move(object));error.clear();return true;
}

} // namespace metalrobo::human::coarse_tissue
