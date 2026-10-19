#include "NumiHumanVesselWallCoupling.hpp"
#include "metalrobo/engine_types.h"
#include "numi/matter/matter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace numi::human::aortic_wall;
using namespace numi::matter;

static void require(bool value,const std::string& message) {
    if(!value) throw std::runtime_error(message);
}

template<class T> static bool byteEqual(const std::vector<T>& a,const std::vector<T>& b) {
    return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);
}

int main() {
    try {
        SectionProof proof;
        proof.stableId=9u; proof.sourceBody=7u;
        proof.sourcePayloadSHA256="1a191fe2f6c4a1cb03ee290bc850b80011cc267577992ea86abb3a4cb39f4108";
        proof.planePoint={-0.03372473683003,0.05810004997253,-0.00946755672337};
        proof.planeNormal={0.0,1.0,0.0}; proof.externalAreaM2=2.4192601236904974e-4;
        #include "actual_nha_contour.inc"
        Config cfg; cfg.circumferentialSegments=24u; cfg.axialSegments=8u;
        cfg.outerLateralContact=true;
        MRBodyStateGPU identity{}; identity.orientation={0.0f,0.0f,0.0f,1.0f};
        Built wall=build(proof,identity,{0.0,0.0,0.0},7u,identity,7u,1.84183e-5,11223.5,cfg);
        wall.objectIndex=0u;
        require(wall.object.femContactNodes.size()==216u,"wall source outer query mask count mismatch");

        WorldSource source; source.frameTimestep=1.0/4000.0; source.environmentCount=1u;
        source.gravity={0.0,0.0,0.0}; source.compactImmutableFEMSurfaces=true;
        source.materials.push_back(makeMaterial(cfg));
        wall.object.materialIndex=0u;
        require(wall.object.femDisabledContactSurfaceFaces.size()==672u,
                "wall did not author its contact-face selection before cooking");
        source.objects.push_back(wall.object);
        ObjectSource other; other.name="probe_other_target";
        other.representation=Representation::fem; other.automaticRepresentation=false;
        other.mixedFEM=false; other.materialIndex=0u; other.deformableContact=true;
        other.femNodes={{{1.0,0.0,0.0}},{{1.0,1.0,0.0}},{{1.0,0.0,1.0}},{{2.0,0.0,0.0}}};
        other.tetrahedra.push_back({{0u,1u,2u,3u}});
        source.objects.push_back(other);
        CompileOptions options; options.maximumRateExponent=0u;
        auto compiled=compileWorld(source,options);
        std::string diagnostic;
        for(const auto& d:compiled.diagnostics) diagnostic+=d.message+"; ";
        require(compiled.succeeded(),"actual Matter compileWorld failed: "+diagnostic);
        require((compiled.world.dispatch.flags&NM_MATTER_STATIC_FEM_SURFACE_MAP)!=0u,
                "compiler did not emit static immutable exterior map");
        require(compiled.world.objects.size()==2u,"expected wall and separate target objects");
        const auto beforeNodes=compiled.world.fem.nodes;
        const auto beforeTets=compiled.world.fem.tetrahedra;
        const auto beforeFaces=compiled.world.fem.surfaceFaces;
        std::vector<NMFEMSurfaceFaceGPU> otherFacesBefore;
        for(const auto& face:beforeFaces) if(face.adjacency.z==1u) otherFacesBefore.push_back(face);
        require(otherFacesBefore.size()==4u,"probe target tetrahedron did not contribute four mapped exterior faces");
        const auto sourceNodesBefore=wall.object.femNodes;
        const auto sourceReferenceBefore=wall.object.femReferenceNodes;
        const auto sourceTetsBefore=wall.object.tetrahedra;
        const auto cavityFacesBefore=wall.cavity.faces;
        double massBefore=0.0;
        for(std::uint32_t i=0;i<compiled.world.objects[0].stateCount;i++)
            massBefore+=compiled.world.fem.nodes[compiled.world.objects[0].stateOffset+i].positionAndMass.w;

        const auto audit=numi::human::vessel_wall_coupling::auditOuterLateralContactSurfaceFaces(
            compiled.world,wall,wall.object.femContactNodes);
        std::vector<NMFEMSurfaceFaceGPU> otherFacesAfter;
        for(const auto& face:compiled.world.fem.surfaceFaces) if(face.adjacency.z==1u) otherFacesAfter.push_back(face);
        auto normalizedFaces=compiled.world.fem.surfaceFaces;
        std::vector<std::uint8_t> outerMask(wall.object.femNodes.size(),0u);
        for(const auto node:wall.object.femContactNodes) outerMask[node]=1u;
        std::uint32_t wallBoundaryCount=0u,wallOuterFaceCount=0u,wallDisabledFaceCount=0u;
        for(auto& face:normalizedFaces) {
            if(face.adjacency.z!=0u) continue;
            ++wallBoundaryCount;
            const auto& tet=compiled.world.fem.tetrahedra[face.adjacency.x];
            const std::array<std::uint32_t,4u> nodes{tet.nodes.x,tet.nodes.y,tet.nodes.z,tet.nodes.w};
            bool outer=true;
            for(unsigned c=0u;c<4u;++c) if(c!=face.sides.x) {
                require(nodes[c]>=compiled.world.objects[0].stateOffset&&
                        nodes[c]<compiled.world.objects[0].stateOffset+compiled.world.objects[0].stateCount,
                        "wall source face contains a node outside its object range");
                outer=outer&&outerMask[nodes[c]-compiled.world.objects[0].stateOffset]!=0u;
            }
            const auto expected=outer?0u:NM_FEM_SURFACE_DISABLE_DEFORMABLE_CONTACT;
            require(face.sides.w==expected,"source face contact flag does not match the outer-lateral selector");
            if(outer) ++wallOuterFaceCount;
            else ++wallDisabledFaceCount;

        }
        double massAfter=0.0;
        for(std::uint32_t i=0;i<compiled.world.objects[0].stateCount;i++)
            massAfter+=compiled.world.fem.nodes[compiled.world.objects[0].stateOffset+i].positionAndMass.w;
        require(audit.queryNodeCount==216u&&audit.retainedWallFaceCount==384u,
                "actual compiled wall mask/count is not 216 nodes and 384 outer faces");
        require(audit.excludedWallFaceCount==672u&&wallBoundaryCount==1056u&&
                wallOuterFaceCount==384u&&wallDisabledFaceCount==672u,
                "actual wall boundary selection counts differ from 384 outer / 672 nonouter of 1056");
        require(byteEqual(beforeNodes,compiled.world.fem.nodes),"wall node/current mass state changed during contact-face marking");
        require(byteEqual(beforeTets,compiled.world.fem.tetrahedra),"wall tetrahedra/reference operators changed during contact-face marking");
        require(byteEqual(otherFacesBefore,otherFacesAfter),"nonwall target face records were not preserved byte-for-byte");
        require(byteEqual(beforeFaces,normalizedFaces),"complete canonical face map changed beyond the single approved wall contact flag");
        require(byteEqual(wall.object.femNodes,sourceNodesBefore)&&
                byteEqual(wall.object.femReferenceNodes,sourceReferenceBefore)&&
                byteEqual(wall.object.tetrahedra,sourceTetsBefore)&&
                byteEqual(wall.cavity.faces,cavityFacesBefore),
                "authored wall current/reference/tet/cavity source changed during contact-face marking");
        require(massBefore==massAfter,"wall cooked nodal mass sum changed during filtering");
        require(compiled.world.fem.surfaceFaces.size()==audit.originalSurfaceFaceCount&&
                audit.canonicalSurfaceFaceCount==audit.originalSurfaceFaceCount,
                "canonical map row count changed while marking contact faces");
        // The live scene copies and recooks the authored source after adding
        // physiology. No post-cook mutation may be needed to restore eligibility.
        const WorldSource continuationSource=source;
        auto recooked=compileWorld(continuationSource,options);
        require(recooked.succeeded(),"authored source failed its independent recook");
        const auto recookAudit=numi::human::vessel_wall_coupling::auditOuterLateralContactSurfaceFaces(
            recooked.world,wall,wall.object.femContactNodes);
        require(byteEqual(compiled.world.fem.surfaceFaces,recooked.world.fem.surfaceFaces),
                "contact-face selection was lost during source recooking");
        require(compiled.world.fingerprint==recooked.world.fingerprint &&
                recookAudit.excludedWallFaceCount==672u,
                "source recook changed contact ownership or fingerprint");
        auto malformed=source;
        malformed.objects[0].femDisabledContactSurfaceFaces.push_back(
            malformed.objects[0].femDisabledContactSurfaceFaces.back());
        require(!compileWorld(malformed,options).succeeded(),
                "compiler accepted duplicate authored face identities");
        malformed=source;
        malformed.objects[0].femDisabledContactSurfaceFaces={0xffffffffu};
        require(!compileWorld(malformed,options).succeeded(),
                "compiler accepted out-of-range authored contact face");
        malformed=source;malformed.compactImmutableFEMSurfaces=false;
        require(!compileWorld(malformed,options).succeeded(),
                "compiler silently ignored a face selector without its map owner");
        auto invalid=compiled.world;
        const auto invalidFace=std::find_if(invalid.fem.surfaceFaces.begin(),invalid.fem.surfaceFaces.end(),
            [](const auto& face){return face.adjacency.z==1u;});
        require(invalidFace!=invalid.fem.surfaceFaces.end(),"probe has no nonwall face for reserved-bit test");
        invalidFace->sides.w=2u;
        invalid.fingerprint=compiledWorldFingerprint(invalid);
        std::string invalidDiagnostic;
        require(!validateCompiledWorldLayout(invalid,&invalidDiagnostic),
                "canonical validator accepted an unknown contact-face flag");
        std::cout<<"{"<<"\"status\":\"PASS\",\"query_nodes\":"<<audit.queryNodeCount
            <<",\"original_surface_faces\":"<<audit.originalSurfaceFaceCount
            <<",\"canonical_surface_faces_after_flags\":"<<audit.canonicalSurfaceFaceCount
            <<",\"retained_wall_outer_faces\":"<<audit.retainedWallFaceCount
            <<",\"excluded_wall_nonouter_faces\":"<<audit.excludedWallFaceCount
            <<",\"preserved_nonwall_face_rows\":"<<otherFacesAfter.size()
            <<",\"wall_tets\":"<<wall.object.tetrahedra.size()
            <<",\"wall_nodes\":"<<wall.object.femNodes.size()
            <<",\"wall_mass_kg\":"<<massAfter
            <<",\"physics_fingerprint\":"<<compiled.world.physicsFingerprint
            <<",\"package_fingerprint\":"<<compiled.world.fingerprint<<"}\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"outer_contact_map_probe=FAIL reason="<<e.what()<<"\n";
        return 1;
    }
}
