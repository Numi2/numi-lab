#include "NumiHumanSkinFEMIntegration.wholebody.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <set>
#include <string>
#include <vector>

namespace ct = metalrobo::human::coarse_tissue;
using ct::Face;
using ct::V3;

namespace {
template <typename T>
std::vector<T> copyArray(const void* p, std::size_t n) {
    const auto* first = static_cast<const T*>(p);
    return std::vector<T>(first, first + n);
}
std::vector<V3> copyPoints(const double* p, std::size_t n) {
    std::vector<V3> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = {p[3*i], p[3*i+1], p[3*i+2]};
    return out;
}
std::vector<Face> copyFaces(const std::uint32_t* p, std::size_t n) {
    std::vector<Face> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = {p[3*i], p[3*i+1], p[3*i+2]};
    return out;
}
numi::matter::MaterialProgram testMaterial(std::string name, double lo, double hi) {
    numi::matter::MaterialProgram m;
    m.name = std::move(name);
    m.supportedRepresentations.push_back(numi::matter::Representation::fem);
    numi::matter::Parameter p;
    p.name = "density"; p.lower = lo; p.upper = hi; p.defaultValue = (lo+hi)*0.5;
    m.parameters.push_back(p);
    return m;
}
void setOutput(char* dst, std::size_t cap, const std::string& text) {
    if (!dst || cap == 0) return;
    const auto n = std::min(cap-1, text.size());
    std::memcpy(dst, text.data(), n);
    dst[n] = '\0';
}
}

extern "C" int run081Candidate(const void* const* a, const std::size_t* dims,
        const char* meshHash, const char* payloadHash, const double* debit,
        char* output, std::size_t outputCapacity) {
    std::ostringstream log;
    log << std::setprecision(17);
    try {
        if (!a || !dims || !meshHash || !payloadHash || !debit) {
            setOutput(output, outputCapacity, "status=INPUT_ERROR missing_pointer\n"); return 3;
        }
        const std::size_t n=dims[0], nf=dims[1], ns=dims[2], nsf=dims[3];
        if (n<4 || nf<4 || ns==0 || nsf==0) {
            setOutput(output, outputCapacity, "status=INPUT_ERROR invalid_dimensions\n"); return 3;
        }
        const auto* outerRaw=static_cast<const double*>(a[0]);
        const auto* innerRaw=static_cast<const double*>(a[1]);
        const auto* deepRaw=static_cast<const double*>(a[2]);
        const auto* faceRaw=static_cast<const std::uint32_t*>(a[3]);
        const auto* parentRaw=static_cast<const std::uint32_t*>(a[4]);
        const auto* idRaw=static_cast<const std::int64_t*>(a[5]);
        const auto* sourceFaceRaw=static_cast<const std::uint32_t*>(a[6]);
        const auto* betaRaw=static_cast<const double*>(a[7]);
        const auto* kindRaw=static_cast<const std::uint8_t*>(a[8]);
        const auto* residualRaw=static_cast<const double*>(a[9]);
        const auto* skinRaw=static_cast<const double*>(a[10]);
        const auto* fatRaw=static_cast<const double*>(a[11]);
        const auto* skinLoRaw=static_cast<const double*>(a[12]);
        const auto* skinHiRaw=static_cast<const double*>(a[13]);
        const auto* fatLoRaw=static_cast<const double*>(a[14]);
        const auto* fatHiRaw=static_cast<const double*>(a[15]);
        const auto* sourceWorldRaw=static_cast<const double*>(a[16]);
        const auto* sourceFaceGeometryRaw=static_cast<const std::uint32_t*>(a[17]);
        const auto* weightsRaw=static_cast<const float*>(a[18]);
        const auto* slotDonorRaw=static_cast<const std::uint32_t*>(a[19]);
        const auto* slotOriginRaw=static_cast<const double*>(a[20]);
        const auto* slotMatrixRaw=static_cast<const double*>(a[21]);

        auto outer=copyPoints(outerRaw,n), inner=copyPoints(innerRaw,n), deep=copyPoints(deepRaw,n);
        auto faces=copyFaces(faceRaw,nf);
        auto parents=copyArray<std::uint32_t>(parentRaw,nf);
        auto sourceIds=copyArray<std::int64_t>(idRaw,n);
        auto sourceFaceIds=copyArray<std::uint32_t>(sourceFaceRaw,n);
        std::vector<std::array<double,3>> beta(n);
        std::vector<std::uint8_t> kinds=copyArray<std::uint8_t>(kindRaw,n);
        std::vector<V3> residual=copyPoints(residualRaw,n);
        auto sourceWorld=copyPoints(sourceWorldRaw,ns);
        auto sourceFaces=copyFaces(sourceFaceGeometryRaw,nsf);
        auto weights=copyArray<float>(weightsRaw,ns*86u);
        auto slotDonor=copyArray<std::uint32_t>(slotDonorRaw,86u);
        auto slotOrigin=copyPoints(slotOriginRaw,86u);
        auto slotMatrix=copyArray<double>(slotMatrixRaw,86u*9u);
        auto skin=copyArray<double>(skinRaw,n), fat=copyArray<double>(fatRaw,n);
        auto skinLo=copyArray<double>(skinLoRaw,n), skinHi=copyArray<double>(skinHiRaw,n);
        auto fatLo=copyArray<double>(fatLoRaw,n), fatHi=copyArray<double>(fatHiRaw,n);
        for (std::size_t i=0;i<n;i++) for (unsigned k=0;k<3;k++) beta[i][k]=betaRaw[3*i+k];

        auto pointAt=[&](std::uint32_t sourceID, std::uint32_t sourceFace,
                const std::array<double,3>& b, std::uint8_t, V3& point, std::string& why) {
            if (sourceID!=NM_INVALID_INDEX) {
                if (sourceID>=sourceWorld.size()) { why="direct source slot out of range"; return false; }
                point=sourceWorld[sourceID]; return true;
            }
            if (sourceFace>=sourceFaces.size()) { why="source face out of range"; return false; }
            const auto f=sourceFaces[sourceFace]; point={};
            for (unsigned k=0;k<3;k++) {
                if (f[k]>=sourceWorld.size()) { why="source triangle index out of range"; return false; }
                point=ct::add(point,ct::mul(sourceWorld[f[k]],b[k]));
            }
            return true;
        };
        auto ownerFrameAt=[&](const std::array<double,86>& w, const V3&,
                ct::RegisteredSkinFatDonorFrame& frame, std::string& why) {
            std::size_t owner=0; double sum=0.0;
            for (std::size_t b=0;b<86;b++) {
                if (!std::isfinite(w[b]) || w[b]<-1.0e-8) { why="invalid interpolated source weight"; return false; }
                sum+=w[b]; if (w[b]>w[owner]) owner=b;
            }
            if (!std::isfinite(sum) || std::abs(sum-1.0)>1.0e-4) {
                why="interpolated source weights fail partition of unity"; return false;
            }
            if (slotDonor[owner]==NM_INVALID_INDEX) { why="dominant source binding has no native CSV donor frame"; return false; }
            frame.donorBody=slotDonor[owner];
            frame.originWorld=slotOrigin[owner];
            for (std::size_t k=0;k<9;k++) frame.worldToDonorLocalLinear[k]=slotMatrix[owner*9u+k];
            return true;
        };

        ct::RegisteredSkinFatCandidateView c;
        c.name="069i registered whole-body surface with bilateral hand replacement";
        c.provider="BodyParts3D .nhskin and native connected-hand registration CSV";
        c.sourceMember="replacement-geometry.npz world/faces/weights + registered-skin.csv wx/wy/wz/frame";
        c.sourceMeshSHA256=meshHash; c.sourcePayloadSHA256=payloadHash;
        c.stableIdentifier=1; c.sourceBodyIndex=NM_INVALID_INDEX; c.aggregateIdentity=true;
        c.outerWorld=outer; c.skinInnerWorld=inner; c.fatInnerWorld=deep;
        c.faces=faces; c.sourceFaceParents=parents; c.sourceFaces=sourceFaces;
        c.sourceVertexIds=sourceIds; c.sourceFaceIds=sourceFaceIds; c.sourceBarycentric=beta;
        c.vertexKinds=kinds; c.sourceResidualWorldM=residual; c.sourceWeights86=weights; c.sourceVertexCount=ns;
        c.skinThicknessM=skin; c.fatThicknessM=fat;
        c.skinMinimumByVertexM=skinLo; c.skinMaximumByVertexM=skinHi;
        c.fatMinimumByVertexM=fatLo; c.fatMaximumByVertexM=fatHi;
        c.skinThicknessProvenance="081 candidate-layers.npz per-node skin_thickness_m; O/I geometry retained";
        c.fatThicknessProvenance="081 candidate-layers.npz per-node fat_thickness_m; I/D geometry retained";
        c.skinBoundsProvenance="081 explicit per-node bounds for new rows; existing whole-body owner bounds for kind-zero inherited rows";
        c.fatBoundsProvenance="081 explicit per-node bounds for new rows; existing whole-body owner bounds for kind-zero inherited rows";
        c.sourcePointToleranceM=1.0e-9;

        ct::RegisteredSurface surface; ct::PairedBoundaryLayer skinLayer,fatLayer;
        ct::SkinFatReference reference; ct::RegisteredSkinFatCandidateBuildAudit buildAudit;
        std::string error;
        if (!ct::buildRegisteredSkinFatCandidateLayers(c,pointAt,ownerFrameAt,surface,skinLayer,fatLayer,reference,buildAudit,error)) {
            log << "status=BUILDER_REJECTED\nreason=" << error << "\n";
            setOutput(output,outputCapacity,log.str()); return 1;
        }
        std::size_t exactSourceIdRows=0, embeddedFaceRows=0;
        std::set<std::uint32_t> distinctOriginalFaceParents;
        std::size_t facesWithoutOriginalParent=0;
        for (auto id:surface.sourceVertexIndices) if (id!=NM_INVALID_INDEX) ++exactSourceIdRows;
        for (auto id:surface.sourceEmbeddingFaceIndices) if (id!=NM_INVALID_INDEX) ++embeddedFaceRows;
        for (auto parent:surface.sourceFaceIndices) {
            if (parent==NM_INVALID_INDEX) ++facesWithoutOriginalParent;
            else distinctOriginalFaceParents.insert(parent);
        }
        ct::GeometryAudit outerAudit;
        std::vector<std::array<std::uint32_t,2>> crossings;
        const bool outerOK=ct::auditClosedSurface(surface,outerAudit,error,true,&crossings);
        log << "status=BUILDER_PASS\nvertices=" << buildAudit.vertexCount
            << "\nfaces=" << buildAudit.faceCount
            << "\nkind_zero_rows=" << buildAudit.kindZeroRows
            << "\nnonzero_kind_rows=" << buildAudit.nonzeroKindRows
            << "\none_hot_rows=" << buildAudit.oneHotRows
            << "\nexact_source_ids=" << exactSourceIdRows
            << "\nembedded_source_face_rows=" << embeddedFaceRows
            << "\nprojected_nononehot_rows=" << (buildAudit.nonzeroKindRows-buildAudit.oneHotRows)
            << "\nunique_original_source_face_parents=" << distinctOriginalFaceParents.size()
            << "\nfaces_without_original_parent=" << facesWithoutOriginalParent
            << "\nsource_embedding_max_error_m=" << buildAudit.maximumBarycentricReconstructionErrorM
            << "\nretained_mechanical_delta_max_m=" << buildAudit.maximumRetainedMechanicalDeltaM
            << "\nsource_surface_audit=" << (outerOK?"PASS":"FAIL")
            << "\nsource_open_boundary_edges=" << outerAudit.boundaryEdgeCount
            << "\nsource_boundary_loops=" << outerAudit.boundaryLoopCount
            << "\nsource_surface_volume_m3=" << outerAudit.volumeM3
            << "\nsource_crossing_pairs=" << crossings.size()
            << "\nskin_layer_volume_m3=" << skinLayer.volumeM3
            << "\nfat_layer_volume_m3=" << fatLayer.volumeM3
            << "\nskin_bounds_envelope_m=" << buildAudit.skinBoundEnvelopeMinimumM << "," << buildAudit.skinBoundEnvelopeMaximumM
            << "\nfat_bounds_envelope_m=" << buildAudit.fatBoundEnvelopeMinimumM << "," << buildAudit.fatBoundEnvelopeMaximumM
            << "\n";
        if (!outerOK) {
            log << "status=SOURCE_AUDIT_REJECTED\nreason=" << error << "\n";
            setOutput(output,outputCapacity,log.str()); return 1;
        }

        auto auditOffset=[&](const std::vector<V3>& points,const std::vector<V3>& local,
                const char* label) {
            ct::RegisteredSurface boundary=surface;
            boundary.name=std::string(surface.name)+" "+label;
            boundary.worldPoints=points; boundary.donorLocalPoints=local;
            ct::GeometryAudit ga; std::vector<std::array<std::uint32_t,2>> hits; std::string why;
            const bool ok=ct::auditClosedSurface(boundary,ga,why,true,&hits);
            log << label << "_audit=" << (ok?"PASS":"FAIL")
                << "\n" << label << "_open_boundary_edges=" << ga.boundaryEdgeCount
                << "\n" << label << "_crossing_pairs=" << hits.size()
                << "\n" << label << "_volume_m3=" << ga.volumeM3
                << "\n" << label << "_failure=" << why << "\n";
        };
        auditOffset(skinLayer.innerWorld,skinLayer.innerLocal,"skin_inner");
        auditOffset(fatLayer.innerWorld,fatLayer.innerLocal,"fat_deep");

        numi::matter::WorldSource world;
        world.compactImmutableFEMSurfaces=true; world.frameTimestep=0.001; world.environmentCount=1;
        world.materials={testMaterial("081_skin",900.0,1300.0),testMaterial("081_fat",700.0,1100.0)};
        const std::array<std::uint64_t,4> materialIdentity{0x08120261023ULL,0x0691u,0x5a1fULL,0x86u};
        ct::PreparedOrganicFEM preparedSkin,preparedFat;
        const bool appended=ct::appendRegisteredSkinFatFEMFromLayers(world,surface,0,"081_skin",
            "existing donor mass partition; CPU preparation fixture only",debit[0],
            1,"081_fat","existing donor mass partition; CPU preparation fixture only",debit[1],
            skin,fat,skinLayer.thicknessProvenance,fatLayer.thicknessProvenance,reference,skinLayer,fatLayer,
            "081 candidate skin","081 candidate fat",materialIdentity,preparedSkin,preparedFat,error);
        log << "append_from_layers=" << (appended?"PASS":"FAIL") << "\nappend_error=" << error << "\n";
        if (!appended) { setOutput(output,outputCapacity,log.str()); return 2; }
        double donorMass=0.0;
        for (const auto& row:preparedSkin.donorMassNodes) donorMass+=row.massKg;
        log << "prepared_skin_volume_m3=" << preparedSkin.volumeM3
            << "\nprepared_fat_volume_m3=" << preparedFat.volumeM3
            << "\nprepared_skin_density_kg_m3=" << preparedSkin.densityKgM3
            << "\nprepared_fat_density_kg_m3=" << preparedFat.densityKgM3
            << "\nprepared_skin_mass_residual_kg=" << preparedSkin.massResidualKg
            << "\nprepared_fat_mass_residual_kg=" << preparedFat.massResidualKg
            << "\ncombined_unique_donor_nodal_mass_kg=" << donorMass
            << "\nworld_objects=" << world.objects.size() << "\n";
        setOutput(output,outputCapacity,log.str()); return 0;
    } catch (const std::exception& e) {
        log << "status=EXCEPTION\nreason=" << e.what() << "\n";
        setOutput(output,outputCapacity,log.str()); return 3;
    } catch (...) {
        log << "status=EXCEPTION\nreason=unknown\n";
        setOutput(output,outputCapacity,log.str()); return 3;
    }
}

