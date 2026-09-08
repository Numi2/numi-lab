#include "metalrobo/NumiHumanSupport.hpp"
#include "metalrobo/numi_human_stand_gpu.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <set>
namespace metalrobo {
namespace {
struct Primitive {
    std::uint32_t body, geometry, kind, reserved;
    float endpointA[3], radius, endpointB[3], friction;
    float gapA, gapB, reserved0, reserved1;
    float orientation[4], radii[4];
};
static_assert(sizeof(Primitive) == 96);
static_assert(sizeof(NumiHumanSupportHeader) == 84);
static_assert(offsetof(NumiHumanSupportContact, supportRadius) == 48);
}
MRArticulatedPointImpulseGPU compileNumiHumanSupportQuery(
    const NumiHumanSupportHeader& h, const NumiHumanSupportContact& c) {
    MRArticulatedPointImpulseGPU q{};
    q.bodyIndex=c.bodyIndex; q.localPoint={c.localPointX,c.localPointY,c.localPointZ,0};
    if (c.supportRadii[0]>0) q.flags=MR_ARTICULATED_POINT_ELLIPSOID_SUPPORT;
    else if (c.supportRadius>0) q.flags=MR_ARTICULATED_POINT_SPHERE_SUPPORT;
    if (q.flags) q.supportPlaneNormalAndRadius={h.groundNormalX,h.groundNormalY,h.groundNormalZ,c.supportRadius};
    q.supportRadii={c.supportRadii[0],c.supportRadii[1],c.supportRadii[2],0};
    q.supportOrientation={c.supportOrientation[0],c.supportOrientation[1],c.supportOrientation[2],c.supportOrientation[3]};
    return q;
}
bool decodeNumiHumanSupportPayload(std::span<const std::byte> bytes,
    std::uint32_t expectedBodyCount,
    const std::array<std::uint8_t, 32>& expectedSource,
    NumiHumanSupportPayload& output, std::string& error) {
    const auto fail = [&](const char* message) { error = message; return false; };
    if (std::endian::native != std::endian::little || bytes.size() < 84)
        return fail("truncated or unsupported-endian NHCNT header");
    NumiHumanSupportPayload candidate;
    auto& h = candidate.header;
    std::memcpy(&h, bytes.data(), 84);
    const std::array<char, 8> v1{'N','H','C','N','T','1',0,0}, v2{'N','H','C','N','T','2',0,0};
    const bool legacy = h.magic == v1 && h.payloadAbi == 1;
    const bool primitive = h.magic == v2 && h.payloadAbi == 2;
    const auto finite = [](float v) { return std::isfinite(v); };
    const float normal2 = h.groundNormalX*h.groundNormalX +
        h.groundNormalY*h.groundNormalY + h.groundNormalZ*h.groundNormalZ;
    if ((!legacy && !primitive) || h.engineBodyCount != expectedBodyCount ||
        h.sourceSha256 != expectedSource || h.reserved0 != 0 ||
        h.contactCount == 0 || h.contactCount > MR_NUMI_HUMAN_STAND_MAX_CONTACTS ||
        !finite(h.groundPointX) || !finite(h.groundPointY) || !finite(h.groundPointZ) ||
        !finite(normal2) || std::abs(normal2-1.0f) > 1.0e-5f ||
        !finite(h.groundFriction) || h.groundFriction < 0)
        return fail("invalid or foreign NHCNT header/plane");
    if (bytes.size() != 84 + std::size_t(h.contactCount)*(legacy ? 48 : 96))
        return fail("NHCNT record extent mismatch");
    std::set<std::uint32_t> identities;
    for (std::uint32_t i=0; i<h.contactCount; ++i) {
        NumiHumanSupportContact contact;
        if (legacy) {
            std::memcpy(&contact, bytes.data()+84+i*48, 48);
            const float gap = (contact.worldWitnessX-h.groundPointX)*h.groundNormalX +
                (contact.worldWitnessY-h.groundPointY)*h.groundNormalY +
                (contact.worldWitnessZ-h.groundPointZ)*h.groundNormalZ;
            if (!finite(contact.localPointX) || !finite(contact.localPointY) || !finite(contact.localPointZ) ||
                !finite(contact.worldWitnessX) || !finite(contact.worldWitnessY) || !finite(contact.worldWitnessZ) ||
                !finite(contact.friction) || contact.friction < 0 ||
                !finite(contact.defaultSignedPlaneDistance) || !finite(gap) || std::abs(gap)>2.0e-4f ||
                contact.reserved0 != 0 || contact.reserved1 != 0)
                return fail("malformed NHCNT1 witness");
            candidate.contacts.push_back(contact);
        } else {
            Primitive p{};
            std::memcpy(&p, bytes.data()+84+i*96, 96);
            if ((p.kind != 1 && p.kind != 2 && p.kind != 3) || p.reserved != 0 ||
                !std::all_of(std::begin(p.endpointA),std::end(p.endpointA),finite) ||
                !std::all_of(std::begin(p.endpointB),std::end(p.endpointB),finite) ||
                !finite(p.radius) || (p.kind == 3 ? p.radius != 0 : p.radius <= 0) || !finite(p.friction) || p.friction < 0 ||
                !finite(p.gapA) || !finite(p.gapB) || p.reserved0 != 0 || p.reserved1 != 0 ||
                (p.kind != 2 && (p.endpointA[0]!=p.endpointB[0] || p.endpointA[1]!=p.endpointB[1] ||
                                p.endpointA[2]!=p.endpointB[2] || p.gapA!=p.gapB)))
                return fail("malformed NHCNT2 sphere/capsule");
            float norm2=0;
            for (float x : p.orientation) norm2+=x*x;
            if (!std::all_of(std::begin(p.radii),std::end(p.radii),finite) || !finite(norm2) ||
                (p.kind==3 ? (p.radii[0]<=0 || p.radii[1]<=0 || p.radii[2]<=0 || p.radii[3]!=0 || std::abs(norm2-1)>1.0e-5f)
                           : (p.radii[0]!=0 || p.radii[1]!=0 || p.radii[2]!=0 || p.radii[3]!=0 || norm2!=0)))
                return fail("invalid NHCNT2 ellipsoid axes/orientation");
            std::copy_n(p.radii,3,contact.supportRadii.begin());
            std::copy_n(p.orientation,4,contact.supportOrientation.begin());
            contact.bodyIndex=p.body; contact.sourceGeometryIndex=p.geometry;
            contact.friction=p.friction; contact.supportRadius=p.radius;
            contact.worldWitnessX=h.groundPointX; contact.worldWitnessY=h.groundPointY;
            contact.worldWitnessZ=h.groundPointZ;
            for (std::uint32_t end=0; end<(p.kind==2 ? 2u : 1u); ++end) {
                const float* point=end==0 ? p.endpointA : p.endpointB;
                contact.localPointX=point[0]; contact.localPointY=point[1]; contact.localPointZ=point[2];
                contact.defaultSignedPlaneDistance=end==0 ? p.gapA : p.gapB;
                candidate.contacts.push_back(contact);
            }
        }
        if (contact.bodyIndex >= expectedBodyCount || contact.sourceGeometryIndex == MR_INVALID_INDEX ||
            !identities.insert(contact.sourceGeometryIndex).second)
            return fail("unresolved or duplicate NHCNT primitive identity");
    }
    if (candidate.contacts.size() > MR_NUMI_HUMAN_STAND_MAX_CONTACTS)
        return fail("expanded NHCNT support capacity exceeded");
    h.contactCount=static_cast<std::uint32_t>(candidate.contacts.size());
    output=std::move(candidate); error.clear(); return true;
}
} // namespace metalrobo
