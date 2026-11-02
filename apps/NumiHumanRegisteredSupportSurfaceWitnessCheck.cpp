#include "NumiHumanRegisteredSupportSurfaceWitness.hpp"
#include <cassert>
#include <iostream>
using namespace metalrobo::human::coarse_tissue;
int main(){
std::vector<std::array<double,3>> points{{0.080068252980709076,-0.40477907657623291,0.0034570030402392149},{0.036083899438381195,-0.51138162612915039,0.01495400071144104},{0.0063767572864890099,-0.41903653740882874,0.011192119680345058}};
std::vector<std::array<std::uint32_t,3>> faces{{0,1,2}};std::vector<std::uint32_t> donors{20,20,20};
const std::array<double,3> query{.047828856855630875,-.41048908233642578,.00059818720910698175};
RegisteredSupportSurfaceWitness w;
assert(findRegisteredSupportSurfaceWitness(points,faces,donors,20,query,.030,w));
assert(w.faceRow==0&&w.surfaceDistanceM<.006222&&w.surfaceDistanceM>.006221);
assert(w.representativeDistanceM>.030&&w.representativeDistanceM<.033);
assert(w.barycentric[0]>=0&&w.barycentric[1]>=0&&w.barycentric[2]>=0);
assert(std::abs(w.barycentric[0]+w.barycentric[1]+w.barycentric[2]-1.)<1e-14);
const auto accepted=w;
assert(!findRegisteredSupportSurfaceWitness(points,faces,donors,21,query,.030,w));
assert(w.faceRow==accepted.faceRow&&w.point==accepted.point);
assert(!findRegisteredSupportSurfaceWitness(points,faces,donors,20,query,.006,w));
donors[2]=21;assert(!findRegisteredSupportSurfaceWitness(points,faces,donors,20,query,.030,w));donors[2]=20;
faces[0][2]=3;assert(!findRegisteredSupportSurfaceWitness(points,faces,donors,20,query,.030,w));faces[0][2]=2;
points[2]=points[1];assert(!findRegisteredSupportSurfaceWitness(points,faces,donors,20,query,.030,w));
struct Binding{std::uint32_t bodyIndex;};struct Vertex{std::array<std::uint32_t,4> bindingIndex;std::array<float,4> weight;};
struct Source{std::vector<Binding> bindings;std::vector<Vertex> vertices;std::vector<float> fullWeights;};
Source source{{{62},{49},{57},{55},{67}},{{{0,1,2,3},{.4f,.3f,.2f,.1f}}},{.284f,.211f,.085f,.068f,.062f}};
std::array<RegisteredSupportSourceInfluence,4> support{};
assert(registeredSupportSourceInfluences(source,0,support));assert(support[0].body==62&&support[1].body==49&&support[3].body==55);
source.fullWeights[0]=std::numeric_limits<float>::quiet_NaN();assert(!registeredSupportSourceInfluences(source,0,support));
source.fullWeights.clear();assert(registeredSupportSourceInfluences(source,0,support));assert(support[1].body==49);
assert(!registeredSupportSourceInfluences(source,1,support));
std::cout<<"registered_support_surface_witness=PASS bound_m=0.030 physical_contact_changed=0\n";
}
