// Narrow, same-build diagnostic checkpoint for one FEM/slider transaction.
// Exact bytes, including compensated limbs and contact history, are required.
#pragma once
#include <fstream>
#include <type_traits>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <string>

namespace slider_checkpoint {
using numi::matter::RuntimeStateSnapshot;
constexpr std::uint64_t magic=0x534c494445523031ull;
template<class T> void scalar(std::ostream& o,const T& v) {
 static_assert(std::is_trivially_copyable_v<T>);o.write(reinterpret_cast<const char*>(&v),sizeof(v));
}
template<class T> void scalar(std::istream& i,T& v) {
 static_assert(std::is_trivially_copyable_v<T>);i.read(reinterpret_cast<char*>(&v),sizeof(v));
}
template<class T> void array(std::ostream& o,const std::vector<T>& v) {
 static_assert(std::is_trivially_copyable_v<T>);const std::uint64_t n=v.size();scalar(o,n);
 if(n)o.write(reinterpret_cast<const char*>(v.data()),n*sizeof(T));
}
template<class T> void array(std::istream& i,std::vector<T>& v) {
 static_assert(std::is_trivially_copyable_v<T>);std::uint64_t n=0;scalar(i,n);
 if(!i||n>1000000 || n*sizeof(T)>16u*1024u*1024u)throw std::runtime_error("invalid slider checkpoint vector");
 v.resize(static_cast<std::size_t>(n));if(n)i.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));
}
#define SLIDER_CHECKPOINT_SCALARS(X) \
 X(sourcePhysicsFingerprint) X(deviceProgramFingerprint) X(controlStep) X(physicsSubstep) \
 X(allocationGeneration) X(learnedWeightRevision) X(identificationGeneration) X(identificationCheckpoint) \
 X(identificationAdvanced) X(sutureProxyBindingRevision) X(coupledTimestepMultiplier) X(coupledTimestepDivisor) \
 X(fgmresIterationBudgetOverride) X(newtonIterationBudgetOverride) X(materialStateStride)
#define SLIDER_CHECKPOINT_ARRAYS(X) \
 X(particles) X(femNodes) X(femMidpointVelocityLow) X(particleMaterialState) X(femMaterialState) X(femFields) \
 X(femTopologyNodes) X(femTopologyTetrahedra) X(cohesiveFaces) X(punctureChannels) X(contactHistories) X(schedulers) \
 X(adaptive) X(topologyStates) X(deformableContactHistories) X(humanSupportHistories) X(humanSupportConsequences) \
 X(vascularState) X(vascularClock) X(learnedWeights) X(identification) X(environmentParameters) X(sutureProxyEdges) \
 X(reactions) X(rigidStates) X(rigidGeneralizedCandidate) X(statuses) X(solverCertificates) X(contactSamples)
inline void save(const std::string& path,const RuntimeStateSnapshot& s,const std::vector<float>& q,const std::vector<float>& v) {
 std::ifstream exists(path,std::ios::binary);if(exists.good())throw std::runtime_error("slider checkpoint already exists");
 std::ofstream o(path,std::ios::binary|std::ios::out);if(!o)throw std::runtime_error("cannot create slider checkpoint");
 scalar(o,magic);
#define WRITE_SCALAR(x) scalar(o,s.x);
 SLIDER_CHECKPOINT_SCALARS(WRITE_SCALAR)
#undef WRITE_SCALAR
#define WRITE_ARRAY(x) array(o,s.x);
 SLIDER_CHECKPOINT_ARRAYS(WRITE_ARRAY)
#undef WRITE_ARRAY
 array(o,q);array(o,v);o.flush();if(!o)throw std::runtime_error("could not finish slider checkpoint");
}
inline void load(const std::string& path,RuntimeStateSnapshot& s,std::vector<float>& q,std::vector<float>& v) {
 std::ifstream i(path,std::ios::binary);std::uint64_t header=0;scalar(i,header);
 if(!i||header!=magic)throw std::runtime_error("invalid slider checkpoint header");
#define READ_SCALAR(x) scalar(i,s.x);
 SLIDER_CHECKPOINT_SCALARS(READ_SCALAR)
#undef READ_SCALAR
#define READ_ARRAY(x) array(i,s.x);
 SLIDER_CHECKPOINT_ARRAYS(READ_ARRAY)
#undef READ_ARRAY
 array(i,q);array(i,v);if(!i||i.peek()!=std::char_traits<char>::eof()||q.size()!=1||v.size()!=1)throw std::runtime_error("invalid slider checkpoint body");
 s.available=true;
}
#undef SLIDER_CHECKPOINT_ARRAYS
#undef SLIDER_CHECKPOINT_SCALARS
}
