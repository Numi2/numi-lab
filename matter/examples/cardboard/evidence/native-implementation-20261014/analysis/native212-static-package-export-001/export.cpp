#include "numi/matter/compiler.hpp"
#include <fstream>
#include <iostream>
#include <span>
#include <type_traits>
using namespace numi::matter;
template<class T> void write(const std::filesystem::path& p, std::span<const T> v){static_assert(std::is_trivially_copyable_v<T>);std::ofstream f(p,std::ios::binary); f.write(reinterpret_cast<const char*>(v.data()),static_cast<std::streamsize>(v.size_bytes()));if(!f)throw std::runtime_error("write failed");}
int main(int argc,char**argv){try{if(argc!=3)throw std::runtime_error("package output-dir");CompiledMatterWorld w;std::string e;if(!readMatterPackage(argv[1],w,nullptr,&e))throw std::runtime_error(e);std::filesystem::path o=argv[2];std::filesystem::create_directories(o);write(o/"tetrahedra.bin",std::span<const NMTetrahedronGPU>(w.fem.tetrahedra));write(o/"nodes.bin",std::span<const NMFEMNodeStateGPU>(w.fem.nodes));write(o/"objects.bin",std::span<const NMContinuumObjectGPU>(w.objects));write(o/"schedulers.bin",std::span<const NMSchedulerStateGPU>(w.schedulers));write(o/"dispatch.bin",std::span<const NMMatterDispatchGPU>(&w.dispatch,1));std::cout<<"{\"fingerprint\":\""<<std::hex<<w.fingerprint<<"\",\"physics_fingerprint\":\""<<w.physicsFingerprint<<"\",\"nodes\":"<<std::dec<<w.fem.nodes.size()<<",\"tetrahedra\":"<<w.fem.tetrahedra.size()<<",\"tet_stride\":"<<sizeof(NMTetrahedronGPU)<<",\"object_stride\":"<<sizeof(NMContinuumObjectGPU)<<",\"node_stride\":"<<sizeof(NMFEMNodeStateGPU)<<"}\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}}
