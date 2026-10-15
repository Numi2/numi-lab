#include "numi/matter/compiler.hpp"
#include "numi/matter/paper_compensated_bytecode.h"
#include "numi/matter/fem_compensated_deformation.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <span>
#include <type_traits>
#include <vector>
using namespace numi::matter;

template<class T> void dump(const std::filesystem::path& p, const std::vector<T>& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size()*sizeof(T)));
    if (!f) throw std::runtime_error("dump failed: " + p.string());
}

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("package outdir candidate_nodes.bin");
        CompiledWorld w; std::string error;
        if (!readMatterPackage(argv[1], w, nullptr, &error)) throw std::runtime_error(error);
        const std::filesystem::path out = argv[2];
        std::filesystem::create_directories(out);
        dump(out/"tetrahedra.bin", w.fem.tetrahedra);
        dump(out/"materials.bin", w.materials);
        dump(out/"instructions.bin", w.instructions);
        dump(out/"scalar_programs.bin", w.scalarPrograms);
        dump(out/"parameters.bin", w.parameters);
        dump(out/"nodes.bin", w.fem.nodes);
        std::cout << "fingerprint=" << std::hex << w.fingerprint << " physics=" << w.physicsFingerprint << std::dec
            << " nodes=" << w.fem.nodes.size() << " tets=" << w.fem.tetrahedra.size()
            << " materials=" << w.materials.size() << " instructions=" << w.instructions.size()
            << " programs=" << w.scalarPrograms.size() << " params=" << w.parameters.size()
            << " sizeof_tet=" << sizeof(NMTetrahedronGPU) << " sizeof_material=" << sizeof(NMMaterialGPU)
            << " sizeof_inst=" << sizeof(NMExpressionInstructionGPU) << "\n";
        const auto& material = w.materials.at(1);
        const auto& program = w.scalarPrograms.at(material.validityProgram);
        std::cout << "medium validityProgram=" << material.validityProgram
            << " params=" << material.parameterOffset << "," << material.parameterCount
            << " state=" << material.stateCount << " validity=" << material.validity.x << "," << material.validity.y
            << " flags=" << material.flags << " programRange=" << program.firstInstruction << ","
            << program.instructionCount << " programFlags=" << program.flags << "\n";
        std::vector<float> parameterValues;
        for (const auto& p : w.parameters) parameterValues.push_back(p.valueAndBounds.x);
        std::vector<NMFEMNodeStateGPU> candidate(1764);
        {
            std::ifstream in(argv[3], std::ios::binary);
            in.read(reinterpret_cast<char*>(candidate.data()), static_cast<std::streamsize>(candidate.size()*sizeof(NMFEMNodeStateGPU)));
            if (!in) throw std::runtime_error("candidate read failed");
        }
        const auto& tet = w.fem.tetrahedra.at(858);
        const unsigned ids[4] = {tet.nodes.x, tet.nodes.y, tet.nodes.z, tet.nodes.w};
        float displacement[12], inverse[9], frame[9], fHigh[9], fLow[9];
        for (unsigned n=0; n<4; ++n) {
            const auto u = candidate[ids[n]].referenceDisplacementAndMode;
            displacement[3*n]=u.x; displacement[3*n+1]=u.y; displacement[3*n+2]=u.z;
        }
        const auto q=tet.materialFrameRotation;
        const float qnorm=1.0f/std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w);
        const float x=q.x*qnorm, y=q.y*qnorm, z=q.z*qnorm, qw=q.w*qnorm;
        const float rotation[9]={
            1-2*(y*y+z*z), 2*(x*y-z*qw), 2*(x*z+y*qw),
            2*(x*y+z*qw), 1-2*(x*x+z*z), 2*(y*z-x*qw),
            2*(x*z-y*qw), 2*(y*z+x*qw), 1-2*(x*x+y*y)};
        const bool framed=(w.objects.at(0).flags & NM_OBJECT_FEM_MATERIAL_FRAME)!=0u;
        const float* rows[3]={&tet.inverseRestRow0.x,&tet.inverseRestRow1.x,&tet.inverseRestRow2.x};
        for(unsigned r=0;r<3;++r) for(unsigned c=0;c<3;++c) {
            inverse[3*r+c]=rows[r][c];
            frame[3*r+c]=framed ? rotation[3*r+c] : (r==c ? 1.0f : 0.0f);
        }
        nmFEMReferenceDeformationWords(displacement,inverse,frame,fHigh,fLow);
        bool valid=true; float zero[9]={};
        const float validityMargin=nmPaperCompensatedBytecode<0u>(
            program,w.instructions.data(),parameterValues.data()+material.parameterOffset,
            static_cast<const float*>(nullptr),static_cast<const float*>(nullptr),
            fHigh,zero,zero,1.0e-4f,material.bulk.y,valid,nullptr,fLow);
        std::cout << "tet858 nodes=" << ids[0] << "," << ids[1] << "," << ids[2] << "," << ids[3]
            << " material=" << tet.identity.x << " object=" << tet.identity.y << " flags=" << tet.identity.w << "\n";
        std::cout << "fHigh="; for(float v:fHigh) std::cout << v << ",";
        std::cout << " fLow="; for(float v:fLow) std::cout << v << ",";
        std::cout << " evaluatedValidity=" << validityMargin << " valid=" << valid << "\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 2; }
}
