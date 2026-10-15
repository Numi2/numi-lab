#include "numi/matter/detail.hpp"
#include "numi/matter/paper_compensated_bytecode.h"
#include "cardboard_paper_projection_scaling.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace numi::matter;
struct CompiledResiduals {
    std::vector<NMScalarProgramGPU> programs;
    std::vector<NMExpressionInstructionGPU> instructions;
};

CompiledResiduals flatten(const ConstitutiveProgram& compiled) {
    CompiledResiduals out;
    for (const ScalarBytecode& code : compiled.implicitResiduals) {
        NMScalarProgramGPU program{};
        program.firstInstruction = static_cast<nm_u32>(out.instructions.size());
        program.instructionCount = static_cast<nm_u32>(code.instructions.size());
        program.maximumStack = code.maximumStack;
        program.flags = 2u; // source snapshot's compensated-bytecode policy
        out.instructions.insert(out.instructions.end(), code.instructions.begin(),
                                code.instructions.end());
        out.programs.push_back(program);
    }
    return out;
}

template <std::size_t N>
void readValues(std::istringstream& line, std::array<float, N>& values) {
    for (float& value : values) {
        double raw = 0.0;
        if (!(line >> raw) || !std::isfinite(raw))
            throw std::runtime_error("malformed/nonfinite input row");
        value = static_cast<float>(raw);
        if (!std::isfinite(value)) throw std::runtime_error("FP32 overflow");
    }
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: replay_compensated_vm LINER.nmatter MEDIUM.nmatter");
        auto linerSource = parseMatterFile(argv[1]);
        auto mediumSource = parseMatterFile(argv[2]);
        if (!linerSource.succeeded() || !mediumSource.succeeded())
            throw std::runtime_error("could not parse source material");
        for (auto* material : {&linerSource.material, &mediumSource.material}) {
            material->paperHillScalarInitializer = true;
            material->paperCompensatedArithmetic = true;
            material->paperHillAccurateProjection = true;
            *material = numi::cardboard::scalePaperProjectionResiduals(
                std::move(*material));
        }
        const auto liner = detail::compileConstitutive(linerSource.material,
                                                       NM_EXPRESSION_STACK_CAPACITY);
        const auto medium = detail::compileConstitutive(mediumSource.material,
                                                        NM_EXPRESSION_STACK_CAPACITY);
        if (!liner.succeeded() || !medium.succeeded())
            throw std::runtime_error("could not compile source material");
        const std::array<CompiledResiduals, 2> code{
            flatten(liner.program), flatten(medium.program)};
        std::cout.precision(17);
        std::string rawLine;
        while (std::getline(std::cin, rawLine)) {
            if (rawLine.empty() || rawLine[0] == '#') continue;
            std::istringstream line(rawLine);
            std::string key;
            unsigned material = 0u;
            if (!(line >> key >> material) || material > 1u)
                throw std::runtime_error("bad key/material");
            std::array<float, 9> f{}, fLow{};
            std::array<float, 13> accepted{}, candidateHigh{}, candidateLow{};
            std::array<float, 20> parameters{};
            readValues(line, f); readValues(line, fLow); readValues(line, accepted);
            readValues(line, candidateHigh); readValues(line, candidateLow);
            readValues(line, parameters);
            std::string extra;
            if (line >> extra) throw std::runtime_error("extra input columns");
            const auto& programs = code[material].programs;
            if (programs.size() != 13u) throw std::runtime_error("expected 13 residual rows");
            const NMPaperCandidateState candidate{candidateHigh.data(), candidateLow.data()};
            float residualMax = 0.0f;
            std::array<float, 13> residual{};
            bool valid = true;
            for (unsigned i = 0; i < 13u; ++i) {
                bool rowValid = true;
                residual[i] = nmPaperCompensatedBytecode<0u>(
                    programs[i], code[material].instructions.data(), parameters.data(),
                    accepted.data(), candidate, f.data(), nullptr, nullptr,
                    0.0f, 0.0f, rowValid, nullptr, fLow.data());
                valid = valid && rowValid;
                residualMax = std::max(residualMax, std::abs(residual[i]));
            }
            if (!valid) throw std::runtime_error("compensated source VM rejected input row");
            std::cout << "{\"key\":\"" << key << "\",\"material\":" << material
                      << ",\"residual_scaled_inf\":" << residualMax
                      << ",\"residual_rows_scaled\":[";
            for (unsigned i = 0; i < 13u; ++i) {
                if (i) std::cout << ',';
                std::cout << residual[i];
            }
            std::cout << "]}\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
