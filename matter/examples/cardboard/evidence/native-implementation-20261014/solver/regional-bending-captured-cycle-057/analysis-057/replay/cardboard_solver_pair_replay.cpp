#include "cardboard_solver_reference.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace paper = numi_cardboard_paper;
namespace oracle = numi_cardboard_solver;

template<class T> void printArray(const T& values, double scale=1.0) {
    std::cout << '[';
    bool first = true;
    for (double value : values) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << value * scale;
    }
    std::cout << ']';
}

template<std::size_t N>
std::array<double,N> readF32Array(std::istream& input) {
    std::array<double,N> values{};
    for (double& value : values) {
        if (!(input >> value) || !std::isfinite(value))
            throw std::runtime_error("invalid captured f32 value");
        value = static_cast<float>(value);
        if (!std::isfinite(value)) throw std::runtime_error("captured value outside f32 range");
    }
    return values;
}

template<std::size_t N>
std::array<double,N> readF32WordPair(std::istream& input) {
    const auto high = readF32Array<N>(input);
    const auto low = readF32Array<N>(input);
    std::array<double,N> values{};
    for (std::size_t i=0; i<N; ++i) values[i] = high[i] + low[i];
    return values;
}

std::array<paper::Parameters,2> loadParameters(const char* path) {
    std::array<paper::Parameters,2> parameters{paper::liner,paper::medium};
    std::array<bool,2> seen{};
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read captured parameter file");
    std::string name;
    unsigned count=0;
    while (input >> name) {
        if (name!="liner" && name!="medium") throw std::runtime_error("unknown parameter material");
        const unsigned index=name=="liner"?0:1;
        if (seen[index]) throw std::runtime_error("duplicate material parameters");
        std::array<double,20> values{};
        for (double& value : values) {
            if (!(input >> value) || !std::isfinite(value)) throw std::runtime_error("invalid material parameter");
            value=static_cast<float>(value);
            if (!std::isfinite(value)) throw std::runtime_error("material parameter outside f32 range");
        }
        auto& p=parameters[index];
        p.density=values[0];
        for (unsigned i=0;i<6;++i) p.c[i]=values[1+i]*1e-6;
        for (unsigned i=0;i<3;++i) p.shear[i]=values[7+i]*1e-6;
        p.sigmaReference=values[10]*1e-6;
        for (unsigned i=0;i<6;++i) p.hill[i]=values[11+i];
        p.qFloor=values[17];
        p.fischerBurmeisterEpsilon=values[18];
        if (!(p.sigmaReference>0 && p.qFloor>0 && p.fischerBurmeisterEpsilon>0 && values[19]>0))
            throw std::runtime_error("nonpositive captured constitutive parameter");
        seen[index]=true;
        ++count;
    }
    if (count!=2) throw std::runtime_error("both captured paper parameter sets are required");
    return parameters;
}

int main(int argc, char** argv) {
    std::cout << std::setprecision(17);
    if (argc!=2) {
        std::cerr << "usage: cardboard_solver_pair_replay <captured-material-parameters.txt>\n"
                  << "stdin rows: key material F_hi[9] F_lo[9] H[9] accepted[13] candidate_hi[13] candidate_lo[13]\n";
        return 1;
    }
    try {
        const auto parameters=loadParameters(argv[1]);
        std::string line;
        unsigned rows=0,failures=0;
        while (std::getline(std::cin,line)) {
            if (line.empty() || line[0]=='#') continue;
            std::istringstream input(line);
            std::string key,kindName;
            if (!(input>>key>>kindName)) throw std::runtime_error("missing replay row identity");
            std::cout << "{\"schema\":\"numi.cardboard.solver-compensated-word-replay.v1\",\"key\":\"" << key << "\"";
            try {
                if (kindName!="liner" && kindName!="medium") throw std::runtime_error("unsupported material");
                const auto& p=parameters[kindName=="liner"?0:1];
                const auto f=readF32WordPair<9>(input);
                const auto h=readF32Array<9>(input);
                const auto accepted=readF32Array<13>(input);
                const auto candidate=readF32WordPair<13>(input);
                std::string extra;
                if (input>>extra) throw std::runtime_error("extra replay columns");
                const auto projected=oracle::project(p,f,accepted);
                const auto tangent=oracle::directionalTangent(p,f,h,accepted,projected.state);
                const auto candidateTangent=oracle::directionalTangent(p,f,h,accepted,candidate);
                const auto candidateStress=paper::firstPiola(p,f,paper::plasticStrain(candidate));
                const double candidateResidual=paper::residualNorm(paper::residual(p,f,accepted,candidate));
                const auto trialStress=paper::normalizedSecondPiola(p,f,paper::plasticStrain(accepted));
                const double trialQ=paper::hillEquivalent(p,trialStress);
                double candidateStateError=0;
                for (unsigned i=0;i<13;++i) candidateStateError=std::max(candidateStateError,std::abs(projected.state[i]-candidate[i]));
                std::cout << ",\"valid\":true,\"word_contract\":\"each_f32_high_low_widened_then_summed_in_f64\",\"oracle_P_Pa\":";
                printArray(projected.firstPiola,1e6);
                std::cout << ",\"oracle_tangent_Pa\":";
                printArray(tangent,1e6);
                std::cout << ",\"oracle_tangent_at_candidate_Pa\":";
                printArray(candidateTangent,1e6);
                std::cout << ",\"oracle_candidate_P_Pa\":";
                printArray(candidateStress,1e6);
                std::cout << ",\"oracle_state\":";
                printArray(projected.state);
                std::cout << ",\"candidate_state\":";
                printArray(candidate);
                std::cout << ",\"oracle_residual\":" << projected.residual
                          << ",\"candidate_residual\":" << candidateResidual
                          << ",\"independent_residual_at_metal_state\":" << candidateResidual
                          << ",\"candidate_state_max_abs_error\":" << candidateStateError
                          << ",\"iterations\":" << projected.iterations
                          << ",\"projected_q\":" << projected.equivalentStress
                          << ",\"elastic_trial_q\":" << trialQ
                          << ",\"elastic_trial_q_dimensionless\":" << trialQ
                          << ",\"lambda_increment\":" << projected.state[12]-accepted[12]
                          << ",\"candidate_lambda_increment\":" << candidate[12]-accepted[12];
            } catch (const std::exception& error) {
                ++failures;
                std::cout << ",\"valid\":false,\"reason\":\"" << error.what() << "\"";
            }
            std::cout << "}\n";
            ++rows;
        }
        if (rows==0) throw std::runtime_error("no replay rows");
        return failures ? 2 : 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
