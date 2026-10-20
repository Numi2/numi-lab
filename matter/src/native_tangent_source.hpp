#pragma once
// Asset-preparation lowering only. No per-step CPU material evaluation.
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "metal/generated_stateless_tangent.metalinc"

template<class World>
static bool nmNativeViscousFloatProgramEligible(
    const World& w,
    const NMMaterialGPU& material,
    const std::uint32_t programIndex,
    const bool allowDirection
) {
    if (programIndex >= w.scalarPrograms.size()) return false;
    const auto& program = w.scalarPrograms[programIndex];
    constexpr std::uint32_t allowedProgramFlags =
        NM_SCALAR_PROGRAM_FLAG_LOCAL_CACHE;
    if (program.instructionCount == 0u ||
        program.instructionCount > 4096u ||
        program.maximumStack == 0u ||
        program.maximumStack > 256u ||
        (program.flags & ~allowedProgramFlags) != 0u ||
        program.firstInstruction > w.instructions.size() ||
        program.instructionCount >
            w.instructions.size() - program.firstInstruction)
        return false;

    std::array<bool, NM_EXPRESSION_CACHE_CAPACITY> cacheWritten{};
    std::size_t depth = 0u;
    std::size_t maximumDepth = 0u;
    const auto pop = [&](const std::size_t count) {
        if (depth < count) return false;
        depth -= count;
        return true;
    };
    const auto push = [&]() {
        ++depth;
        maximumDepth = std::max(maximumDepth, depth);
        return maximumDepth <= program.maximumStack;
    };
    const bool localCache =
        (program.flags & NM_SCALAR_PROGRAM_FLAG_LOCAL_CACHE) != 0u;

    for (std::uint32_t local = 0u;
         local < program.instructionCount; ++local) {
        const auto& instruction =
            w.instructions[program.firstInstruction + local];
        switch (instruction.opcode) {
        case NM_EXPR_CONSTANT:
        case NM_EXPR_DT:
        case NM_EXPR_TEMPERATURE:
            if (!push()) return false;
            break;
        case NM_EXPR_PARAMETER:
            if (instruction.index >= material.parameterCount || !push())
                return false;
            break;
        case NM_EXPR_F:
        case NM_EXPR_RATE:
            if (instruction.index >= 9u || !push()) return false;
            break;
        case NM_EXPR_DF:
            if (!allowDirection || instruction.index >= 9u || !push())
                return false;
            break;
        case NM_EXPR_STATE:
        case NM_EXPR_NEXT_STATE:
            return false;
        case NM_EXPR_ADD:
        case NM_EXPR_SUBTRACT:
        case NM_EXPR_MULTIPLY:
        case NM_EXPR_DIVIDE:
        case NM_EXPR_MIN:
        case NM_EXPR_MAX:
            if (!pop(2u) || !push()) return false;
            break;
        case NM_EXPR_NEGATE:
        case NM_EXPR_LOG:
        case NM_EXPR_EXP:
        case NM_EXPR_SQRT:
        case NM_EXPR_ABS:
        case NM_EXPR_POW_INTEGER:
        case NM_EXPR_EXPM1_MINUS_X:
            if (!pop(1u) || !push()) return false;
            break;
        case NM_EXPR_CLAMP:
        case NM_EXPR_SELECT_NONNEGATIVE:
            if (!pop(3u) || !push()) return false;
            break;
        case NM_EXPR_FIBER_EXP_LINEAR:
            if (!pop(5u) || !push()) return false;
            break;
        case NM_EXPR_CACHE_STORE:
            if (!localCache || depth == 0u ||
                instruction.index >= NM_EXPRESSION_CACHE_CAPACITY)
                return false;
            cacheWritten[instruction.index] = true;
            break;
        case NM_EXPR_CACHE_LOAD:
            if (!localCache ||
                instruction.index >= NM_EXPRESSION_CACHE_CAPACITY ||
                !cacheWritten[instruction.index] || !push())
                return false;
            break;
        default:
            return false;
        }
    }
    return depth == 1u;
}

template<class World>
static bool nmNativeStatelessViscousMaterialEligible(
    const World& w,
    const NMMaterialGPU& material
) {
    constexpr std::uint32_t unsupportedMaterialFlags =
        NM_MATERIAL_HAS_STATE | NM_MATERIAL_HAS_IMPLICIT_STATE |
        NM_MATERIAL_PAPER_HILL_SCALAR_INITIALIZER |
        NM_MATERIAL_PAPER_COMPENSATED_ARITHMETIC |
        NM_MATERIAL_PAPER_HILL_ACCURATE_PROJECTION |
        NM_MATERIAL_GENERIC_COMPENSATED_IMPLICIT_PROJECTION |
        NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION |
        NM_MATERIAL_GENERIC_BILINEAR15_STRICT_STATE_CERTIFICATION |
        NM_MATERIAL_CANONICAL_COMPRESSIBLE_NEOHOOKEAN_TANGENT |
        NM_MATERIAL_SHARED_15_STATE_TANGENT_FACTORIZATION |
        NM_MATERIAL_HAS_TANGENT_BUNDLE;
    if ((material.flags & NM_MATERIAL_HAS_DISSIPATION) == 0u ||
        (material.flags & unsupportedMaterialFlags) != 0u ||
        material.constitutiveKind != NM_CONSTITUTIVE_BYTECODE ||
        material.projectionKind != NM_MATERIAL_PROJECTION_GENERIC ||
        material.stateCount != 0u ||
        material.stressProgramOffset == NM_INVALID_INDEX ||
        material.tangentProgramOffset == NM_INVALID_INDEX ||
        material.viscousStressProgramOffset == NM_INVALID_INDEX ||
        material.viscousTangentProgramOffset == NM_INVALID_INDEX)
        return false;

    for (std::uint32_t component = 0u; component < 9u; ++component) {
        if (!nmNativeViscousFloatProgramEligible(
                w, material, material.stressProgramOffset + component, false) ||
            !nmNativeViscousFloatProgramEligible(
                w, material, material.tangentProgramOffset + component, true) ||
            !nmNativeViscousFloatProgramEligible(
                w, material, material.viscousStressProgramOffset + component,
                false) ||
            !nmNativeViscousFloatProgramEligible(
                w, material, material.viscousTangentProgramOffset + component,
                true))
            return false;
    }
    return true;
}

template<class World>
static bool nmNativeStatelessBaseMaterialEligible(
    const World& w,
    const NMMaterialGPU& material
) {
    constexpr std::uint32_t unsupportedMaterialFlags =
        NM_MATERIAL_HAS_STATE | NM_MATERIAL_HAS_IMPLICIT_STATE |
        NM_MATERIAL_PAPER_HILL_SCALAR_INITIALIZER |
        NM_MATERIAL_PAPER_COMPENSATED_ARITHMETIC |
        NM_MATERIAL_PAPER_HILL_ACCURATE_PROJECTION |
        NM_MATERIAL_GENERIC_COMPENSATED_IMPLICIT_PROJECTION |
        NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION |
        NM_MATERIAL_GENERIC_BILINEAR15_STRICT_STATE_CERTIFICATION |
        NM_MATERIAL_CANONICAL_COMPRESSIBLE_NEOHOOKEAN_TANGENT |
        NM_MATERIAL_SHARED_15_STATE_TANGENT_FACTORIZATION |
        NM_MATERIAL_HAS_TANGENT_BUNDLE;
    if (material.constitutiveKind != NM_CONSTITUTIVE_NEO_HOOKEAN ||
        material.projectionKind != NM_MATERIAL_PROJECTION_GENERIC ||
        material.stateCount != 0u ||
        (material.flags & unsupportedMaterialFlags) != 0u ||
        (material.flags & NM_MATERIAL_HAS_DISSIPATION) != 0u ||
        material.stressProgramOffset == NM_INVALID_INDEX ||
        material.tangentProgramOffset == NM_INVALID_INDEX ||
        material.dissipationProgram != NM_INVALID_INDEX ||
        material.viscousStressProgramOffset != NM_INVALID_INDEX ||
        material.viscousTangentProgramOffset != NM_INVALID_INDEX)
        return false;

    for (std::uint32_t component = 0u; component < 9u; ++component) {
        if (!nmNativeViscousFloatProgramEligible(
                w, material, material.stressProgramOffset + component, false) ||
            !nmNativeViscousFloatProgramEligible(
                w, material, material.tangentProgramOffset + component, true))
            return false;
    }
    return true;
}

template<class World>
static bool nmNativeStatelessTangentBundleMaterialEligible(
    const World& w,
    const NMMaterialGPU& material
) {
    constexpr std::uint32_t unsupportedMaterialFlags =
        NM_MATERIAL_HAS_STATE | NM_MATERIAL_HAS_IMPLICIT_STATE |
        NM_MATERIAL_HAS_DISSIPATION |
        NM_MATERIAL_PAPER_HILL_SCALAR_INITIALIZER |
        NM_MATERIAL_PAPER_COMPENSATED_ARITHMETIC |
        NM_MATERIAL_PAPER_HILL_ACCURATE_PROJECTION |
        NM_MATERIAL_GENERIC_COMPENSATED_IMPLICIT_PROJECTION |
        NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION |
        NM_MATERIAL_GENERIC_BILINEAR15_STRICT_STATE_CERTIFICATION |
        NM_MATERIAL_SHARED_15_STATE_TANGENT_FACTORIZATION;
    if ((material.flags & NM_MATERIAL_HAS_TANGENT_BUNDLE) == 0u ||
        (material.flags & unsupportedMaterialFlags) != 0u ||
        material.constitutiveKind != NM_CONSTITUTIVE_BYTECODE ||
        material.projectionKind != NM_MATERIAL_PROJECTION_GENERIC ||
        material.stateCount != 0u ||
        material.tangentProgramOffset == NM_INVALID_INDEX ||
        material.stressProgramOffset == NM_INVALID_INDEX ||
        material.dissipationProgram != NM_INVALID_INDEX ||
        material.viscousStressProgramOffset != NM_INVALID_INDEX ||
        material.viscousTangentProgramOffset != NM_INVALID_INDEX)
        return false;
    for (std::uint32_t component = 0u; component < 9u; ++component) {
        if (!nmNativeViscousFloatProgramEligible(
                w, material, material.stressProgramOffset + component, false))
            return false;
    }

    std::size_t cursor =
        static_cast<std::size_t>(material.tangentProgramOffset) + 9u;
    std::size_t preludeCount = 0u;
    while (cursor < w.scalarPrograms.size() &&
           preludeCount < NM_TANGENT_BUNDLE_CACHE_CAPACITY) {
        const auto flags = w.scalarPrograms[cursor].flags;
        if ((flags & (NM_SCALAR_PROGRAM_FLAG_TANGENT_BUNDLE_CACHE |
                      NM_SCALAR_PROGRAM_FLAG_TANGENT_BUNDLE_PRELUDE)) !=
            (NM_SCALAR_PROGRAM_FLAG_TANGENT_BUNDLE_CACHE |
             NM_SCALAR_PROGRAM_FLAG_TANGENT_BUNDLE_PRELUDE))
            break;
        if ((flags & NM_SCALAR_PROGRAM_FLAG_COMPENSATED) != 0u)
            return false;
        ++cursor;
        ++preludeCount;
    }
    if (preludeCount == 0u ||
        cursor > w.scalarPrograms.size() ||
        NM_TANGENT_BUNDLE_OUTPUT_COUNT >
            w.scalarPrograms.size() - cursor)
        return false;
    for (std::size_t output = 0u;
         output < NM_TANGENT_BUNDLE_OUTPUT_COUNT; ++output) {
        const auto flags = w.scalarPrograms[cursor + output].flags;
        if ((flags & NM_SCALAR_PROGRAM_FLAG_TANGENT_BUNDLE_CACHE) == 0u ||
            (flags & (NM_SCALAR_PROGRAM_FLAG_TANGENT_BUNDLE_PRELUDE |
                      NM_SCALAR_PROGRAM_FLAG_COMPENSATED)) != 0u)
            return false;
    }
    return true;
}

template<class World>
static bool nmNativeStatelessDirectFloatMaterialEligible(
    const World& w,
    const NMMaterialGPU& material
) {
    return nmNativeStatelessTangentBundleMaterialEligible(w, material) ||
        nmNativeStatelessViscousMaterialEligible(w, material) ||
        nmNativeStatelessBaseMaterialEligible(w, material);
}

template<class World>
static std::uint64_t nmNativeTangentHash(const World& w) {
    std::uint64_t h=1469598103934665603ull;
    const auto add=[&](const auto& rows) {
        const auto* p=reinterpret_cast<const unsigned char*>(rows.data());
        for(std::size_t i=0;i<rows.size()*sizeof(rows[0]);++i)
            h=(h^p[i])*1099511628211ull;
    };
    add(w.materials);add(w.scalarPrograms);add(w.instructions);
    return h;
}

template<class World>
static bool nmWriteNativeTangentSource(const World& w,const char* path,std::string& error) {
    std::ostringstream functions,cases,preludeCases,columnCases,stressCases;
    std::ostringstream viscousStressCases,viscousTangentCases;
    std::ostringstream bundleVectorCases,baseOnlyStressCases,baseOnlyTangentCases;
    std::set<std::uint32_t> emitted,directEmitted;
    std::map<std::pair<bool,std::uint32_t>,std::uint32_t> canonicalProgram;
    std::map<std::string,std::uint32_t> canonicalBySignature[2];
    const auto emit=[&](std::uint32_t pi,const bool direct=false) {
        auto& emittedPrograms=direct?directEmitted:emitted;
        if(!emittedPrograms.insert(pi).second)return;
        if(pi>=w.scalarPrograms.size())throw std::runtime_error("native program index");
        const auto& p=w.scalarPrograms[pi];
        if((p.flags&NM_SCALAR_PROGRAM_FLAG_COMPENSATED)!=0u)
            throw std::runtime_error("compensated native program unsupported");
        // Material programs often repeat byte-for-byte across regional copies.
        // Deduplicate only identical instruction records and scalar metadata.
        std::string signature;
        const auto appendWord=[&](const std::uint32_t value) {
            signature.append(reinterpret_cast<const char*>(&value),sizeof(value));
        };
        appendWord(direct ? 1u : 0u);
        appendWord(p.instructionCount);
        appendWord(p.maximumStack);
        appendWord(p.flags);
        for(std::uint32_t j=0u;j<p.instructionCount;++j) {
            const auto& instruction=w.instructions.at(p.firstInstruction+j);
            appendWord(instruction.opcode);
            appendWord(instruction.index);
            appendWord(static_cast<std::uint32_t>(instruction.integer));
            appendWord(instruction.reserved);
            for(const float value : {instruction.immediate.x,
                                     instruction.immediate.y,
                                     instruction.immediate.z,
                                     instruction.immediate.w}) {
                std::uint32_t bits=0u;
                std::memcpy(&bits,&value,sizeof(bits));
                appendWord(bits);
            }
        }
        auto& signatures=canonicalBySignature[direct?1u:0u];
        const auto inserted=signatures.emplace(signature,pi);
        canonicalProgram[{direct,pi}]=inserted.first->second;
        if(!inserted.second)return;
        const std::string symbolPrefix=direct?"nmNativeDirectP":"nmNativeP";
        functions<<"template <typename CachePointer>\ninline float "<<symbolPrefix<<pi<<"(device const float* p, thread const float* f, thread const float* d, thread const float* rate, float dt, float temp, CachePointer c, thread bool& valid, thread bool& nonfinite) {\n#pragma clang fp contract(off)\nif(!(dt>=0.0f)||!isfinite(dt)||!isfinite(temp)){valid=false;return 0.0f;}\n";
        std::vector<std::string> st;unsigned serial=0;
        auto pop=[&](){if(st.empty())throw std::runtime_error("native stack underflow");auto v=st.back();st.pop_back();return v;};
        auto push=[&](std::string expr){
            const auto v="v"+std::to_string(serial++);
            // The scalar VM checks every pushed intermediate, not only the
            // final result. Preserve that rejection behavior in the direct
            // lowering as well as the legacy tangent-bundle lowering.
            functions<<"const float "<<v<<" = "<<expr<<";";
            functions<<" if(!isfinite("<<v<<")) {nonfinite=true;valid=false;return 0.0f;}";
            functions<<"\n";
            st.push_back(v);
        };
        for(std::uint32_t j=0;j<p.instructionCount;++j) {
            const auto& i=w.instructions.at(p.firstInstruction+j);
            auto idx=std::to_string(i.index);
            switch(i.opcode) {
            case NM_EXPR_CONSTANT: {
                std::uint32_t bits;float val=i.immediate.x;std::memcpy(&bits,&val,4);
                push("as_type<float>("+std::to_string(bits)+"u)");break;
            }
            case NM_EXPR_PARAMETER:push("p["+idx+"]");break;
            case NM_EXPR_F:push("f["+idx+"]");break;
            case NM_EXPR_DF:push("d["+idx+"]");break;
            case NM_EXPR_DT:push("dt");break;
            case NM_EXPR_TEMPERATURE:push("temp");break;
            case NM_EXPR_RATE:push("rate["+idx+"]");break;
            case NM_EXPR_CACHE_STORE:
                if(st.empty()||i.index>=NM_EXPRESSION_CACHE_CAPACITY)throw std::runtime_error("native cache store");
                functions<<"c["<<idx<<"]="<<st.back()<<";\n";break;
            case NM_EXPR_CACHE_LOAD:
                if(i.index>=NM_EXPRESSION_CACHE_CAPACITY)throw std::runtime_error("native cache load");
                push("c["+idx+"]");break;
            case NM_EXPR_ADD:case NM_EXPR_SUBTRACT:case NM_EXPR_MULTIPLY:case NM_EXPR_DIVIDE:case NM_EXPR_MIN:case NM_EXPR_MAX:{
                auto b=pop(),a=pop();
                if(i.opcode==NM_EXPR_DIVIDE)functions<<"if(abs("<<b<<")<=kGeometryFloor){valid=false;return 0.0f;}\n";
                if(i.opcode==NM_EXPR_MIN||i.opcode==NM_EXPR_MAX)push(std::string(i.opcode==NM_EXPR_MIN?"min(":"max(")+a+","+b+")");
                else push(a+(i.opcode==NM_EXPR_ADD?"+":i.opcode==NM_EXPR_SUBTRACT?"-":i.opcode==NM_EXPR_MULTIPLY?"*":"/")+b);
                break;
            }
            case NM_EXPR_NEGATE:{auto a=pop();push("-"+a);break;}
            case NM_EXPR_LOG:case NM_EXPR_EXP:case NM_EXPR_SQRT:case NM_EXPR_ABS:case NM_EXPR_EXPM1_MINUS_X:{
                auto a=pop();
                if(i.opcode==NM_EXPR_LOG) {
                    if(direct)functions<<"if(!("<<a<<">0.0f)){valid=false;return 0.0f;}\n";
                    else functions<<"if("<<a<<"<=0.0f){valid=false;return 0.0f;}\n";
                }
                if(i.opcode==NM_EXPR_SQRT)functions<<"if("<<a<<"<0.0f){valid=false;return 0.0f;}\n";
                push(i.opcode==NM_EXPR_EXPM1_MINUS_X?"(exp("+a+")-1.0f)-"+a:std::string(i.opcode==NM_EXPR_LOG?"log(":i.opcode==NM_EXPR_EXP?"exp(":i.opcode==NM_EXPR_SQRT?"sqrt(":"abs(")+a+")");break;
            }
            case NM_EXPR_POW_INTEGER:{
                auto a=pop();const auto v="pow"+std::to_string(serial++);
                if(i.integer<0)functions<<"if(abs("<<a<<")<=kGeometryFloor){valid=false;return 0.0f;}\n";
                functions<<"float "<<v<<"=1.0f; {float factor="<<a<<"; uint power="<<std::uint32_t(i.integer<0?-std::int64_t(i.integer):i.integer)<<"u; while(power!=0u){if((power&1u)!=0u)"<<v<<"*=factor;factor*=factor;power>>=1u;}}\n";
                push(i.integer<0?"1.0f/"+v:v);break;
            }
            case NM_EXPR_CLAMP:{
                auto hi=pop(),lo=pop(),a=pop();
                functions<<"if("<<hi<<"<"<<lo<<"){valid=false;return 0.0f;}\n";
                push("clamp("+a+","+lo+","+hi+")");break;
            }
            case NM_EXPR_SELECT_NONNEGATIVE:{
                auto neg=pop(),pos=pop(),cond=pop();push("("+cond+">=0.0f?"+pos+":"+neg+")");break;
            }
            case NM_EXPR_FIBER_EXP_LINEAR:{
                auto limit=pop(),c5=pop(),c4=pop(),c3=pop(),a=pop();
                push("numi_matter_fiber::evaluate<float>("+a+","+c3+","+c4+","+c5+","+limit+","+std::to_string(i.integer)+"u,valid)");
                functions<<"if(!valid)return 0.0f;\n";break;
            }
            default:throw std::runtime_error("unsupported native material opcode "+std::to_string(i.opcode));
            }
        }
        if(st.size()!=1)throw std::runtime_error("native stack final");
        if(direct)
            functions<<"if(!isfinite("<<st.back()<<")){nonfinite=true;valid=false;return 0.0f;} return "<<st.back()<<"; }\n";
        else
            functions<<"return "<<st.back()<<"; }\n";
    };
    const auto nativeProgramSymbol=[&](const std::uint32_t pi,const bool direct=false) {
        const auto found=canonicalProgram.find({direct,pi});
        if(found==canonicalProgram.end())
            throw std::runtime_error("native program symbol was not emitted");
        return std::string(direct?"nmNativeDirectP":"nmNativeP")+
            std::to_string(found->second);
    };
    const auto groupAdjacentIdenticalCases=[](const std::string& input) {
        struct Group { std::vector<std::string> labels; std::string body; };
        std::vector<Group> groups;
        std::size_t cursor=0u;
        while(cursor<input.size()) {
            while(cursor<input.size() &&
                  (input[cursor]=='\n' || input[cursor]=='\r' ||
                   input[cursor]==' ' || input[cursor]=='\t'))
                ++cursor;
            if(cursor==input.size())break;
            if(input.compare(cursor,5u,"case ")!=0)
                throw std::runtime_error("invalid generated native case stream");
            const std::size_t colon=input.find(':',cursor+5u);
            if(colon==std::string::npos)
                throw std::runtime_error("unterminated generated native case label");
            const std::string label=input.substr(cursor+5u,colon-(cursor+5u));
            std::size_t open=colon+1u;
            while(open<input.size() && (input[open]==' ' || input[open]=='\t'))
                ++open;
            if(open==input.size() || input[open]!='{')
                throw std::runtime_error("generated native case body is not braced");
            int depth=0;
            std::size_t close=open;
            for(;close<input.size();++close) {
                if(input[close]=='{')++depth;
                else if(input[close]=='}' && --depth==0)break;
            }
            if(close==input.size() || depth!=0)
                throw std::runtime_error("unterminated generated native case body");
            const std::string body=input.substr(open+1u,close-open-1u);
            if(!groups.empty() && groups.back().body==body)
                groups.back().labels.push_back(label);
            else
                groups.push_back(Group{{label},body});
            cursor=close+1u;
        }
        std::ostringstream output;
        for(const auto& group:groups) {
            for(const auto& label:group.labels)
                output<<"case "<<label<<": ";
            output<<"{"<<group.body<<"}\n";
        }
        return output.str();
    };
    try {
        std::size_t eligible=0;
        for(const auto& m:w.materials) {
            if((m.flags&NM_MATERIAL_HAS_TANGENT_BUNDLE)==0u)continue;
            stressCases<<"case "<<m.stressProgramOffset<<"u: {\n";
            for(unsigned k=0;k<9;++k) {
                emit(m.stressProgramOffset+k);
                stressCases<<"{float c[NM_EXPRESSION_CACHE_CAPACITY]={0}; out["<<k<<"u]="<<nativeProgramSymbol(m.stressProgramOffset+k)<<"(parameters,f,d,rate,dt,temp,c,valid,nonfinite);if(!valid)return true;}\n";
            }
            stressCases<<"return true;}\n";
            std::uint32_t pi=m.tangentProgramOffset+9u;
            preludeCases<<"case "<<m.tangentProgramOffset<<"u: {\n";
            while(pi<w.scalarPrograms.size()&&(w.scalarPrograms[pi].flags&NM_SCALAR_PROGRAM_FLAG_TANGENT_BUNDLE_PRELUDE)!=0u) {
                emit(pi);
                preludeCases<<"(void)"<<nativeProgramSymbol(pi)<<"(p,f,d,rate,dt,temp,c,valid,nonfinite);if(!valid)return true;\n";
                ++pi;
            }
            for(unsigned k=0;k<9;++k)emit(pi+k);
            bundleVectorCases<<"case "<<m.tangentProgramOffset<<"u: {\n"
                << "if(!evaluateNativeStatelessTangentBundlePrelude(material,parameters,f,rate,dt,temp,c,valid,nonfinite))return false;if(!valid)return true;\n";
            for(unsigned k=0;k<9;++k)
                bundleVectorCases<<"out["<<k<<"u]="<<nativeProgramSymbol(pi+k)
                    <<"(p,f,d,rate,dt,temp,c,valid,nonfinite);if(!valid)return true;\n";
            bundleVectorCases<<"return true;}\n";
            cases<<"case "<<m.tangentProgramOffset<<"u: {\n"
                <<"if(!evaluateNativeStatelessTangentBundlePrelude(material,parameters,f,rate,dt,temp,c,valid,nonfinite))return false;if(!valid)return true;\n"
                <<"float columnValues[9];\nfor(uint column=0u;column<9u;++column){\n"
                <<"if(!evaluateNativeStatelessTangentBundleColumn(material,parameters,f,rate,dt,temp,column,c,columnValues,valid,nonfinite))return false;if(!valid)return true;\n"
                <<"for(uint row=0u;row<9u;++row)out[row*9u+column]=columnValues[row];\n"
                <<"} return true; }\n";
            preludeCases<<"return true;}\n";
            columnCases<<"case "<<m.tangentProgramOffset<<"u: {\nfor(uint k=0u;k<9u;++k)d[k]=(k==column)?1.0f:0.0f;\n";
            for(unsigned k=0;k<9;++k)columnCases<<"out["<<k<<"u]="<<nativeProgramSymbol(pi+k)<<"(p,f,d,rate,dt,temp,c,valid,nonfinite);if(!valid)return true;\n";
            columnCases<<"return true;}\n";++eligible;
        }
        std::size_t viscousEligible=0u;
        for(const auto& m:w.materials) {
            if(!nmNativeStatelessViscousMaterialEligible(w,m))continue;
            viscousStressCases<<"case "<<m.stressProgramOffset<<"u: {\n";
            viscousTangentCases<<"case "<<m.tangentProgramOffset<<"u: {\n";
            for(unsigned k=0;k<9;++k) {
                const std::uint32_t baseStress=m.stressProgramOffset+k;
                const std::uint32_t viscousStress=m.viscousStressProgramOffset+k;
                const std::uint32_t baseTangent=m.tangentProgramOffset+k;
                const std::uint32_t viscousTangent=m.viscousTangentProgramOffset+k;
                emit(baseStress,true);
                emit(viscousStress,true);
                emit(baseTangent,true);
                emit(viscousTangent,true);
                viscousStressCases
                    <<"{float c[NM_EXPRESSION_CACHE_CAPACITY]={0};out["<<k
                    <<"u]="<<nativeProgramSymbol(baseStress,true)
                    <<"(p,f,d,rate,dt,temp,c,valid,nonfinite);if(!valid)return true;}\n"
                    <<"{float c[NM_EXPRESSION_CACHE_CAPACITY]={0};float v="
                    <<nativeProgramSymbol(viscousStress,true)
                    <<"(p,f,d,rate,dt,temp,c,valid,nonfinite);if(!valid)return true;out["
                    <<k<<"u]+=v;}\n";
                viscousTangentCases
                    <<"{float c[NM_EXPRESSION_CACHE_CAPACITY]={0};out["<<k
                    <<"u]="<<nativeProgramSymbol(baseTangent,true)
                    <<"(p,f,deformationDirection,rate,dt,temp,c,valid,nonfinite);if(!valid)return true;}\n"
                    <<"{float c[NM_EXPRESSION_CACHE_CAPACITY]={0};float v="
                    <<nativeProgramSymbol(viscousTangent,true)
                    <<"(p,f,rateDirection,rate,dt,temp,c,valid,nonfinite);if(!valid)return true;out["
                    <<k<<"u]+=v;}\n";
            }
            viscousStressCases<<"return true;}\n";
            viscousTangentCases<<"return true;}\n";
            ++viscousEligible;
        }
        std::size_t baseOnlyEligible=0u;
        for(const auto& m:w.materials) {
            if(!nmNativeStatelessBaseMaterialEligible(w,m))continue;
            baseOnlyStressCases<<"case "<<m.stressProgramOffset<<"u: {\n";
            baseOnlyTangentCases<<"case "<<m.tangentProgramOffset<<"u: {\n";
            for(unsigned k=0;k<9;++k) {
                const std::uint32_t stress=m.stressProgramOffset+k;
                const std::uint32_t tangent=m.tangentProgramOffset+k;
                emit(stress,true);
                emit(tangent,true);
                baseOnlyStressCases<<"{float c[NM_EXPRESSION_CACHE_CAPACITY]={0};out["
                    <<k<<"u]="<<nativeProgramSymbol(stress,true)
                    <<"(p,f,d,rate,dt,temp,c,valid,nonfinite);if(!valid)return true;}\n";
                baseOnlyTangentCases<<"{float c[NM_EXPRESSION_CACHE_CAPACITY]={0};out["
                    <<k<<"u]="<<nativeProgramSymbol(tangent,true)
                    <<"(p,f,deformationDirection,rate,dt,temp,c,valid,nonfinite);if(!valid)return true;}\n";
            }
            baseOnlyStressCases<<"return true;}\n";
            baseOnlyTangentCases<<"return true;}\n";
            ++baseOnlyEligible;
        }
        if(eligible==0u&&viscousEligible==0u&&baseOnlyEligible==0u)
            throw std::runtime_error("no eligible stateless tangent or direct-float materials");
        std::ostringstream out;
        out<<"// Generated exact scalar-operation lowering. Bound to compiled material/program bytes.\n#ifndef __METAL_VERSION__\nstatic constexpr std::uint64_t nmNativeTangentSourceHash="<<nmNativeTangentHash(w)<<"ull;\n#else\nconstant bool nmNativeMaterialEnabled [[function_constant(31)]];\nconstant bool nmNativeMaterialActive=is_function_constant_defined(nmNativeMaterialEnabled)?nmNativeMaterialEnabled:false;\nconstant uint nmNativeTangentProgramOffsetFunctionConstant [[function_constant(35)]];\nconstant uint nmNativeTangentProgramOffsetSelector=is_function_constant_defined(nmNativeTangentProgramOffsetFunctionConstant)?nmNativeTangentProgramOffsetFunctionConstant:NM_INVALID_INDEX;\nconstant uint nmNativeStressProgramOffsetFunctionConstant [[function_constant(36)]];\nconstant uint nmNativeStressProgramOffsetSelector=is_function_constant_defined(nmNativeStressProgramOffsetFunctionConstant)?nmNativeStressProgramOffsetFunctionConstant:NM_INVALID_INDEX;\n";
        if(viscousEligible>0u)
            out<<"#define NM_NATIVE_STATELESS_VISCOUS_HELPERS_AVAILABLE 1\n";
        if(baseOnlyEligible>0u)
            out<<"#define NM_NATIVE_STATELESS_BASE_ONLY_HELPERS_AVAILABLE 1\n";
        out<<functions.str();
        out<<"template <typename CachePointer>\ninline bool evaluateNativeStatelessTangentBundlePrelude(const NMMaterialGPU material, device const float* parameters, thread const float* f, thread const float* rate, float dt, float temp, CachePointer c, thread bool& valid, thread bool& nonfinite) {\nif(!nmNativeMaterialActive || (material.flags&(1u<<12u))==0u)return false;\ndevice const float* p=parameters;float d[9]={0};const uint selectedOffset=nmNativeTangentProgramOffsetSelector;if(selectedOffset!=NM_INVALID_INDEX && material.tangentProgramOffset!=selectedOffset)return false;switch(selectedOffset!=NM_INVALID_INDEX?selectedOffset:material.tangentProgramOffset){\n"<<groupAdjacentIdenticalCases(preludeCases.str())<<"default:return false;} }\n";
        out<<"template <typename CachePointer>\ninline bool evaluateNativeStatelessTangentBundleColumn(const NMMaterialGPU material, device const float* parameters, thread const float* f, thread const float* rate, float dt, float temp, uint column, CachePointer c, thread float* out, thread bool& valid, thread bool& nonfinite) {\nif(!nmNativeMaterialActive || (material.flags&(1u<<12u))==0u)return false;if(column>=9u||out==nullptr){valid=false;return true;}\ndevice const float* p=parameters;float d[9]={0};const uint selectedOffset=nmNativeTangentProgramOffsetSelector;if(selectedOffset!=NM_INVALID_INDEX && material.tangentProgramOffset!=selectedOffset)return false;switch(selectedOffset!=NM_INVALID_INDEX?selectedOffset:material.tangentProgramOffset){\n"<<groupAdjacentIdenticalCases(columnCases.str())<<"default:return false;} }\n";
        out<<"inline bool evaluateNativeStatelessTangentBundle(const NMMaterialGPU material, device const float* parameters, thread const float* f, thread const float* rate, float dt, float temp, thread float* out, thread bool& valid, thread bool& nonfinite) {\nif(!nmNativeMaterialActive || (material.flags&(1u<<12u))==0u)return false;if(out==nullptr){valid=false;return true;}\nfloat c[NM_TANGENT_BUNDLE_CACHE_CAPACITY]={0};const uint selectedOffset=nmNativeTangentProgramOffsetSelector;if(selectedOffset!=NM_INVALID_INDEX && material.tangentProgramOffset!=selectedOffset)return false;switch(selectedOffset!=NM_INVALID_INDEX?selectedOffset:material.tangentProgramOffset){\n"<<groupAdjacentIdenticalCases(cases.str())<<"default:return false;} }\n";
        out<<"inline bool evaluateNativeStatelessTangentBundleVector(const NMMaterialGPU material, device const float* parameters, thread const float* f, thread const float* d, thread const float* rate, float dt, float temp, thread float* out, thread bool& valid, thread bool& nonfinite) {\nif(!nmNativeMaterialActive || (material.flags&(1u<<12u))==0u)return false;if(out==nullptr){valid=false;return true;}device const float* p=parameters;float c[NM_EXPRESSION_CACHE_CAPACITY]={0};const uint selectedOffset=nmNativeTangentProgramOffsetSelector;if(selectedOffset!=NM_INVALID_INDEX && material.tangentProgramOffset!=selectedOffset)return false;switch(selectedOffset!=NM_INVALID_INDEX?selectedOffset:material.tangentProgramOffset){\n"<<groupAdjacentIdenticalCases(bundleVectorCases.str())<<"default:return false;} }\n#define NM_NATIVE_STATELESS_TANGENT_BUNDLE_VECTOR_AVAILABLE 1\n";

        out<<"inline bool evaluateNativeStatelessStressBundle(const NMMaterialGPU material, device const float* parameters, thread const float* f, thread const float* rate, float dt, float temp, thread float* out, thread bool& valid, thread bool& nonfinite) {\nif(!nmNativeMaterialActive || (material.flags&(1u<<12u))==0u)return false; float d[9]={0};const uint selectedOffset=nmNativeStressProgramOffsetSelector;if(selectedOffset!=NM_INVALID_INDEX && material.stressProgramOffset!=selectedOffset)return false;switch(selectedOffset!=NM_INVALID_INDEX?selectedOffset:material.stressProgramOffset){\n"<<groupAdjacentIdenticalCases(stressCases.str())<<"default:return false;} }\n";
        if(baseOnlyEligible>0u) {
            out<<"inline bool nmNativeStatelessBaseOnlyMaterialActive(const NMMaterialGPU material) {\n"
                <<"constexpr uint blocked=NM_MATERIAL_HAS_STATE|NM_MATERIAL_HAS_IMPLICIT_STATE|NM_MATERIAL_PAPER_HILL_SCALAR_INITIALIZER|NM_MATERIAL_PAPER_COMPENSATED_ARITHMETIC|NM_MATERIAL_PAPER_HILL_ACCURATE_PROJECTION|NM_MATERIAL_GENERIC_COMPENSATED_IMPLICIT_PROJECTION|NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION|NM_MATERIAL_GENERIC_BILINEAR15_STRICT_STATE_CERTIFICATION|NM_MATERIAL_CANONICAL_COMPRESSIBLE_NEOHOOKEAN_TANGENT|NM_MATERIAL_SHARED_15_STATE_TANGENT_FACTORIZATION|NM_MATERIAL_HAS_TANGENT_BUNDLE;\n"
                <<"return nmNativeMaterialActive && (material.flags&(1u<<12u))!=0u && material.constitutiveKind==NM_CONSTITUTIVE_NEO_HOOKEAN && material.projectionKind==NM_MATERIAL_PROJECTION_GENERIC && material.stateCount==0u && (material.flags&blocked)==0u && (material.flags&NM_MATERIAL_HAS_DISSIPATION)==0u && material.stressProgramOffset!=NM_INVALID_INDEX && material.tangentProgramOffset!=NM_INVALID_INDEX && material.viscousStressProgramOffset==NM_INVALID_INDEX && material.viscousTangentProgramOffset==NM_INVALID_INDEX; }\n";
            out<<"inline bool evaluateNativeStatelessBaseOnlyStress(const NMMaterialGPU material, device const float* parameters, thread const float* f, thread const float* rate, float dt, float temp, thread float* out, thread bool& valid, thread bool& nonfinite) {\nif(!nmNativeStatelessBaseOnlyMaterialActive(material))return false;if(out==nullptr){valid=false;return true;}device const float* p=parameters;float d[9]={0};const uint selectedOffset=nmNativeStressProgramOffsetSelector;if(selectedOffset!=NM_INVALID_INDEX && material.stressProgramOffset!=selectedOffset)return false;switch(selectedOffset!=NM_INVALID_INDEX?selectedOffset:material.stressProgramOffset){\n"<<groupAdjacentIdenticalCases(baseOnlyStressCases.str())<<"default:return false;} }\n";
            out<<"inline bool evaluateNativeStatelessBaseOnlyTangentAction(const NMMaterialGPU material, device const float* parameters, thread const float* f, thread const float* deformationDirection, thread const float* rate, float dt, float temp, thread float* out, thread bool& valid, thread bool& nonfinite) {\nif(!nmNativeStatelessBaseOnlyMaterialActive(material))return false;if(out==nullptr){valid=false;return true;}device const float* p=parameters;const uint selectedOffset=nmNativeTangentProgramOffsetSelector;if(selectedOffset!=NM_INVALID_INDEX && material.tangentProgramOffset!=selectedOffset)return false;switch(selectedOffset!=NM_INVALID_INDEX?selectedOffset:material.tangentProgramOffset){\n"<<groupAdjacentIdenticalCases(baseOnlyTangentCases.str())<<"default:return false;} }\n";
        }
        out<<"inline bool nmNativeStatelessViscousMaterialActive(const NMMaterialGPU material) {\n";
        out<<"if(!nmNativeMaterialActive || (material.flags&(1u<<12u))==0u)return false;\n";
        out<<"constexpr uint blocked=NM_MATERIAL_HAS_STATE|NM_MATERIAL_HAS_IMPLICIT_STATE|NM_MATERIAL_PAPER_HILL_SCALAR_INITIALIZER|NM_MATERIAL_PAPER_COMPENSATED_ARITHMETIC|NM_MATERIAL_PAPER_HILL_ACCURATE_PROJECTION|NM_MATERIAL_GENERIC_COMPENSATED_IMPLICIT_PROJECTION|NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION|NM_MATERIAL_GENERIC_BILINEAR15_STRICT_STATE_CERTIFICATION|NM_MATERIAL_CANONICAL_COMPRESSIBLE_NEOHOOKEAN_TANGENT|NM_MATERIAL_SHARED_15_STATE_TANGENT_FACTORIZATION|NM_MATERIAL_HAS_TANGENT_BUNDLE;\n";
        out<<"return material.constitutiveKind==NM_CONSTITUTIVE_BYTECODE && material.projectionKind==NM_MATERIAL_PROJECTION_GENERIC && material.stateCount==0u && (material.flags&NM_MATERIAL_HAS_DISSIPATION)!=0u && (material.flags&blocked)==0u && material.viscousStressProgramOffset!=NM_INVALID_INDEX && material.viscousTangentProgramOffset!=NM_INVALID_INDEX; }\n";
        out<<"inline bool evaluateNativeStatelessViscousMaterialStress(const NMMaterialGPU material, device const float* parameters, thread const float* f, thread const float* rate, float dt, float temp, thread float* out, thread bool& valid, thread bool& nonfinite) {\nif(!nmNativeStatelessViscousMaterialActive(material))return false;if(out==nullptr){valid=false;return true;}device const float* p=parameters;float d[9]={0};const uint selectedOffset=nmNativeStressProgramOffsetSelector;if(selectedOffset!=NM_INVALID_INDEX && material.stressProgramOffset!=selectedOffset)return false;switch(selectedOffset!=NM_INVALID_INDEX?selectedOffset:material.stressProgramOffset){\n"<<groupAdjacentIdenticalCases(viscousStressCases.str())<<"default:return false;} }\n";
        out<<"inline bool evaluateNativeStatelessViscousMaterialTangentAction(const NMMaterialGPU material, device const float* parameters, thread const float* f, thread const float* deformationDirection, thread const float* rateDirection, thread const float* rate, float dt, float temp, thread float* out, thread bool& valid, thread bool& nonfinite) {\nif(!nmNativeStatelessViscousMaterialActive(material))return false;if(out==nullptr){valid=false;return true;}device const float* p=parameters;const uint selectedOffset=nmNativeTangentProgramOffsetSelector;if(selectedOffset!=NM_INVALID_INDEX && material.tangentProgramOffset!=selectedOffset)return false;switch(selectedOffset!=NM_INVALID_INDEX?selectedOffset:material.tangentProgramOffset){\n"<<groupAdjacentIdenticalCases(viscousTangentCases.str())<<"default:return false;} }\n#endif\n";
        const std::string generated = out.str();
        {
            std::ifstream previous(path, std::ios::binary);
            if (previous) {
                const std::string existing{
                    std::istreambuf_iterator<char>(previous),
                    std::istreambuf_iterator<char>()};
                if (previous.bad())
                    throw std::runtime_error("cannot read existing native source");
                if (existing == generated) return true;
            }
        }
        const std::string temporaryPath = std::string(path) + ".tmp";
        {
            std::ofstream temporary(
                temporaryPath, std::ios::binary | std::ios::trunc);
            if (!temporary)
                throw std::runtime_error("cannot open temporary native source");
            temporary.write(generated.data(),
                            static_cast<std::streamsize>(generated.size()));
            temporary.flush();
            if (!temporary) {
                std::remove(temporaryPath.c_str());
                throw std::runtime_error("native source write failed");
            }
        }
        if (std::rename(temporaryPath.c_str(), path) != 0) {
            const int renameError = errno;
            std::remove(temporaryPath.c_str());
            throw std::runtime_error(
                std::string("cannot atomically replace native source: ") +
                std::strerror(renameError));
        }
        return true;
    } catch(const std::exception& e){error=e.what();return false;}
}
