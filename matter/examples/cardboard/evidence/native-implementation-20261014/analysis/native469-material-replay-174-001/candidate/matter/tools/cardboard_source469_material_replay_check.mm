#include <map>
#include <optional>
#define main source169_capture_diagnostic_legacy_main
#include "cardboard_bilinear15_capture_diagnostic_check.mm"
#undef main

namespace {
struct Source469Point {
    std::array<Record,5> states{}; // center, then captured stages 1..4
    std::array<float,81> cachedTangent{};
    std::array<std::array<float,9>,5> capturedStress{};
    std::array<std::array<float,15>,5> capturedProjectedHigh{};
    std::array<std::array<float,15>,5> capturedProjectedLow{};
    std::array<std::array<float,4>,5> capturedProjection{};
};
struct Source469Capture {
    std::vector<Source469Point> env0;
    std::uint64_t fingerprint=0u;
    std::vector<float> parameters;
    std::array<double,2> epsilon{};
};
struct Sample {
    Record row;
    std::string label;
    std::string stateLabel;
    int stateIndex=-1;
    int column=-1;
    int sign=0;
    int repeatOf=-1;
    double epsilon=0.0;
    bool captureReference=false;
    std::array<double,9> requestedFDelta{};
    std::array<double,9> realizedFDelta{};
    std::array<double,9> requestedRateDelta{};
    std::array<double,9> realizedRateDelta{};
};
struct MPSelection { std::uint32_t tet=0u, material=0u, state=0u; std::string name; };
struct ReferenceRow {
    std::array<double,9> stress{};
    std::array<double,81> tangent{};
    std::optional<std::array<double,15>> mpState;
    std::optional<std::array<double,9>> mpStress;
    std::optional<std::array<double,81>> mpTangent;
};
using ReferenceKey=std::pair<std::uint32_t,std::uint32_t>; // (tet,state)
using ReferenceMap=std::map<ReferenceKey,ReferenceRow>;

std::uint64_t hex64(const std::string& text,const std::string& label) {
    std::string value=text;
    if(value.rfind("0x",0u)==0u||value.rfind("0X",0u)==0u)value.erase(0u,2u);
    need(!value.empty()&&value.size()<=16u,label+" is not a bounded hexadecimal uint64");
    std::size_t consumed=0u;const auto result=std::stoull(value,&consumed,16);
    need(consumed==value.size(),label+" contains non-hexadecimal suffix");return result;
}
std::array<float,9> matrix9(id value,const std::string& label) {
    return flatArray<9>(value,label);
}
std::vector<double> flatDoubleVector(id value,const std::size_t expected,const std::string& label) {
    need([value isKindOfClass:[NSArray class]],label+" is not an array");
    NSArray* array=(NSArray*)value;need(array.count==expected,label+" has wrong length");
    std::vector<double> result;result.reserve(expected);
    for(id item in array)result.push_back(number(item,label));
    return result;
}
std::array<float,15> state15OrZero(id value,const std::uint32_t count,const std::string& label) {
    std::array<float,15> result{};
    if(count==0u) {
        if(value!=nil&&[value isKindOfClass:[NSArray class]])
            need(((NSArray*)value).count==0u,label+" unexpectedly contains state for a stateless material");
        return result;
    }
    need(count==15u,label+" is not the source-fit 15-state material");
    const auto state=flatVector(value,15u,label);std::copy(state.begin(),state.end(),result.begin());
    return result;
}
Record recordFromCenter(NSDictionary* row,NSDictionary* center,const CompiledWorld& world) {
    Record result;
    result.env=static_cast<std::uint32_t>(number(field(row,"environment"),"capture environment"));
    result.tet=static_cast<std::uint32_t>(number(field(row,"tet_id"),"capture tet"));
    result.material=static_cast<std::uint32_t>(number(field(row,"material_index"),"capture material"));
    result.materialName=stringValue(field(row,"material_name"),"capture material name");
    need(result.env==0u&&result.tet<1920u&&result.material<3u,
         "source469 selected row falls outside env0's compiled 1920-tet layout");
    const auto F=matrix9(field(center,"F_local"),"center F_local");
    const auto FLow=matrix9(field(center,"F_local_low"),"center F_local_low");
    const auto FDot=matrix9(field(center,"Fdot_local_per_s"),"center Fdot_local_per_s");
    const auto H=matrix9(field(center,"F_direction_local_per_s"),"center F direction");
    const auto HRate=matrix9(field(center,"rate_direction_local_per_s2"),"center rate direction");
    std::copy(F.begin(),F.end(),result.input.begin());
    std::copy(FLow.begin(),FLow.end(),result.input.begin()+9u);
    std::copy(FDot.begin(),FDot.end(),result.input.begin()+18u);
    std::copy(H.begin(),H.end(),result.input.begin()+27u);
    std::copy(HRate.begin(),HRate.end(),result.input.begin()+36u);
    NSArray* dtTemp=(NSArray*)field(center,"dt_seconds_bulk_modulus_kind_frame_active");
    need(dtTemp.count==4u,"source469 center dt/temperature metadata shape changed");
    result.input[45]=static_cast<float>(number(dtTemp[0],"center dt"));
    result.input[46]=static_cast<float>(number(dtTemp[1],"center temperature"));
    need(result.input[45]>0.0f&&result.input[46]>0.0f,
         "source469 center dt/temperature is nonpositive");
    need(result.material<world.materials.size()&&
         std::bit_cast<std::uint32_t>(result.input[46])==
         std::bit_cast<std::uint32_t>(world.materials[result.material].bulk.y),
         "captured source469 temperature word differs from compiled material descriptor");
    const auto stateCount=static_cast<std::uint32_t>(number(
        field(row,"material_state_count"),"material state count"));
    result.accepted=state15OrZero(field(center,"accepted_state"),stateCount,"center accepted history");
    return result;
}
Record recordFromSide(const Record& center,const NSDictionary* side,const unsigned expectedStage,
                      const double expectedEpsilon) {
    need(number(field((NSDictionary*)side,"stage"),"captured side stage")==double(expectedStage),
         "source469 side order/stage does not match the frozen direction contract");
    need(number(field((NSDictionary*)side,"epsilon"),"captured side epsilon")==expectedEpsilon,
         "source469 side epsilon differs from the capture header");
    const double sideDt=number(field((NSDictionary*)side,"dt_seconds"),"captured side dt");
    need(std::bit_cast<std::uint32_t>(static_cast<float>(sideDt))==
         std::bit_cast<std::uint32_t>(center.input[45]),"captured side dt float word changed");
    Record result=center;
    const auto F=matrix9(field((NSDictionary*)side,"F_local"),"side F_local");
    const auto FLow=matrix9(field((NSDictionary*)side,"F_local_low"),"side F_local_low");
    const auto FDot=matrix9(field((NSDictionary*)side,"Fdot_local_per_s"),"side Fdot_local_per_s");
    std::copy(F.begin(),F.end(),result.input.begin());
    std::copy(FLow.begin(),FLow.end(),result.input.begin()+9u);
    std::copy(FDot.begin(),FDot.end(),result.input.begin()+18u);
    // The same accepted pre-projection history and captured directional action
    // apply at every perturbation endpoint.
    return result;
}

void verify469Package(const CompiledWorld& world,NSDictionary* header,
                      NSDictionary* runManifest,const std::string& captureSHA,
                      const std::string& packageSHA) {
    const auto fingerprint=hex64(stringValue(field(header,"world_fingerprint_hex"),"capture fingerprint"),
                                 "capture world fingerprint");
    id runFingerprint=field(runManifest,"compiled_world_fingerprint");
    need([runFingerprint isKindOfClass:[NSNumber class]],"run compiled fingerprint is not an integer");
    const auto exactRunFingerprint=static_cast<std::uint64_t>([(NSNumber*)runFingerprint unsignedLongLongValue]);
    need(world.fingerprint==fingerprint&&exactRunFingerprint==fingerprint,
         "captured run and exact compiled.nmatterpack fingerprint differ");
    need(world.dispatch.environmentCount==2u&&world.dispatch.femNodeCount==882u&&
         world.dispatch.tetrahedronCount==1920u&&world.materials.size()==3u&&
         world.fem.tetrahedra.size()==1920u,
         "compiled package dimensions/material table differ from source469 capture");
    NSArray* materials=(NSArray*)field(header,"materials");
    need(materials.count==3u,"source469 header omits a compiled material");
    const std::array<const char*,3> names={kLinerName,kMediumName,"starch2007_finite_glue_elastic"};
    const std::array<std::uint32_t,3> offsets={0u,28u,56u};
    const std::array<std::uint32_t,3> counts={28u,28u,3u};
    for(std::size_t i=0u;i<3u;++i) {
        NSDictionary* md=(NSDictionary*)materials[i];const auto& compiled=world.materials[i];
        need(stringValue(field(md,"name"),"captured material name")==names[i]&&
             number(field(md,"index"),"captured material index")==double(i)&&
             number(field(md,"parameter_offset"),"captured material offset")==double(offsets[i])&&
             compiled.parameterOffset==offsets[i]&&compiled.parameterCount==counts[i],
             "capture/package per-material parameter slice differs");
    }
    need(world.materials[0].stateCount==15u&&world.materials[1].stateCount==15u&&
         world.materials[0].flags==213u&&world.materials[1].flags==213u&&
         world.materials[0].localNewtonIterations==32u&&
         world.materials[1].localNewtonIterations==32u&&world.materials[2].stateCount==0u,
         "exact469 package does not retain accurate compensated source15 descriptors");
    NSDictionary* projection=(NSDictionary*)field(runManifest,"compensated_source_paper_projection");
    NSDictionary* flags=(NSDictionary*)field(projection,"effective_compiled_flags");
    need([(NSNumber*)field(projection,"enabled") boolValue]&&
         [(NSNumber*)field(projection,"accurate_bilinear15_projection") boolValue]&&
         number(field(flags,"liner_flags"),"manifest liner flags")==213.0&&
         number(field(flags,"medium_flags"),"manifest medium flags")==213.0&&
         number(field(flags,"liner_local_newton_iterations"),"manifest liner budget")==32.0&&
         number(field(flags,"medium_local_newton_iterations"),"manifest medium budget")==32.0,
         "source469 run manifest and compiled projection policy differ");
    NSArray* environmentParameters=(NSArray*)field(header,"parameters_by_environment");
    need(environmentParameters.count==2u&&world.parameters.size()==59u,
         "source469 per-environment parameter table shape differs");
    auto p0=flatVector(environmentParameters[0],59u,"environment0 parameters");
    auto p1=flatVector(environmentParameters[1],59u,"environment1 parameters");
    for(std::size_t i=0u;i<59u;++i) {
        const float compiled=world.parameters[i].valueAndBounds.x;
        need(std::bit_cast<std::uint32_t>(compiled)==std::bit_cast<std::uint32_t>(p0[i])&&
             std::bit_cast<std::uint32_t>(p0[i])==std::bit_cast<std::uint32_t>(p1[i]),
             "source469 package/environment material parameter word differs at "+std::to_string(i));
    }
    need(stringValue(field(header,"schema"),"capture schema")==kCaptureSchema&&
         number(field(header,"root"),"capture root")==20.0&&
         number(field(header,"newton_iteration"),"capture Newton iteration")==63.0&&
         number(field(header,"environment_count"),"capture environments")==2.0&&
         number(field(header,"tetrahedron_count_per_environment"),"capture tet count")==1920.0,
         "source469 capture identity/schema differs from the bound run");
    (void)captureSHA;(void)packageSHA;
}

Source469Capture readSource469(const std::filesystem::path& path,const CompiledWorld& world,
                               NSDictionary* runManifest,const std::string& captureSHA,
                               const std::string& packageSHA) {
    std::ifstream input(path);need(input.good(),"cannot open exact469 capture JSONL");
    Source469Capture result;result.env0.resize(1920u);std::string line;bool first=true;std::vector<bool> seen(1920u,false);
    std::array<std::size_t,3> counts{};
    while(std::getline(input,line)) {
        if(line.empty())continue;
        NSDictionary* d=parseObject(line,first?"469 header":"469 record");
        if(first) {
            verify469Package(world,d,runManifest,captureSHA,packageSHA);
            const auto epsilon=flatVector(field(d,"epsilon_values"),2u,"capture epsilon grid");
            result.epsilon={epsilon[0],epsilon[1]};
            need(result.epsilon[0]==0.03125&&result.epsilon[1]==0.015625,
                 "source469 captured epsilon pairs changed");
            result.fingerprint=world.fingerprint;
            for(const auto& p:world.parameters)result.parameters.push_back(p.valueAndBounds.x);
            first=false;continue;
        }
        if(stringValue(field(d,"record"),"record type")!="tetrahedron"||
           number(field(d,"environment"),"record environment")!=0.0)continue;
        const auto tet=static_cast<std::uint32_t>(number(field(d,"tet_id"),"record tet"));
        const auto material=static_cast<std::uint32_t>(number(field(d,"material_index"),"record material"));
        need(tet<1920u&&material<3u&&!seen[tet],"duplicate/out-of-range source469 env0 tet row");
        seen[tet]=true;++counts[material];
        need(world.fem.tetrahedra[tet].identity.x==material,
             "source469 row material differs from exact compiled tetrahedron descriptor");
        const auto stateCount=static_cast<std::uint32_t>(number(field(d,"material_state_count"),"row state count"));
        const auto parameterCount=static_cast<std::uint32_t>(number(field(d,"material_parameter_count"),"row parameter count"));
        need(stateCount==world.materials[material].stateCount&&
             parameterCount==world.materials[material].parameterCount,
             "source469 row state/parameter count differs from compiled material");
        const auto name=stringValue(field(d,"material_name"),"row material name");
        need((material==0u&&name==kLinerName)||(material==1u&&name==kMediumName)||
             (material==2u&&name=="starch2007_finite_glue_elastic"),
             "source469 row index/name binding differs");
        Source469Point point;NSDictionary* center=(NSDictionary*)field(d,"center");
        point.states[0]=recordFromCenter(d,center,world);
        point.capturedStress[0]=matrix9(field(center,"P_local_Pa"),"center captured stress");
        auto centerHigh=state15OrZero(field(center,"projected_state"),stateCount,"center projected state high");
        auto centerLow=state15OrZero(field(center,"projected_state_low"),stateCount,"center projected state low");
        auto centerProjection=flatVector(field(center,"projection"),4u,"center projection metadata");
        std::copy(centerHigh.begin(),centerHigh.end(),point.capturedProjectedHigh[0].begin());
        std::copy(centerLow.begin(),centerLow.end(),point.capturedProjectedLow[0].begin());
        std::copy(centerProjection.begin(),centerProjection.end(),point.capturedProjection[0].begin());
        auto tangent=flatArray<81>(field((NSDictionary*)field(d,"cached_algorithmic_tangent_local"),
                   "matrix_row_major_local_dP_dF_Pa_per_dimensionless_F"),"cached tangent");
        point.cachedTangent=tangent;
        NSArray* sides=(NSArray*)field(d,"sides");need(sides.count==4u,"source469 record lacks four exact endpoints");
        for(unsigned side=0u;side<4u;++side) {
            const double expectedEpsilon=result.epsilon[side/2u];
            point.states[side+1u]=recordFromSide(point.states[0],(NSDictionary*)sides[side],
                                                side+1u,expectedEpsilon);
            point.capturedStress[side+1u]=matrix9(
                field((NSDictionary*)sides[side],"P_local_Pa"),"side captured stress");
            const auto sideHigh=state15OrZero(field((NSDictionary*)sides[side],"projected_state"),stateCount,"side projected state high");
            const auto sideLow=state15OrZero(field((NSDictionary*)sides[side],"projected_state_low"),stateCount,"side projected state low");
            const auto sideProjection=flatVector(field((NSDictionary*)sides[side],"projection_diagnostics"),4u,"side projection diagnostics");
            std::copy(sideHigh.begin(),sideHigh.end(),point.capturedProjectedHigh[side+1u].begin());
            std::copy(sideLow.begin(),sideLow.end(),point.capturedProjectedLow[side+1u].begin());
            std::copy(sideProjection.begin(),sideProjection.end(),point.capturedProjection[side+1u].begin());
        }
        result.env0[tet]=std::move(point);
    }
    need(!first&&result.env0.size()==1920u&&std::all_of(seen.begin(),seen.end(),[](bool x){return x;}),
         "source469 env0 capture must include every tet exactly once");
    need(counts[0]==1152u&&counts[1]==576u&&counts[2]==192u,
         "source469 env0 material grade distribution changed");
    return result;
}

std::vector<MPSelection> readMPSelection(const std::filesystem::path& reportPath,
    const std::filesystem::path& arraysPath,const std::string& expectedReportSHA,
    const std::string& expectedArraysSHA,const std::string& captureSHA) {
    need(sha256(reportPath)==expectedReportSHA&&sha256(arraysPath)==expectedArraysSHA,
         "MP reference report/array hash differs from explicit binding");
    NSDictionary* report=readJSONDocument(reportPath);
    need(stringValue(field(report,"schema"),"MP report schema")==
             "numi.cardboard.native469-explicit-mp-fallback.v3",
         "MP reference report schema changed");
    NSDictionary* bindings=(NSDictionary*)field(report,"bindings");
    NSDictionary* nativeFiles=(NSDictionary*)field(bindings,"native_files");
    need(stringValue(field(nativeFiles,"final-newton"),"MP capture binding")==captureSHA&&
         stringValue(field(bindings,"source_snapshot"),"MP source snapshot")==
             "/Users/home/cardboard-implementation-20261014/snapshots/source-right-grip-telemetry-157",
         "MP reference report is not bound to exact469/source157 inputs");
    NSArray* rows=(NSArray*)field(report,"mp_results");need(rows.count==17u,"MP report must retain all17 failed-row roots");
    std::vector<MPSelection> result;std::set<std::tuple<unsigned,unsigned,unsigned>> unique;
    for(id item in rows) {
        NSDictionary* row=(NSDictionary*)item;
        need(stringValue(field(row,"fallback_status"),"MP fallback status")=="mp_gates_pass",
             "MP selection contains an unqualified fallback reference row");
        MPSelection x;
        x.tet=static_cast<std::uint32_t>(number(field(row,"tet"),"MP row tet"));
        x.material=static_cast<std::uint32_t>(number(field(row,"material_index"),"MP material index"));
        x.state=static_cast<std::uint32_t>(number(field(row,"state_index"),"MP state index"));
        x.name=stringValue(field(row,"state"),"MP state name");
        need(x.state<5u&&x.tet<1920u&&x.material<2u&&
             ((x.state==0u&&x.name=="center")||(x.state>0u&&x.name=="s"+std::to_string(x.state))),
             "MP selected point identity/state mapping invalid");
        need(unique.emplace(x.tet,x.state,x.material).second,"duplicate MP selection point");
        result.push_back(x);
    }
    return result;
}

ReferenceMap readReferenceExport(const std::filesystem::path& path,const std::string& expectedSHA,
    const std::string& expectedCaptureSHA,const std::string& expectedReportSHA,const std::string& expectedPackageManifestSHA,
    const std::string& expectedArraysSHA,const std::vector<MPSelection>& selections,
    const Source469Capture& capture) {
    need(sha256(path)==expectedSHA,"source469 independent reference export hash differs");
    std::ifstream input(path);need(input.good(),"cannot open source469 independent reference export");
    ReferenceMap references;std::string line;bool first=true;
    while(std::getline(input,line)) {
        if(line.empty())continue;
        NSDictionary* d=parseObject(line,first?"reference export header":"reference export row");
        if(first) {
            need(stringValue(field(d,"schema"),"reference export schema")==
                     "numi.cardboard.native469-selected-reference-v1"&&
                 stringValue(field(d,"capture_sha256"),"reference capture binding")==expectedCaptureSHA&&
                 stringValue(field(d,"report_sha256"),"reference MP report binding")==expectedReportSHA&&
                 stringValue(field(d,"mp_package_manifest_sha256"),"MP package manifest binding")==expectedPackageManifestSHA&&
                 stringValue(field(d,"arrays_sha256"),"reference arrays binding")==expectedArraysSHA&&
                 number(field(d,"environment"),"reference environment")==0.0&&
                 number(field(d,"rows_per_tet"),"reference state count")==5.0,
                 "reference export header/source bindings differ");
            first=false;continue;
        }
        need(stringValue(field(d,"record"),"reference row record")=="row",
             "unknown record in reference export");
        const auto tet=static_cast<std::uint32_t>(number(field(d,"tet_id"),"reference tet"));
        const auto state=static_cast<std::uint32_t>(number(field(d,"state_index"),"reference state"));
        const auto material=static_cast<std::uint32_t>(number(field(d,"material_index"),"reference material"));
        need(tet<1920u&&state<5u&&capture.env0[tet].states[0].material==material,
             "reference export row differs from exact source469 tet/material identity");
        ReferenceRow row;
        auto stress=flatDoubleVector(field(d,"fp64_stress"),9u,"independent FP64 stress");
        auto tangent=flatDoubleVector(field(d,"fp64_tangent"),81u,"independent FP64 tangent");
        std::copy(stress.begin(),stress.end(),row.stress.begin());
        std::copy(tangent.begin(),tangent.end(),row.tangent.begin());
        if(![field(d,"mp_state") isKindOfClass:[NSNull class]]) {
            const auto stateValues=flatDoubleVector(field(d,"mp_state"),15u,"MP state root");
            const auto stressValues=flatDoubleVector(field(d,"mp_stress"),9u,"MP stress root");
            const auto tangentValues=flatDoubleVector(field(d,"mp_tangent"),81u,"MP tangent root");
            row.mpState.emplace();row.mpStress.emplace();row.mpTangent.emplace();
            std::copy(stateValues.begin(),stateValues.end(),row.mpState->begin());
            std::copy(stressValues.begin(),stressValues.end(),row.mpStress->begin());
            std::copy(tangentValues.begin(),tangentValues.end(),row.mpTangent->begin());
        }
        need(references.emplace(ReferenceKey{tet,state},std::move(row)).second,
             "duplicate source469 independent reference row");
    }
    need(!first,"source469 independent reference export has no header");
    std::set<unsigned> selectedTets;for(const auto& item:selections)selectedTets.insert(item.tet);
    selectedTets.insert(0u);selectedTets.insert(576u);selectedTets.insert(864u);
    for(const auto tet:selectedTets)for(std::uint32_t state=0u;state<5u;++state)
        need(references.contains({tet,state}),"independent reference export lacks selected tet/state");
    for(const auto& item:selections)
        need(references.at({item.tet,item.state}).mpState.has_value(),
             "MP-selected row has no explicit MP root/tangent arrays");
    return references;
}

std::pair<float,float> splitFloat64(const double value) {
    const float high=static_cast<float>(value);
    const float low=static_cast<float>(value-static_cast<double>(high));
    return {high,low};
}
Sample endpoint(const Record& base,const double epsilon,const int sign,
                const std::string& stateLabel) {
    Sample sample;sample.row=base;sample.label="generated_endpoint";
    sample.stateLabel=stateLabel;sample.epsilon=epsilon;sample.sign=sign;
    const double dt=double(base.input[45]);
    for(std::size_t i=0u;i<9u;++i) {
        const double h=double(base.input[27u+i]);
        const double center=double(base.input[i])+double(base.input[9u+i]);
        const double target=center+double(sign)*epsilon*dt*h;
        const auto [high,low]=splitFloat64(target);
        sample.row.input[i]=high;sample.row.input[9u+i]=low;
        const double realized=double(high)+double(low);
        sample.requestedFDelta[i]=double(sign)*epsilon*dt*h;
        sample.realizedFDelta[i]=realized-center;
        const double rateCenter=double(base.input[18u+i]);
        const double rateTarget=rateCenter+double(sign)*epsilon*h;
        const float rateRealized=static_cast<float>(rateTarget);
        sample.row.input[18u+i]=rateRealized;
        sample.requestedRateDelta[i]=double(sign)*epsilon*h;
        sample.realizedRateDelta[i]=double(rateRealized)-rateCenter;
    }
    return sample;
}
Record withBasis(const Record& base,const unsigned column) {
    need(column<9u,"basis column outside 3x3 local deformation");
    Record result=base;
    std::fill(result.input.begin()+27u,result.input.begin()+45u,0.0f);
    result.input[27u+column]=1.0f;
    result.input[36u+column]=1.0f/std::max(base.input[45u],1.0e-12f);
    return result;
}
void writeSamples(const std::filesystem::path& path,const std::string& schema,
                  const std::vector<Sample>& samples,const std::vector<Source469Point>& points) {
    std::ofstream file(path);need(file.good(),"cannot create exact469 sample input manifest");
    file<<"{\"record\":\"header\",\"schema\":\""<<schema
        <<"\",\"sample_count\":"<<samples.size()<<",\"basis_contract\":\"local H=e_j and rateDirection=H/max(dt,1e-12); captured direction is separately preserved\"}\n";
    for(std::size_t i=0u;i<samples.size();++i) {
        const auto& s=samples[i];
        file<<"{\"record\":\"sample\",\"sample_index\":"<<i<<",\"label\":\""<<s.label
            <<"\",\"state_label\":\""<<s.stateLabel<<"\",\"state_index\":"<<s.stateIndex
            <<",\"tet_id\":"<<s.row.tet<<",\"material_index\":"<<s.row.material
            <<",\"column\":"<<s.column<<",\"epsilon\":"<<std::setprecision(17)<<s.epsilon
            <<",\"sign\":"<<s.sign<<",\"repeat_of\":"<<s.repeatOf<<",\"environment\":"<<s.row.env
            <<",\"input_f32_bits\":";
        bitsArrayJSON(file,s.row.input.data(),s.row.input.size());
        file<<",\"accepted_state_f32_bits\":";bitsArrayJSON(file,s.row.accepted.data(),s.row.accepted.size());
        file<<",\"requested_F_delta\":[";for(unsigned j=0;j<9u;++j){if(j)file<<',';numberJSON(file,s.requestedFDelta[j]);}
        file<<"],\"realized_F_delta\":[";for(unsigned j=0;j<9u;++j){if(j)file<<',';numberJSON(file,s.realizedFDelta[j]);}
        file<<"],\"requested_rate_delta\":[";for(unsigned j=0;j<9u;++j){if(j)file<<',';numberJSON(file,s.requestedRateDelta[j]);}
        file<<"],\"realized_rate_delta\":[";for(unsigned j=0;j<9u;++j){if(j)file<<',';numberJSON(file,s.realizedRateDelta[j]);}
        file<<"],\"captured_center_stress\":";
        if(s.captureReference)floatArrayJSON(file,points[s.row.tet].capturedStress[0].data(),9u);else file<<"null";
        file<<",\"captured_center_tangent_row_major\":";
        if(s.captureReference)floatArrayJSON(file,points[s.row.tet].cachedTangent.data(),81u);else file<<"null";
        file<<"}\n";
    }
    file.flush();need(file.good(),"exact469 sample manifest write failed");
}
void appendBasisSamples(std::vector<Sample>& out,const Record& row,const std::string& state,
                        const unsigned stateIndex,const bool captureReference) {
    for(unsigned column=0u;column<9u;++column) {
        Sample s;s.row=withBasis(row,column);s.label="captured_state_tangent_column";
        s.stateLabel=state;s.stateIndex=static_cast<int>(stateIndex);s.column=static_cast<int>(column);
        s.captureReference=captureReference;out.push_back(std::move(s));
    }
}
std::vector<float> runSamples(Metal& metal,const CompiledWorld& world,const std::string& policy,
    const std::vector<Sample>& samples,const std::filesystem::path& outdir,const std::string& captureSHA) {
    std::vector<Record> rows;rows.reserve(samples.size());for(const auto& s:samples)rows.push_back(s.row);
    return runBatch(metal,world,policy,rows,outdir,captureSHA,true);
}
void writeMatrixAudit(const std::filesystem::path& path,const Source469Capture& capture,
                      const std::vector<float>& centerOutput) {
    need(centerOutput.size()==capture.env0.size()*kOutputStride,
         "all-center batch output has a mismatched record count");
    std::ofstream file(path);need(file.good(),"cannot write all-center comparison");
    file<<"{\"record\":\"header\",\"schema\":\"numi.cardboard.source469-center-material-parity.v1\",\"rows\":"<<capture.env0.size()
        <<",\"interpretation\":\"measured same-state native stress and captured tangent-action residuals; no qualification inference\"}\n";
    double maxStressAbs=0.0,maxTangentAbs=0.0;std::size_t stressOK=0u,tangentOK=0u;
    for(std::size_t i=0u;i<capture.env0.size();++i) {
        const auto& point=capture.env0[i];const auto& row=point.states[0];const float* output=centerOutput.data()+i*kOutputStride;
        double stress2=0.0,tangent2=0.0,stressRef2=0.0,tangentRef2=0.0;
        std::array<double,9> action{};
        for(unsigned r=0u;r<9u;++r) {
            double expected=0.0;
            for(unsigned c=0u;c<9u;++c)
                expected+=double(point.cachedTangent[9u*r+c])*double(row.input[27u+c]);
            action[r]=expected;
            const double ds=double(output[8u+r])-double(point.capturedStress[0][r]);
            const double da=double(output[26u+r])-expected;
            stress2+=ds*ds;tangent2+=da*da;stressRef2+=double(point.capturedStress[0][r])*double(point.capturedStress[0][r]);
            tangentRef2+=expected*expected;maxStressAbs=std::max(maxStressAbs,std::abs(ds));maxTangentAbs=std::max(maxTangentAbs,std::abs(da));
        }
        stressOK+=output[0]==1.0f;tangentOK+=output[1]==1.0f&&output[2]==1.0f;
        file<<"{\"record\":\"row\",\"environment\":0,\"tet_id\":"<<row.tet<<",\"material_index\":"<<row.material
            <<",\"material_name\":\""<<row.materialName<<"\",\"stress_valid\":"<<(output[0]==1.0f?"true":"false")
            <<",\"tangent_valid\":"<<(output[1]==1.0f?"true":"false")<<",\"projected\":"<<(output[2]==1.0f?"true":"false")
            <<",\"projection_iterations\":"<<output[5]<<",\"projection_residual\":";numberJSON(file,output[3]);
        file<<",\"captured_stress_abs_l2\":";numberJSON(file,std::sqrt(stress2));
        file<<",\"captured_stress_relative_l2\":";numberJSON(file,stressRef2>0.0?std::sqrt(stress2/stressRef2):NAN);
        file<<",\"captured_tangent_action_abs_l2\":";numberJSON(file,std::sqrt(tangent2));
        file<<",\"captured_tangent_action_relative_l2\":";numberJSON(file,tangentRef2>0.0?std::sqrt(tangent2/tangentRef2):NAN);
        file<<",\"native_output_f32_bits\":";bitsArrayJSON(file,output,kOutputStride);file<<"}\n";
    }
    file<<"{\"record\":\"end\",\"stress_valid_rows\":"<<stressOK<<",\"tangent_valid_rows\":"<<tangentOK
        <<",\"max_stress_abs_entry_error\":";numberJSON(file,maxStressAbs);
    file<<",\"max_captured_action_abs_entry_error\":";numberJSON(file,maxTangentAbs);file<<"}\n";
}
std::pair<double,double> vectorErrors(const double* observed,const double* expected,const std::size_t count) {
    double error2=0.0,reference2=0.0;
    for(std::size_t i=0u;i<count;++i){const double d=observed[i]-expected[i];error2+=d*d;reference2+=expected[i]*expected[i];}
    return {std::sqrt(error2),reference2>0.0?std::sqrt(error2/reference2):NAN};
}
void writeCenterBasisAudit(const std::filesystem::path& path,const std::vector<Sample>& samples,
    const std::vector<float>& output,const Source469Capture& capture) {
    need(samples.size()==1920u*9u&&output.size()==samples.size()*kOutputStride,
         "all-center nine-column replay must contain exactly 17,280 samples");
    std::ofstream file(path);need(file.good(),"cannot write all-center nine-column audit");
    file<<"{\"record\":\"header\",\"schema\":\"numi.cardboard.source469-center-nine-column-parity.v1\","
        <<"\"rows\":"<<samples.size()<<",\"interpretation\":\"native direct tangent actions for all9 basis directions at all1920 captured env0 centers versus same-state captured matrices; not independent FP64 derivative qualification\"}\n";
    std::size_t stressOK=0u,tangentOK=0u,projectionOK=0u;
    double maximumStressAbs=0.0,maximumTangentAbs=0.0,maximumTangentRelative=0.0;
    for(std::size_t i=0u;i<samples.size();++i) {
        const auto& sample=samples[i];const auto* value=output.data()+i*kOutputStride;
        need(sample.row.tet<1920u&&sample.stateIndex==0&&sample.column>=0&&sample.column<9,
             "all-center basis sample identity is invalid");
        const auto& point=capture.env0[sample.row.tet];
        std::array<double,9> observedStress{},capturedStress{},observedAction{},capturedColumn{};
        double actionError2=0.0,actionReference2=0.0;
        for(std::size_t r=0u;r<9u;++r) {
            observedStress[r]=double(value[8u+r]);capturedStress[r]=double(point.capturedStress[0][r]);
            observedAction[r]=double(value[26u+r]);
            capturedColumn[r]=double(point.cachedTangent[9u*r+static_cast<unsigned>(sample.column)]);
            const double difference=observedAction[r]-capturedColumn[r];
            actionError2+=difference*difference;actionReference2+=capturedColumn[r]*capturedColumn[r];
            maximumTangentAbs=std::max(maximumTangentAbs,std::abs(difference));
        }
        const auto stressError=vectorErrors(observedStress.data(),capturedStress.data(),9u);
        const double actionAbs=std::sqrt(actionError2);
        const double actionRel=actionReference2>0.0?std::sqrt(actionError2/actionReference2):NAN;
        maximumStressAbs=std::max(maximumStressAbs,stressError.first);
        if(std::isfinite(actionRel))maximumTangentRelative=std::max(maximumTangentRelative,actionRel);
        stressOK+=value[0]==1.0f;tangentOK+=value[1]==1.0f;projectionOK+=value[2]==1.0f;
        file<<"{\"record\":\"basis\",\"tet_id\":"<<sample.row.tet<<",\"material_index\":"<<sample.row.material
            <<",\"column\":"<<sample.column<<",\"stress_valid\":"<<(value[0]==1.0f?"true":"false")
            <<",\"tangent_valid\":"<<(value[1]==1.0f?"true":"false")
            <<",\"projection_valid\":"<<(value[2]==1.0f?"true":"false")<<",\"projection_iterations\":";
        numberJSON(file,value[5]);file<<",\"projection_residual\":";numberJSON(file,value[3]);
        file<<",\"same_state_stress_abs_l2\":";numberJSON(file,stressError.first);
        file<<",\"tangent_column_abs_l2\":";numberJSON(file,actionAbs);
        file<<",\"tangent_column_relative_l2\":";numberJSON(file,actionRel);
        file<<",\"native_output_f32_bits\":";bitsArrayJSON(file,value,kOutputStride);file<<"}\n";
    }
    file<<"{\"record\":\"end\",\"stress_valid_rows\":"<<stressOK<<",\"tangent_valid_rows\":"<<tangentOK
        <<",\"projection_valid_rows\":"<<projectionOK<<",\"max_same_state_stress_abs_l2\":";
    numberJSON(file,maximumStressAbs);file<<",\"max_tangent_column_abs_entry\":";numberJSON(file,maximumTangentAbs);
    file<<",\"max_tangent_column_relative_l2\":";numberJSON(file,maximumTangentRelative);
    file<<",\"interpretation\":\"same-source captured matrix parity only; complete native row retention does not establish independent MP/FP64 accuracy\"}\n";
}
void referenceAction(const std::array<double,81>& matrix,const Record& row,double* action) {
    for(std::size_t r=0u;r<9u;++r){action[r]=0.0;for(std::size_t c=0u;c<9u;++c)
        action[r]+=matrix[9u*r+c]*double(row.input[27u+c]);}
}
void writeSelectedReferenceAudit(const std::filesystem::path& path,const std::vector<Sample>& samples,
    const std::vector<float>& output,const ReferenceMap& references,const std::string& referenceSHA) {
    need(output.size()==samples.size()*kOutputStride,"selected native/reference comparison shape mismatch");
    std::ofstream file(path);need(file.good(),"cannot write selected FP64/MP reference comparisons");
    file<<"{\"record\":\"header\",\"schema\":\"numi.cardboard.native469-selected-reference-comparison.v1\","
        <<"\"reference_export_sha256\":\""<<referenceSHA<<"\",\"rows\":"<<samples.size()
        <<",\"interpretation\":\"native source-function replay versus independent FP64/MP arrays; metrics are observations, not qualification\"}\n";
    double maximumStressRelative=0.0,maximumTangentRelative=0.0,maximumMPStressRelative=0.0,maximumMPTangentRelative=0.0;
    std::size_t stressRelativeResolved=0u,tangentRelativeResolved=0u,mpStressResolved=0u,mpTangentResolved=0u;
    for(std::size_t i=0u;i<samples.size();++i){
        const auto& sample=samples[i];const auto* value=output.data()+i*kOutputStride;
        const auto found=references.find({sample.row.tet,static_cast<std::uint32_t>(sample.stateIndex)});
        need(found!=references.end(),"selected sample lacks an independent reference row");
        const auto& ref=found->second;
        std::array<double,9> nativeStress{},nativeAction{},fp64Action{},mpAction{};
        for(std::size_t j=0u;j<9u;++j)nativeStress[j]=double(value[8u+j]);
        referenceAction(ref.tangent,sample.row,fp64Action.data());
        if(ref.mpTangent)referenceAction(*ref.mpTangent,sample.row,mpAction.data());
        for(std::size_t j=0u;j<9u;++j)nativeAction[j]=double(value[26u+j]);
        const auto stressFP64=vectorErrors(nativeStress.data(),ref.stress.data(),9u);
        const auto actionFP64=vectorErrors(nativeAction.data(),fp64Action.data(),9u);
        maximumStressRelative=std::max(maximumStressRelative,std::isfinite(stressFP64.second)?stressFP64.second:0.0);
        maximumTangentRelative=std::max(maximumTangentRelative,std::isfinite(actionFP64.second)?actionFP64.second:0.0);
        stressRelativeResolved+=std::isfinite(stressFP64.second);tangentRelativeResolved+=std::isfinite(actionFP64.second);
        file<<"{\"record\":\"comparison\",\"sample_index\":"<<i<<",\"tet_id\":"<<sample.row.tet
            <<",\"material_index\":"<<sample.row.material<<",\"state_label\":\""<<sample.stateLabel
            <<"\",\"column\":"<<sample.column<<",\"epsilon\":"<<std::setprecision(17)<<sample.epsilon
            <<",\"sign\":"<<sample.sign<<",\"stress_valid\":"<<(value[0]==1.0f?"true":"false")
            <<",\"tangent_valid\":"<<(value[1]==1.0f?"true":"false")
            <<",\"projection_valid\":"<<(value[2]==1.0f?"true":"false")
            <<",\"projection_iterations\":";numberJSON(file,value[5]);
        file<<",\"projection_residual\":";numberJSON(file,value[3]);
        file<<",\"stress_vs_fp64_abs_l2\":";numberJSON(file,stressFP64.first);
        file<<",\"stress_vs_fp64_relative_l2\":";numberJSON(file,stressFP64.second);
        file<<",\"directional_tangent_vs_fp64_abs_l2\":";numberJSON(file,actionFP64.first);
        file<<",\"directional_tangent_vs_fp64_relative_l2\":";numberJSON(file,actionFP64.second);
        file<<",\"mp_state_available\":"<<(ref.mpState?"true":"false");
        if(ref.mpState){
            std::array<double,15> nativeState{};
            for(std::size_t j=0u;j<15u;++j)nativeState[j]=double(value[35u+j])+double(value[50u+j]);
            const auto stateError=vectorErrors(nativeState.data(),ref.mpState->data(),15u);
            const auto stressError=vectorErrors(nativeStress.data(),ref.mpStress->data(),9u);
            const auto tangentError=vectorErrors(nativeAction.data(),mpAction.data(),9u);
            maximumMPStressRelative=std::max(maximumMPStressRelative,std::isfinite(stressError.second)?stressError.second:0.0);
            maximumMPTangentRelative=std::max(maximumMPTangentRelative,std::isfinite(tangentError.second)?tangentError.second:0.0);
            mpStressResolved+=std::isfinite(stressError.second);mpTangentResolved+=std::isfinite(tangentError.second);
            file<<",\"state_vs_mp_abs_l2\":";numberJSON(file,stateError.first);
            file<<",\"state_vs_mp_relative_l2\":";numberJSON(file,stateError.second);
            file<<",\"stress_vs_mp_abs_l2\":";numberJSON(file,stressError.first);
            file<<",\"stress_vs_mp_relative_l2\":";numberJSON(file,stressError.second);
            file<<",\"directional_tangent_vs_mp_abs_l2\":";numberJSON(file,tangentError.first);
            file<<",\"directional_tangent_vs_mp_relative_l2\":";numberJSON(file,tangentError.second);
        }
        file<<",\"native_output_f32_bits\":";bitsArrayJSON(file,value,kOutputStride);file<<"}\n";
    }
    file<<"{\"record\":\"end\",\"fp64_stress_relative_rows\":"<<stressRelativeResolved
        <<",\"fp64_tangent_relative_rows\":"<<tangentRelativeResolved<<",\"mp_stress_rows\":"<<mpStressResolved
        <<",\"mp_tangent_rows\":"<<mpTangentResolved<<",\"max_fp64_stress_relative_l2\":";
    numberJSON(file,maximumStressRelative);file<<",\"max_fp64_tangent_relative_l2\":";numberJSON(file,maximumTangentRelative);
    file<<",\"max_mp_stress_relative_l2\":";numberJSON(file,maximumMPStressRelative);
    file<<",\"max_mp_tangent_relative_l2\":";numberJSON(file,maximumMPTangentRelative);
    file<<",\"repeatability_is_accuracy_floor\":false,\"status\":\"diagnostic_complete_not_qualification\"}\n";
}
void writeEpsilonAudit(const std::filesystem::path& path,const std::vector<Sample>& samples,
    const std::vector<float>& output,const std::vector<Sample>& baseSamples,
    const std::vector<float>& baseOutput,const ReferenceMap& references,const std::string& referenceSHA) {
    need(output.size()==samples.size()*kOutputStride&&baseOutput.size()==baseSamples.size()*kOutputStride,
         "epsilon sweep comparison output shape mismatch");
    using Key=std::tuple<std::uint32_t,std::uint32_t,std::uint64_t>;
    struct Pair {std::optional<std::size_t> plus,minus;};
    std::map<Key,Pair> pairs;
    for(std::size_t i=0u;i<samples.size();++i){
        const auto& s=samples[i];if(s.column>=0||s.label!="generated_epsilon_endpoint")continue;
        const Key key{s.row.tet,static_cast<std::uint32_t>(s.stateIndex),std::bit_cast<std::uint64_t>(s.epsilon)};
        auto& pair=pairs[key];if(s.sign==1){need(!pair.plus,"duplicate positive epsilon endpoint");pair.plus=i;}
        else if(s.sign==-1){need(!pair.minus,"duplicate negative epsilon endpoint");pair.minus=i;}
        else need(false,"epsilon endpoint has no signed side");
    }
    std::map<std::pair<std::uint32_t,std::uint32_t>,std::size_t> bases;
    for(std::size_t i=0u;i<baseSamples.size();++i){const auto& s=baseSamples[i];if(s.column==-1)
        bases.emplace(std::pair{s.row.tet,static_cast<std::uint32_t>(s.stateIndex)},i);}
    std::ofstream file(path);need(file.good(),"cannot write epsilon finite-difference comparisons");
    file<<"{\"record\":\"header\",\"schema\":\"numi.cardboard.native469-epsilon-fd-comparison.v1\","
        <<"\"reference_export_sha256\":\""<<referenceSHA<<"\",\"pairs\":"<<pairs.size()
        <<",\"epsilon_unit\":\"dimensionless interpolation parameter; endpoint kinematics follow captured dt*H\","
        <<"\"repeatability_is_noise_floor\":false,\"interpretation\":\"realized split-float F and rate endpoints, one-sided and centered P differences versus independent tangent; retain branch and invalid-endpoint status\"}\n";
    std::size_t complete=0u,invalid=0u;double maxAbsolute=0.0,maxRelative=0.0;
    for(const auto& [key,pair]:pairs){
        const auto [tet,state,epsBits]=key;need(pair.plus&&pair.minus,"epsilon pair is incomplete");
        const auto plusIndex=*pair.plus,minusIndex=*pair.minus;
        const auto& plus=samples[plusIndex];const auto& minus=samples[minusIndex];
        const auto baseIt=bases.find({tet,state});need(baseIt!=bases.end(),"epsilon target lacks native center result");
        const std::size_t baseIndex=baseIt->second;
        const auto* p=output.data()+plusIndex*kOutputStride;const auto* m=output.data()+minusIndex*kOutputStride;
        const auto* b=baseOutput.data()+baseIndex*kOutputStride;
        const auto refIt=references.find({tet,state});need(refIt!=references.end(),"epsilon target lacks independent tangent");
        const auto& ref=refIt->second;
        std::array<double,9> centeredObserved{},centeredPredicted{},plusObserved{},plusPredicted{},minusObserved{},minusPredicted{};
        std::array<double,9> deltaEffPlus{},deltaEffMinus{};
        for(std::size_t j=0u;j<9u;++j){
            centeredObserved[j]=0.5*(double(p[8u+j])-double(m[8u+j]));
            plusObserved[j]=double(p[8u+j])-double(b[8u+j]);
            minusObserved[j]=double(m[8u+j])-double(b[8u+j]);
        }
        const double dt=double(plus.row.input[45]);
        for(std::size_t j=0u;j<9u;++j){
            deltaEffPlus[j]=0.5*(plus.realizedFDelta[j]+dt*plus.realizedRateDelta[j]);
            deltaEffMinus[j]=0.5*(minus.realizedFDelta[j]+dt*minus.realizedRateDelta[j]);
        }
        auto apply=[&](const std::array<double,81>& C,const std::array<double,9>& dx,std::array<double,9>& dy){
            for(std::size_t r=0u;r<9u;++r){dy[r]=0.0;for(std::size_t c=0u;c<9u;++c)dy[r]+=C[9u*r+c]*dx[c];}
        };
        std::array<double,9> deltaCenter{};
        for(std::size_t j=0u;j<9u;++j)deltaCenter[j]=0.5*(deltaEffPlus[j]-deltaEffMinus[j]);
        apply(ref.tangent,deltaCenter,centeredPredicted);apply(ref.tangent,deltaEffPlus,plusPredicted);
        apply(ref.tangent,deltaEffMinus,minusPredicted);
        const auto centerErr=vectorErrors(centeredObserved.data(),centeredPredicted.data(),9u);
        const auto plusErr=vectorErrors(plusObserved.data(),plusPredicted.data(),9u);
        const auto minusErr=vectorErrors(minusObserved.data(),minusPredicted.data(),9u);
        maxAbsolute=std::max({maxAbsolute,centerErr.first,plusErr.first,minusErr.first});
        if(std::isfinite(centerErr.second))maxRelative=std::max(maxRelative,centerErr.second);
        if(std::isfinite(plusErr.second))maxRelative=std::max(maxRelative,plusErr.second);
        if(std::isfinite(minusErr.second))maxRelative=std::max(maxRelative,minusErr.second);
        const bool valid=p[0]==1.0f&&m[0]==1.0f&&b[0]==1.0f&&p[2]==1.0f&&m[2]==1.0f&&b[2]==1.0f;
        complete+=valid;invalid+=!valid;
        const double epsilon=std::bit_cast<double>(epsBits);
        file<<"{\"record\":\"pair\",\"tet_id\":"<<tet<<",\"state_index\":"<<state
            <<",\"state_label\":\""<<plus.stateLabel<<"\",\"epsilon\":"<<std::setprecision(17)<<epsilon
            <<",\"plus_direct_valid\":"<<(p[0]==1.0f?"true":"false")<<",\"minus_direct_valid\":"<<(m[0]==1.0f?"true":"false")
            <<",\"plus_projection_valid\":"<<(p[2]==1.0f?"true":"false")<<",\"minus_projection_valid\":"<<(m[2]==1.0f?"true":"false")
            <<",\"plus_iterations\":";numberJSON(file,p[5]);file<<",\"minus_iterations\":";numberJSON(file,m[5]);
        file<<",\"realized_delta_f_plus\":[";for(std::size_t j=0u;j<9u;++j){if(j)file<<',';numberJSON(file,plus.realizedFDelta[j]);}file<<']';
        file<<",\"realized_delta_f_minus\":[";for(std::size_t j=0u;j<9u;++j){if(j)file<<',';numberJSON(file,minus.realizedFDelta[j]);}file<<']';
        file<<",\"realized_delta_fdot_plus\":[";for(std::size_t j=0u;j<9u;++j){if(j)file<<',';numberJSON(file,plus.realizedRateDelta[j]);}file<<']';
        file<<",\"realized_delta_fdot_minus\":[";for(std::size_t j=0u;j<9u;++j){if(j)file<<',';numberJSON(file,minus.realizedRateDelta[j]);}file<<']';
        file<<",\"centered_secant_vs_fp64_tangent_abs_l2\":";numberJSON(file,centerErr.first);
        file<<",\"centered_secant_vs_fp64_tangent_relative_l2\":";numberJSON(file,centerErr.second);
        file<<",\"plus_one_sided_vs_fp64_tangent_abs_l2\":";numberJSON(file,plusErr.first);
        file<<",\"minus_one_sided_vs_fp64_tangent_abs_l2\":";numberJSON(file,minusErr.first);
        file<<",\"pair_status\":\""<<(valid?"both_endpoints_valid":"one_or_more_endpoints_invalid")<<"\",\"native_output_bits_plus\":";
        bitsArrayJSON(file,p,kOutputStride);file<<",\"native_output_bits_minus\":";bitsArrayJSON(file,m,kOutputStride);file<<"}\n";
        if(ref.mpTangent){
            (void)ref.mpTangent; // MP endpoint roots are unavailable; native sides are compared with their independent FP64 center IFT only.
        }
    }
    file<<"{\"record\":\"end\",\"complete_pairs\":"<<complete<<",\"invalid_pairs\":"<<invalid
        <<",\"max_absolute_secant_error\":";numberJSON(file,maxAbsolute);file<<",\"max_relative_secant_error\":";
    numberJSON(file,maxRelative);file<<",\"noise_floor_from_repeatability\":false,\"status\":\"diagnostic_complete_not_qualification\"}\n";
}
void writeAllCapturedStatesAudit(const std::filesystem::path& path,const std::vector<Sample>& samples,
    const std::vector<float>& output,const Source469Capture& capture) {
    need(output.size()==samples.size()*kOutputStride&&samples.size()==1920u*5u,
         "all-captured-state replay must contain exactly five rows for every env0 tet");
    std::ofstream file(path);need(file.good(),"cannot write all captured center/side audit");
    file<<"{\"record\":\"header\",\"schema\":\"numi.cardboard.native469-all-captured-states.v1\","
        <<"\"rows\":"<<samples.size()<<",\"scope\":\"env0 all1920 center and four captured endpoints; production helper replays direct stress and tangent projection at actual H\","
        <<"\"interpretation\":\"same-source input replay; captured stress/state agreement is not independent FP64 derivative or coupon qualification\"}\n";
    std::size_t stressValid=0u,projectionValid=0u,highWordMismatches=0u,lowWordMismatches=0u;
    double maxStressAbs=0.0,maxStateAbs=0.0,maxStressRel=0.0,maxStateRel=0.0;
    for(std::size_t i=0u;i<samples.size();++i){
        const auto& sample=samples[i];const auto* value=output.data()+i*kOutputStride;
        need(sample.row.tet<1920u&&sample.stateIndex>=0&&sample.stateIndex<5,"bad all-state replay identity");
        const auto& point=capture.env0[sample.row.tet];const auto state=static_cast<std::size_t>(sample.stateIndex);
        std::array<double,9> nativeStress{},capturedStress{};
        for(std::size_t j=0u;j<9u;++j){nativeStress[j]=double(value[8u+j]);capturedStress[j]=double(point.capturedStress[state][j]);}
        const auto stressError=vectorErrors(nativeStress.data(),capturedStress.data(),9u);
        std::array<double,15> observedState{},expectedState{};
        for(std::size_t j=0u;j<15u;++j){
            observedState[j]=double(value[35u+j])+double(value[50u+j]);
            expectedState[j]=double(point.capturedProjectedHigh[state][j])+double(point.capturedProjectedLow[state][j]);
            highWordMismatches+=std::bit_cast<std::uint32_t>(value[35u+j])!=
                                std::bit_cast<std::uint32_t>(point.capturedProjectedHigh[state][j]);
            lowWordMismatches+=std::bit_cast<std::uint32_t>(value[50u+j])!=
                               std::bit_cast<std::uint32_t>(point.capturedProjectedLow[state][j]);
        }
        const auto stateError=vectorErrors(observedState.data(),expectedState.data(),15u);
        maxStressAbs=std::max(maxStressAbs,stressError.first);maxStateAbs=std::max(maxStateAbs,stateError.first);
        if(std::isfinite(stressError.second))maxStressRel=std::max(maxStressRel,stressError.second);
        if(std::isfinite(stateError.second))maxStateRel=std::max(maxStateRel,stateError.second);
        const auto& projection=point.capturedProjection[state];
        const bool stateful=sample.row.material<2u;
        stressValid+=value[0]==1.0f;projectionValid+=value[2]==1.0f;
        file<<"{\"record\":\"state\",\"tet_id\":"<<sample.row.tet<<",\"material_index\":"<<sample.row.material
            <<",\"state_label\":\""<<sample.stateLabel<<"\",\"direct_stress_valid\":"<<(value[0]==1.0f?"true":"false")
            <<",\"tangent_projection_valid\":"<<(value[2]==1.0f?"true":"false")<<",\"native_projection_residual\":";
        numberJSON(file,value[3]);file<<",\"capture_projection_residual\":";numberJSON(file,projection[0]);
        file<<",\"native_projection_iterations\":";numberJSON(file,value[5]);
        file<<",\"capture_projection_iterations\":";numberJSON(file,projection[2]);
        file<<",\"projection_comparison_applicable\":"<<(stateful?"true":"false")
            <<",\"direct_stress_vs_captured_abs_l2\":";numberJSON(file,stressError.first);
        file<<",\"direct_stress_vs_captured_relative_l2\":";numberJSON(file,stressError.second);
        file<<",\"projected_state_vs_capture_abs_l2\":";numberJSON(file,stateError.first);
        file<<",\"projected_state_vs_capture_relative_l2\":";numberJSON(file,stateError.second);
        file<<",\"native_output_f32_bits\":";bitsArrayJSON(file,value,kOutputStride);file<<"}\n";
    }
    file<<"{\"record\":\"end\",\"stress_valid_rows\":"<<stressValid<<",\"tangent_projection_valid_rows\":"<<projectionValid
        <<",\"captured_state_high_word_mismatches\":"<<highWordMismatches
        <<",\"captured_state_low_word_mismatches\":"<<lowWordMismatches<<",\"max_stress_abs_l2\":";
    numberJSON(file,maxStressAbs);file<<",\"max_stress_relative_l2\":";numberJSON(file,maxStressRel);
    file<<",\"max_projected_state_abs_l2\":";numberJSON(file,maxStateAbs);file<<",\"max_projected_state_relative_l2\":";
    numberJSON(file,maxStateRel);file<<",\"repeatability_is_accuracy_floor\":false,\"status\":\"replay_complete_not_qualification\"}\n";
}
void writeExact469Summary(const std::filesystem::path& path,const std::string& captureSHA,
    const std::string& packageSHA,const std::string& runManifestSHA,const std::string& receiptSHA,
    const std::string& mpReportSHA,const std::string& mpArraysSHA,const std::string& mpPackageManifestSHA,
    const std::string& referenceSHA,const Source469Capture& capture,
    const std::vector<MPSelection>& selected,const std::vector<float>& repeatOutput,
    const std::vector<Sample>& repeats,const std::size_t centerBasisRows,
    const std::size_t selectedReplayRows,const std::size_t allCapturedRows,
    const std::size_t epsilonRows) {
    std::size_t mismatches=0u,compared=0u;
    for(std::size_t i=0u;i<repeats.size();++i) if(repeats[i].repeatOf>=0) {
        const auto a=static_cast<std::size_t>(repeats[i].repeatOf);const auto b=i;
        for(std::size_t slot=0u;slot<kOutputStride;++slot) {
            ++compared;mismatches+=std::bit_cast<std::uint32_t>(repeatOutput[a*kOutputStride+slot])!=
                                      std::bit_cast<std::uint32_t>(repeatOutput[b*kOutputStride+slot]);
        }
    }
    std::ofstream file(path);need(file.good(),"cannot write native replay manifest");
    file<<"{\"schema\":\"numi.cardboard.native469-material-replay.v1\",\"source_snapshot\":\"source-warm-cgls-refinement-169\","
        <<"\"checker_base_snapshot_sha256\":\""<<kBaseSnapshotSHA<<"\","
        <<"\"capture_sha256\":\""<<captureSHA<<"\",\"compiled_package_sha256\":\""<<packageSHA
        <<"\",\"run_manifest_sha256\":\""<<runManifestSHA<<"\",\"receipt_sha256\":\""<<receiptSHA
        <<"\",\"mp_reference_report_sha256\":\""<<mpReportSHA<<"\",\"mp_reference_arrays_sha256\":\""<<mpArraysSHA
        <<"\",\"mp_reference_package_manifest_sha256\":\""<<mpPackageManifestSHA
        <<"\",\"selected_reference_export_sha256\":\""<<referenceSHA
        <<"\",\"world_fingerprint\":\""<<std::hex<<capture.fingerprint<<std::dec
        <<"\",\"capture_root\":20,\"capture_newton_iteration\":63,\"environment_scope\":0,"
        <<"\"env0_center_rows\":"<<capture.env0.size()<<",\"paper_rows\":1728,\"starch_rows\":192,"
        <<"\"all_center_nine_basis_rows\":"<<centerBasisRows
        <<",\"all_five_captured_state_rows\":"<<allCapturedRows
        <<",\"selected_captured_and_basis_rows\":"<<selectedReplayRows
        <<",\"epsilon_sweep_rows\":"<<epsilonRows
        <<",\"selected_mp_failure_roots\":"<<selected.size()<<",\"repeatability_word_mismatches\":"<<mismatches
        <<",\"repeatability_words_compared\":"<<compared
        <<",\"repeatability_is_accuracy_floor\":false,\"physical_transactions\":0,\"full_solver_qualification\":false,"
        <<"\"material_qualification\":false,\"status\":\"native_material_replay_dispatch_complete\"}\n";
}

int runSource469(int argc,char** argv,const bool doNative) {
    need(argc==27&&std::string(argv[1])==(doNative?"--native-469":"--cpu-admission-469"),
         "source469 usage/argument count mismatch");
    const std::string expectedCapture=argv[3],expectedPackage=argv[5],expectedManifest=argv[7],
        expectedReceipt=argv[9],expectedMP=argv[11],expectedArrays=argv[13],
        expectedMPManifest=argv[15],expectedReferences=argv[17];
    const std::filesystem::path capturePath=argv[18],packagePath=argv[19],manifestPath=argv[20],
        receiptPath=argv[21],mpPath=argv[22],arraysPath=argv[23],mpManifestPath=argv[24],
        referencePath=argv[25],outdir=argv[26];
    for(const auto* hash:{&expectedCapture,&expectedPackage,&expectedManifest,&expectedReceipt,&expectedMP,&expectedArrays,&expectedMPManifest,&expectedReferences})
        need(hash->size()==64u&&std::all_of(hash->begin(),hash->end(),[](unsigned char c){return std::isxdigit(c)!=0;}),
             "source469 all input hash arguments must be 64 hexadecimal characters");
    const auto captureSHA=sha256(capturePath),packageSHA=sha256(packagePath),manifestSHA=sha256(manifestPath),
        receiptSHA=sha256(receiptPath),mpSHA=sha256(mpPath),arraysSHA=sha256(arraysPath),
        mpManifestSHA=sha256(mpManifestPath),referenceSHA=sha256(referencePath);
    need(captureSHA==expectedCapture&&packageSHA==expectedPackage&&manifestSHA==expectedManifest&&
         receiptSHA==expectedReceipt&&mpSHA==expectedMP&&arraysSHA==expectedArrays&&
         mpManifestSHA==expectedMPManifest&&referenceSHA==expectedReferences,
         "source469 input bytes differ from explicit command bindings");
    NSDictionary* manifest=readJSONDocument(manifestPath);NSDictionary* receipt=readJSONDocument(receiptPath);
    need([(NSNumber*)field(receipt,"execution_started") boolValue]&&
         [(NSNumber*)field(receipt,"source_inputs_unchanged") boolValue]&&
         number(field(receipt,"exit_code"),"origin solver exit code")==2.0,
         "source469 origin is not the retained completed failed physical run");
    CompiledWorld world;std::string packageError;
    need(readPackage(packagePath,world,nullptr,&packageError),"exact469 compiled package rejected: "+packageError);
    const auto capture=readSource469(capturePath,world,manifest,captureSHA,packageSHA);
    const auto selection=readMPSelection(mpPath,arraysPath,mpSHA,arraysSHA,captureSHA);
    NSDictionary* mpManifest=readJSONDocument(mpManifestPath);
    NSDictionary* included=(NSDictionary*)field(mpManifest,"included_files_sha256");
    need(stringValue(field(included,"run-002/report.json"),"MP package report output binding")==mpSHA&&
         stringValue(field(included,"run-002/mixed_reference_arrays.npz"),"MP package array output binding")==arraysSHA,
         "MP package manifest does not bind supplied report and arrays");
    const auto references=readReferenceExport(referencePath,referenceSHA,captureSHA,mpSHA,mpManifestSHA,
                                               arraysSHA,selection,capture);
    std::map<std::pair<unsigned,unsigned>,const MPSelection*> uniquePoints;
    for(const auto& x:selection)uniquePoints[{x.tet,x.state}]=&x;
    std::set<unsigned> selectedTets;for(const auto& x:selection)selectedTets.insert(x.tet);
    for(unsigned material=0u;material<3u;++material) {
        auto it=std::find_if(capture.env0.begin(),capture.env0.end(),[&](const Source469Point& x){return x.states[0].material==material;});
        need(it!=capture.env0.end(),"source469 lacks one of the three compiled material grades");selectedTets.insert(it->states[0].tet);
    }
    std::vector<Sample> centerSamples;centerSamples.reserve(capture.env0.size());
    std::vector<Sample> allCapturedSamples;allCapturedSamples.reserve(capture.env0.size()*5u);
    for(const auto& point:capture.env0) {
        Sample s;s.row=point.states[0];s.label="env0_center_captured_direction";s.stateLabel="center";
        s.stateIndex=0;s.captureReference=true;centerSamples.push_back(std::move(s));
        for(unsigned state=0u;state<5u;++state){
            Sample q;q.row=point.states[state];q.label="env0_all_tets_captured_state_action";
            q.stateLabel=state==0u?"center":"s"+std::to_string(state);q.stateIndex=static_cast<int>(state);
            q.captureReference=state==0u;allCapturedSamples.push_back(std::move(q));
        }
    }
    std::vector<Sample> centerBasisSamples,columnSamples,epsilonSamples,repeatSamples;
    centerBasisSamples.reserve(capture.env0.size()*9u);
    for(const auto& point:capture.env0)
        appendBasisSamples(centerBasisSamples,point.states[0],"center",0u,true);
    need(centerBasisSamples.size()==1920u*9u,
         "center basis plan must cover all nine columns for all1920 env0 tetrahedra");
    for(const auto tet:selectedTets) {
        const auto& point=capture.env0[tet];
        for(unsigned state=0u;state<5u;++state) {
            const std::string label=state==0u?"center":"s"+std::to_string(state);
            Sample action;action.row=point.states[state];action.label="captured_state_direction_action";
            action.stateLabel=label;action.stateIndex=static_cast<int>(state);action.captureReference=state==0u;
            columnSamples.push_back(action);
            appendBasisSamples(columnSamples,point.states[state],label,state,state==0u);
        }
    }
    constexpr std::array<int,9> epsilonPowers={-7,-8,-9,-10,-12,-14,-16,-18,-20};
    std::set<std::pair<unsigned,unsigned>> epsilonTargets;
    for(const auto& x:selection)epsilonTargets.emplace(x.tet,x.state);
    for(const auto tet:selectedTets)epsilonTargets.emplace(tet,0u);
    for(const auto& [tet,stateIndex]:epsilonTargets) {
        const auto& base=capture.env0[tet].states[stateIndex];
        const std::string label=stateIndex==0u?"center":"s"+std::to_string(stateIndex);
        for(const int power:epsilonPowers) {
            const double epsilon=std::ldexp(1.0,power);
            for(const int sign:{1,-1}) {
                Sample endpointSample=endpoint(base,epsilon,sign,label);
                endpointSample.label="generated_epsilon_endpoint";endpointSample.stateIndex=static_cast<int>(stateIndex);
                epsilonSamples.push_back(endpointSample);
                for(unsigned column=0u;column<9u;++column) {
                    Sample c=endpointSample;c.row=withBasis(endpointSample.row,column);c.column=static_cast<int>(column);
                    c.label="generated_epsilon_tangent_column";epsilonSamples.push_back(std::move(c));
                }
            }
        }
    }
    for(const auto& x:selection) {
        const auto& base=capture.env0[x.tet].states[x.state];
        const std::string label=x.state==0u?"center":"s"+std::to_string(x.state);
        Sample first;first.row=base;first.label="repeatability_original";first.stateLabel=label;
        first.stateIndex=static_cast<int>(x.state);repeatSamples.push_back(first);
        Sample second=first;second.label="repeatability_duplicate";second.repeatOf=static_cast<int>(repeatSamples.size()-1u);
        repeatSamples.push_back(std::move(second));
    }
    need(!centerSamples.empty()&&!columnSamples.empty()&&!epsilonSamples.empty()&&!repeatSamples.empty(),
         "source469 sample plan unexpectedly empty");
    if(!doNative) {
        std::cout<<"{\"record\":\"source469_cpu_admission\",\"status\":\"pass\",\"capture_sha256\":\""<<captureSHA
            <<"\",\"package_sha256\":\""<<packageSHA<<"\",\"world_fingerprint\":\""<<std::hex<<capture.fingerprint<<std::dec
            <<"\",\"env0_centers\":"<<centerSamples.size()<<",\"paper_centers\":1728,\"starch_centers\":192"
            <<",\"center_all_nine_basis_rows\":"<<centerBasisSamples.size()
            <<",\"captured_columns_and_sides\":"<<columnSamples.size()<<",\"epsilon_cases\":"<<epsilonSamples.size()
            <<",\"repeatability_cases\":"<<repeatSamples.size()<<",\"selected_reference_roots\":"<<selection.size()
            <<",\"all_capture_center_and_side_rows\":"<<allCapturedSamples.size()
            <<",\"independent_reference_rows\":"<<references.size()
            <<",\"physical_gpu_execution\":false}\n";return 0;
    }
    need(!std::filesystem::exists(outdir),"source469 native output directory must be new/O_EXCL");
    need(std::filesystem::create_directories(outdir),"could not create source469 native output directory");
    writeSamples(outdir/"center-all-inputs.jsonl","numi.cardboard.source469-center-inputs.v1",centerSamples,capture.env0);
    writeSamples(outdir/"center-all-nine-basis-inputs.jsonl","numi.cardboard.source469-center-nine-basis-inputs.v1",centerBasisSamples,capture.env0);
    writeSamples(outdir/"all-captured-center-side-inputs.jsonl","numi.cardboard.source469-all-captured-center-side-inputs.v1",allCapturedSamples,capture.env0);
    writeSamples(outdir/"selected-captured-inputs.jsonl","numi.cardboard.source469-selected-captured-inputs.v1",columnSamples,capture.env0);
    writeSamples(outdir/"epsilon-grid-inputs.jsonl","numi.cardboard.source469-epsilon-grid-inputs.v1",epsilonSamples,capture.env0);
    writeSamples(outdir/"repeatability-inputs.jsonl","numi.cardboard.source469-repeat-inputs.v1",repeatSamples,capture.env0);
    Metal metal;
    const auto centerOutput=runSamples(metal,world,"source469_center_1728paper_192starch",centerSamples,outdir,captureSHA);
    writeMatrixAudit(outdir/"center-captured-operator-comparison.jsonl",capture,centerOutput);
    const auto centerBasisOutput=runSamples(metal,world,"source469_center_all_nine_basis_1920",centerBasisSamples,outdir,captureSHA);
    writeCenterBasisAudit(outdir/"center-all-nine-basis-comparison.jsonl",centerBasisSamples,centerBasisOutput,capture);
    const auto selectedOutput=runSamples(metal,world,"source469_selected_captured_sides_columns",columnSamples,outdir,captureSHA);
    writeSelectedReferenceAudit(outdir/"selected-fp64-mp-reference-comparison.jsonl",
                                columnSamples,selectedOutput,references,referenceSHA);
    const auto epsilonOutput=runSamples(metal,world,"source469_selected_epsilon_grid",epsilonSamples,outdir,captureSHA);
    writeEpsilonAudit(outdir/"epsilon-fd-comparison.jsonl",epsilonSamples,epsilonOutput,
                      columnSamples,selectedOutput,references,referenceSHA);
    const auto repeatOutput=runSamples(metal,world,"source469_repeat_identical",repeatSamples,outdir,captureSHA);
    const auto allCapturedOutput=runSamples(metal,world,"source469_all_1920_centers_4_sides",allCapturedSamples,outdir,captureSHA);
    writeAllCapturedStatesAudit(outdir/"all-captured-center-side-comparison.jsonl",
                                allCapturedSamples,allCapturedOutput,capture);
    std::set<std::pair<unsigned,unsigned>> traced;
    for(const auto& x:selection)if(traced.emplace(x.tet,x.state).second) {
        const auto& row=capture.env0[x.tet].states[x.state];
        runTrace(metal,world,"source469_"+(x.state==0u?std::string("center"):"s"+std::to_string(x.state))+"_tet"+std::to_string(x.tet),
                 row,outdir,captureSHA,true);
    }
    writeExact469Summary(outdir/"source469-replay-manifest.json",captureSHA,packageSHA,manifestSHA,receiptSHA,
                         mpSHA,arraysSHA,mpManifestSHA,referenceSHA,capture,selection,repeatOutput,repeatSamples,
                         centerBasisSamples.size(),columnSamples.size(),allCapturedSamples.size(),epsilonSamples.size());
    std::cout<<"{\"record\":\"source469_native_replay\",\"status\":\"dispatch_complete\",\"center_rows\":"
        <<centerSamples.size()<<",\"selected_capture_rows\":"<<columnSamples.size()<<",\"epsilon_rows\":"<<epsilonSamples.size()
        <<",\"repeat_rows\":"<<repeatSamples.size()<<",\"center_output_words\":"<<centerOutput.size()
        <<",\"center_all_nine_basis_rows\":"<<centerBasisSamples.size()<<",\"center_all_nine_basis_output_words\":"<<centerBasisOutput.size()
        <<",\"all_capture_output_words\":"<<allCapturedOutput.size()
        <<",\"selected_output_words\":"<<selectedOutput.size()<<",\"epsilon_output_words\":"<<epsilonOutput.size()
        <<",\"repeat_output_words\":"<<repeatOutput.size()<<",\"physical_transactions\":0,\"material_qualification\":false}\n";
    return 0;
}
}

int main(int argc,char** argv) {@autoreleasepool {try {
    if(argc>1&&std::string(argv[1])=="--cpu-admission-469")return runSource469(argc,argv,false);
    if(argc>1&&std::string(argv[1])=="--native-469")return runSource469(argc,argv,true);
    std::cerr<<"usage: cardboard_source469_material_replay_check --cpu-admission-469|--native-469 <eight SHA bindings and exact469 input/output paths>\n";
    return 2;
} catch(const std::exception& e) {
    std::cerr<<"FAIL source-bound469 material replay: "<<e.what()<<'\n';return 1;
}}}
