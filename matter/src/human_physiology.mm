#import <Foundation/Foundation.h>
#import <CommonCrypto/CommonDigest.h>
#include "numi/matter/human_physiology.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string_view>

namespace numi::matter {
namespace {
void need(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("Human physiology: " + message);
}
std::string text(id value) {
    need([value isKindOfClass:NSString.class], "expected string");
    NSData* bytes = [(NSString*)value dataUsingEncoding:NSUTF8StringEncoding];
    need(bytes != nil && bytes.length > 0 && bytes.length <= 4096, "invalid UTF-8 identity");
    std::string result(static_cast<const char*>(bytes.bytes), bytes.length);
    need(result.find('\0') == std::string::npos, "NUL in identity");
    return result;
}
// Foundation validates JSON grammar; retain the raw object structure to reject
// duplicate keys before its dictionary conversion can hide them. Decode each
// key as a JSON string so escaped spellings cannot bypass this check.
void rejectDuplicateKeys(NSData* data) {
    struct ObjectFrame {bool object;bool key;std::set<std::string> seen;};
    std::vector<ObjectFrame> frames;
    const char* bytes=static_cast<const char*>(data.bytes);
    for(NSUInteger i=0;i<data.length;++i){
        const char ch=bytes[i];
        if(ch=='{' || ch=='[') frames.push_back({ch=='{',ch=='{',{}});
        else if(ch=='}' || ch==']') {need(!frames.empty(),"invalid JSON nesting");frames.pop_back();}
        else if(ch==',' && !frames.empty() && frames.back().object) frames.back().key=true;
        else if(ch==':' && !frames.empty() && frames.back().object) frames.back().key=false;
        else if(ch=='"') {
            const NSUInteger begin=i;
            for(++i;i<data.length;++i){if(bytes[i]=='\\'){++i;continue;}if(bytes[i]=='"')break;}
            need(i<data.length,"unterminated JSON string");
            if(!frames.empty() && frames.back().object && frames.back().key){
                NSData* keyData=[NSData dataWithBytes:bytes+begin length:i-begin+1];
                id key=[NSJSONSerialization JSONObjectWithData:keyData options:NSJSONReadingFragmentsAllowed error:nil];
                need(frames.back().seen.insert(text(key)).second,"duplicate JSON object key");
            }
        }
    }
    need(frames.empty(),"invalid JSON nesting");
}
NSDictionary* record(id value, std::initializer_list<const char*> keys, std::initializer_list<const char*> extra = {}) {
    need([value isKindOfClass:NSDictionary.class], "expected object");
    NSDictionary* result = value;
    std::set<std::string> expected;
    for (const char* key : keys) expected.emplace(key);
    for (const char* key : extra) expected.emplace(key);
    need(result.count == expected.size(), "missing or unsupported object fields");
    for (id key in result) need(expected.contains(text(key)), "unsupported field " + text(key));
    return result;
}
NSArray* list(id value) {
    need([value isKindOfClass:NSArray.class], "expected array");
    need([(NSArray*)value count] <= 100000, "array exceeds source admission limit");
    return value;
}
double number(id value) {
    need([value isKindOfClass:NSNumber.class] &&
         CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(), "expected finite number");
    double result = [(NSNumber*)value doubleValue];
    need(std::isfinite(result), "nonfinite number");
    return result;
}
std::uint32_t identifier(id value) {
    double result = number(value);
    need(result >= 1 && result < NM_INVALID_INDEX && result == std::floor(result), "invalid stable identifier");
    return static_cast<std::uint32_t>(result);
}
std::array<std::uint64_t,4> digest(id value) {
    std::string hex = text(value);
    need(hex.size() == 64, "SHA256 must have 64 lowercase hexadecimal digits");
    std::array<std::uint64_t,4> result{};
    for (std::size_t i=0; i<64; ++i) {
        char ch=hex[i];
        need((ch>='0' && ch<='9') || (ch>='a' && ch<='f'), "invalid SHA256");
        result[i/16] = (result[i/16] << 4) | static_cast<unsigned>(ch<='9' ? ch-'0' : ch-'a'+10);
    }
    need(std::any_of(result.begin(),result.end(),[](auto x){return x!=0;}), "missing SHA256 identity");
    return result;
}
std::vector<double> amounts(id value, std::size_t count) {
    NSArray* array=list(value);
    need(array.count==count,"species amount arity mismatch");
    std::vector<double> result;
    for(id item in array) { double v=number(item); need(v>=0,"negative species amount"); result.push_back(v); }
    return result;
}
void positive(double value) { need(value>0,"physical scale, tolerance, or coefficient must be positive"); }
}

bool readHumanPhysiologyNetwork(const std::filesystem::path& path,
                               VascularNetworkSource& output, std::string* error) {
    @autoreleasepool { try {
        if (error) error->clear();
        NSData* data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
        need(data!=nil && data.length>0 && data.length<=32u*1024u*1024u,"cannot read bounded source payload");
        NSError* jsonError=nil;
        id json=[NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonError];
        need(json!=nil, "invalid JSON");
        rejectDuplicateKeys(data);
        NSDictionary* root=record(json,{"schema","model_id","qualification","law","authored_graph_sha256","source_graph_sha256","residual_tolerance","species","compartments","connections","tissue_reservoirs","exchanges"});
        const auto schema=text(root[@"schema"]);
        const bool cardiac=schema=="HumanPack.physiology-native.v2";
        need(cardiac || schema=="HumanPack.physiology-native.v1","unsupported schema");
        (void)text(root[@"model_id"]);
        const auto qualification=text(root[@"qualification"]);
        need(cardiac ? qualification=="source_model_reproduction" : (qualification=="fixture_only" || qualification=="uncalibrated"),"unsupported qualification claim");
        need(text(root[@"law"])==(cardiac ? "closed_periodic_elastance_orifice_v2" : "closed_linear_compliance_transport_v1"),"unsupported constitutive law");
        const double tolerance=number(root[@"residual_tolerance"]);
        need(tolerance>0 && tolerance<1,"invalid residual tolerance");
        VascularNetworkSource result;
        unsigned char hash[CC_SHA256_DIGEST_LENGTH];
        CC_SHA256(data.bytes,static_cast<CC_LONG>(data.length),hash);
        for (std::size_t i=0;i<32;++i) result.contentIdentity[i/8]=(result.contentIdentity[i/8]<<8)|hash[i];
        result.authoredIdentity=digest(root[@"authored_graph_sha256"]);
        result.sourceIdentity=digest(root[@"source_graph_sha256"]);
        std::uint32_t next=1;
        std::set<std::string> allNames, volumeOwners;
        auto identity=[&](NSDictionary* r, std::string& previous) {
            std::string name=text(r[@"id"]);
            need(previous.empty() || previous<name,"record IDs must be strictly sorted");
            previous=name;
            need(allNames.insert(name).second,"duplicate semantic ID");
            auto value=identifier(r[@"stable_identifier"]);
            need(value==next++,"stable identifiers must be canonical global sequence");
            return value;
        };
        auto owner=[&](NSDictionary* r) {
            need(volumeOwners.insert(text(r[@"physical_volume_owner_id"])).second,"duplicate physical-volume owner");
        };
        std::string previous;
        for (id item in list(root[@"species"])) {
            NSDictionary* r=record(item,{"id","stable_identifier","description","amount_scale_mol","amount_residual_tolerance"});
            VascularSpeciesSource v; v.stableIdentifier=identity(r,previous); v.name=text(r[@"id"]);
            (void)text(r[@"description"]);
            v.amountScale=number(r[@"amount_scale_mol"]); positive(v.amountScale);
            v.amountResidualTolerance=number(r[@"amount_residual_tolerance"]); need(v.amountResidualTolerance==tolerance,"row tolerance differs from contract");
            result.species.push_back(v);
        }
        need(cardiac || !result.species.empty(),"species registry is empty");
        previous.clear();
        for (id item in list(root[@"compartments"])) {
            NSDictionary* r=record(item,{"id","stable_identifier","anatomical_region_id","physical_volume_owner_id","reference_volume_m3","reference_pressure_pa","external_pressure_pa","compliance_m3_per_pa","initial_volume_m3","volume_scale_m3","volume_residual_tolerance","initial_species_mol"}, cardiac ? std::initializer_list<const char*>{"storage_kind","pressure_law","elastance_min_pa_per_m3","elastance_max_pa_per_m3","period_seconds","activation_start","activation_end","source_pi"} : std::initializer_list<const char*>{});
            VascularCompartmentSource v; v.stableIdentifier=identity(r,previous); owner(r);
            v.anatomicalIdentifier=text(r[@"anatomical_region_id"]);
            v.referenceVolume=number(r[@"reference_volume_m3"]); v.referencePressure=number(r[@"reference_pressure_pa"]);
            v.externalPressure=number(r[@"external_pressure_pa"]); v.compliance=number(r[@"compliance_m3_per_pa"]);
            v.initialVolume=number(r[@"initial_volume_m3"]); v.volumeScale=number(r[@"volume_scale_m3"]);
            v.volumeResidualTolerance=number(r[@"volume_residual_tolerance"]); need(v.volumeResidualTolerance==tolerance,"row tolerance differs from contract");
            if(cardiac) {
                const auto storage=text(r[@"storage_kind"]), law=text(r[@"pressure_law"]);
                need(storage=="absolute_volume" || storage=="storage_displacement","unknown storage coordinate");
                v.storageKind=storage=="absolute_volume" ? VascularStorageKind::absoluteVolume : VascularStorageKind::storageDisplacement;
                need(law=="linear_compliance" || law=="ventricular_elastance" || law=="atrial_elastance","unknown pressure law");
                v.pressureLaw=law=="linear_compliance" ? VascularPressureLaw::linearCompliance : law=="ventricular_elastance" ? VascularPressureLaw::ventricularElastance : VascularPressureLaw::atrialElastance;
                v.elastanceMin=number(r[@"elastance_min_pa_per_m3"]);v.elastanceMax=number(r[@"elastance_max_pa_per_m3"]);
                v.periodSeconds=number(r[@"period_seconds"]);v.activationStart=number(r[@"activation_start"]);
                v.activationEnd=number(r[@"activation_end"]);v.sourcePi=number(r[@"source_pi"]);
            }
            if(v.storageKind==VascularStorageKind::absoluteVolume){positive(v.referenceVolume);positive(v.initialVolume);}
            if(v.pressureLaw==VascularPressureLaw::linearCompliance)positive(v.compliance);
            positive(v.volumeScale);
            v.initialSpeciesAmounts=amounts(r[@"initial_species_mol"],result.species.size()); result.compartments.push_back(v);
        }
        need(result.compartments.size()>=2,"closed circulation requires at least two compartments");
        previous.clear();
        for (id item in list(root[@"connections"])) {
            NSDictionary* r=record(item,{"id","stable_identifier","from","to","resistance_pa_s_per_m3","inertance_pa_s2_per_m3","initial_flow_m3_per_s","flow_scale_m3_per_s","pressure_scale_pa","flow_residual_tolerance"}, cardiac ? std::initializer_list<const char*>{"flow_law","orifice_coefficient_m3_per_s_sqrt_pa"} : std::initializer_list<const char*>{});
            VascularConnectionSource v; v.stableIdentifier=identity(r,previous);
            v.fromCompartment=identifier(r[@"from"]);v.toCompartment=identifier(r[@"to"]);
            v.resistance=number(r[@"resistance_pa_s_per_m3"]);v.inertance=number(r[@"inertance_pa_s2_per_m3"]);v.initialFlow=number(r[@"initial_flow_m3_per_s"]);
            v.flowScale=number(r[@"flow_scale_m3_per_s"]);v.pressureScale=number(r[@"pressure_scale_pa"]);
            v.flowResidualTolerance=number(r[@"flow_residual_tolerance"]); need(v.flowResidualTolerance==tolerance,"row tolerance differs from contract");
            if(cardiac) {
                const auto law=text(r[@"flow_law"]);
                need(law=="resistance_inertance" || law=="one_way_orifice","unknown flow law");
                v.flowLaw=law=="resistance_inertance" ? VascularFlowLaw::resistanceInertance : VascularFlowLaw::oneWayOrifice;
                v.orificeCoefficient=number(r[@"orifice_coefficient_m3_per_s_sqrt_pa"]);
            }
            if(v.flowLaw==VascularFlowLaw::resistanceInertance)positive(v.resistance);
            positive(v.flowScale);positive(v.pressureScale); need(v.inertance>=0,"negative inertance");
            result.connections.push_back(v);
        }
        previous.clear();
        for (id item in list(root[@"tissue_reservoirs"])) {
            NSDictionary* r=record(item,{"id","stable_identifier","anatomical_region_id","physical_volume_owner_id","volume_m3","initial_species_mol"});
            VascularTissueSource v; v.stableIdentifier=identity(r,previous); owner(r);
            v.anatomicalIdentifier=text(r[@"anatomical_region_id"]);v.volume=number(r[@"volume_m3"]);positive(v.volume);
            v.initialSpeciesAmounts=amounts(r[@"initial_species_mol"],result.species.size());result.tissues.push_back(v);
        }
        previous.clear();
        for (id item in list(root[@"exchanges"])) {
            NSDictionary* r=record(item,{"id","stable_identifier","compartment","tissue_reservoir","species","clearance_m3_per_s","partition_coefficient"});
            VascularExchangeSource v; v.stableIdentifier=identity(r,previous);v.compartment=identifier(r[@"compartment"]);v.tissue=identifier(r[@"tissue_reservoir"]);v.species=identifier(r[@"species"]);
            v.permeabilitySurface=number(r[@"clearance_m3_per_s"]);v.partitionCoefficient=number(r[@"partition_coefficient"]);
            positive(v.permeabilitySurface);positive(v.partitionCoefficient); result.exchanges.push_back(v);
        }
        // compileWorld owns endpoint/incidence validation and native admission.
        // The result is assigned only after all source parsing succeeds.
        output=std::move(result);return true;
    } catch(const std::exception& exception) {
        if(error) *error=exception.what();return false;
    }}
}
}
