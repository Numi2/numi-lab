#pragma once
#include <array>
#include <map>
#include <set>
#include <string>

// Load-time anatomical admission only. CVSim21 retains sole ownership of
// compartment blood volumes, pressures and flows in the resident Matter state.
inline void numiHumanVerifyVascularAnatomy(NSDictionary* bindings, NSDictionary* provenance,
    const LoadedTorsoAnatomy& anatomy) {
    const std::array<const char*,21> names{{
        "ascending_aorta","brachiocephalic_arteries","upper_body_arteries","upper_body_veins",
        "superior_vena_cava","descending_thoracic_aorta","abdominal_aorta","renal_arteries",
        "renal_veins","splanchnic_arteries","splanchnic_veins","lower_body_arteries",
        "lower_body_veins","abdominal_veins","inferior_vena_cava","right_atrium",
        "right_ventricle","pulmonary_arteries","pulmonary_veins","left_atrium","left_ventricle"}};
    std::map<unsigned,std::set<unsigned>> direct{{1,{6}},{5,{10}},{6,{8}},{7,{9}},
        {15,{11}},{16,{318}},{17,{319}},{20,{320}},{21,{321}}};
    std::set<unsigned> available;
    for(const auto& surface:anatomy.records) {
        require(available.insert(surface.stableId).second,"anatomy has duplicate stable IDs");
        if(surface.layer==5)direct[18].insert(surface.stableId);
        if(surface.layer==6)direct[19].insert(surface.stableId);
    }
    require(!direct[18].empty()&&!direct[19].empty(),"circulation lacks registered pulmonary vessels");
    NSArray* rows=bindings[@"vascular_compartment_bindings"];
    NSDictionary* sourceMap=provenance[@"source_id_map"];
    require([rows isKindOfClass:NSArray.class]&&rows.count==names.size()&&
        [sourceMap isKindOfClass:NSDictionary.class],"circulation requires all 21 anatomical compartment associations");
    for(unsigned index=0;index<names.size();++index) {
        NSDictionary* row=rows[index];
        require([row isKindOfClass:NSDictionary.class],"invalid vascular anatomical association");
        NSNumber* compartment=row[@"cvsim_compartment_id"];
        NSString* name=row[@"name"],*region=row[@"anatomical_region_id"],*kind=row[@"binding_kind"];
        NSString* expectedName=[NSString stringWithUTF8String:names[index]];
        NSString* expectedRegion=[@"source_aggregate:CVSim21:" stringByAppendingString:expectedName];
        require([compartment isKindOfClass:NSNumber.class]&&[compartment doubleValue]==index+1&&
            [name isKindOfClass:NSString.class]&&[name isEqualToString:expectedName]&&
            [region isKindOfClass:NSString.class]&&[region isEqualToString:expectedRegion],
            "vascular anatomical association differs from the admitted CVSim21 order");
        const bool named=direct.contains(index+1);
        require([kind isKindOfClass:NSString.class]&&
            [kind isEqualToString:named?@"named_vessel_or_chamber":@"aggregate_region"],
            "vascular anatomical association confuses a named structure with a reduced regional bed");
        NSArray* values=row[@"stable_ids"];
        require([values isKindOfClass:NSArray.class],"vascular association is missing anatomical stable IDs");
        std::set<unsigned> ids;
        for(id value in values) {
            require([value isKindOfClass:NSNumber.class]&&[value doubleValue]==[value unsignedIntValue],
                "invalid vascular anatomical stable ID");
            const unsigned idValue=[value unsignedIntValue];
            NSDictionary* source=sourceMap[[NSString stringWithFormat:@"%u",idValue]];
            require(available.contains(idValue)&&ids.insert(idValue).second&&
                [source isKindOfClass:NSDictionary.class]&&[source[@"source_member"] isKindOfClass:NSString.class]&&
                [source[@"source_sha256"] isKindOfClass:NSString.class],
                "vascular association refers to an absent or duplicated source identity");
        }
        if(named)require(ids==direct.at(index+1),"named vascular association omits or substitutes registered anatomy");
        else {
            require(index==11||index==12||!ids.empty(),"vascular region lacks its representative anatomy");
            require([row[@"unresolved_geometry"] isKindOfClass:NSString.class],
                "reduced vascular bed must declare its unresolved distal anatomy");
        }
        // Optional appended named vessel surfaces belong to these same
        // regional hydraulic states. Reject a receipt that leaves one orphaned.
        for(NSString* key in sourceMap) {
            NSDictionary* source=sourceMap[key];
            NSNumber* owner=source[@"vascular_compartment_id"];
            if(!owner)continue;
            require([owner isKindOfClass:NSNumber.class]&&[owner doubleValue]==[owner unsignedIntValue]&&
                [owner unsignedIntValue]>=1&&[owner unsignedIntValue]<=21,
                "named peripheral vessel has an invalid circulation owner");
            if([owner unsignedIntValue]==index+1)require(ids.contains([key intValue]),
                "named peripheral vessel is absent from its circulation association");
        }
    }
}
