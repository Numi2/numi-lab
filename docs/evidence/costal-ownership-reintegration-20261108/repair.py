from pathlib import Path
for root in (Path("/Users/numi/numi-lab-tracker-step-rollback-20261108"),Path("/Users/numi/numi-lab-reconciliation-20261108")):
 p=root/"CMakeLists.txt";s=p.read_text();old="${CMAKE_CURRENT_SOURCE_DIR}/matter/materials/silicone.nmatter";new="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/material_column_silicone.nmatter";assert s.count(old)==11;s=s.replace(old,new);p.write_text(s)
 p=root/"apps/numanx_human_matter_owner_probe.mm";s=p.read_text();old="""    id<MTLFunction> function = [library newFunctionWithName:name];
    require(function != nil, "missing Metal function " +
        std::string{name.UTF8String});
    NSError* error = nil;
""";new="""    NSError* error = nil;
    id<MTLFunction> function = nil;
    if ([name isEqualToString:@\"mr_numi_human_stand_step\"]) {
        MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
        const bool disabled = false;
        const float distance = 0.0f;
        [constants setConstantValue:&disabled type:MTLDataTypeBool atIndex:0u];
        [constants setConstantValue:&disabled type:MTLDataTypeBool atIndex:2u];
        [constants setConstantValue:&distance type:MTLDataTypeFloat atIndex:8u];
        [constants setConstantValue:&disabled type:MTLDataTypeBool atIndex:9u];
        [constants setConstantValue:&disabled type:MTLDataTypeBool atIndex:18u];
        function = [library newFunctionWithName:name constantValues:constants error:&error];
    } else {
        function = [library newFunctionWithName:name];
    }
    require(function != nil, "missing or unspecialized Metal function " +
        std::string{name.UTF8String});
    error = nil;
""";assert s.count(old)==1;s=s.replace(old,new);p.write_text(s)
