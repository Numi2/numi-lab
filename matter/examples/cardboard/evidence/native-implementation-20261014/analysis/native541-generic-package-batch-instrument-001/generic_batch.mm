#define main cardboard_legacy_capture_main
#include "/Users/n/cardboard-implementation-20261014-host-macmini/snapshots/source-cohesive-quadrature-area-205/matter/tools/cardboard_bilinear15_capture_diagnostic_check.mm"
#undef main

int main(int argc,char** argv) { @autoreleasepool { try {
    need(argc==8 && (std::string(argv[1])=="--cpu-admission" ||
                    std::string(argv[1])=="--native"),
         "usage: source-bound-batch --cpu-admission|--native PACKAGE SHA INPUT.json SHA OUTDIR CAPTURE_SHA");
    const bool native=std::string(argv[1])=="--native";
    const std::filesystem::path packagePath(argv[2]),inputPath(argv[4]),outdir(argv[6]);
    const std::string packageSHA=sha256(packagePath),inputSHA=sha256(inputPath),captureSHA(argv[7]);
    need(packageSHA==argv[3] && inputSHA==argv[5],"package/input SHA differs from explicit CLI binding");
    need(captureSHA.size()==64u,"capture SHA must be explicit");
    NSDictionary* input=readJSONDocument(inputPath);
    need(stringValue(field(input,"schema"),"schema")=="numi.cardboard.source-bound-native-batch.v1",
         "unexpected batch input schema");
    need(stringValue(field(input,"source_capture_sha256"),"capture SHA")==captureSHA &&
         stringValue(field(input,"compiled_package_sha256"),"package SHA")==packageSHA,
         "input source-capture/package binding mismatch");
    CompiledWorld world;std::string error;
    need(readPackage(packagePath,world,nullptr,&error),"exact compiled package rejected: "+error);
    need(world.materials.size()>=2u,"package omits paper materials");
    NSArray* params=(NSArray*)field(input,"parameters_by_material");
    need([params isKindOfClass:[NSArray class]] && params.count==2u,"two parameter rows required");
    const std::uint32_t required=NM_MATERIAL_PAPER_COMPENSATED_ARITHMETIC|
        NM_MATERIAL_GENERIC_COMPENSATED_IMPLICIT_PROJECTION|
        NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION;
    for(std::size_t m=0;m<2u;++m) {
        const auto& mat=world.materials[m];
        need(mat.stateCount==15u && mat.parameterCount==28u &&
             mat.localNewtonIterations==32u && (mat.flags&required)==required,
             "exact package must preserve accurate compensated paper policy and32 iterations");
        const auto p=flatVector(params[m],28u,"source parameters");
        for(std::size_t k=0;k<28u;++k)
            need(std::bit_cast<std::uint32_t>(p[k])==
                 std::bit_cast<std::uint32_t>(world.parameters[mat.parameterOffset+k].valueAndBounds.x),
                 "source capture parameter words differ from exact package");
    }
    NSArray* entries=(NSArray*)field(input,"rows");
    need([entries isKindOfClass:[NSArray class]] && entries.count>0u && entries.count<=4096u,
         "batch row count must be1..4096");
    std::vector<Record> rows;rows.reserve(entries.count);
    for(id entry:entries) {
        NSDictionary* d=(NSDictionary*)entry;
        need([d isKindOfClass:[NSDictionary class]],"batch record must be an object");
        Record r;
        const double mat=number(field(d,"material_index"),"material index");
        const double env=number(field(d,"environment"),"environment");
        const double tet=number(field(d,"tet_id"),"tet id");
        need(mat==0.0||mat==1.0,"only bound paper material rows supported");
        need(env>=0.0&&env<=1.0&&env==std::floor(env),"environment metadata invalid");
        need(tet>=0.0&&tet<=double(UINT32_MAX)&&tet==std::floor(tet),"tet metadata invalid");
        r.material=std::uint32_t(mat);r.env=std::uint32_t(env);r.tet=std::uint32_t(tet);
        r.materialName=r.material==0u?kLinerName:kMediumName;
        r.input=flatArray<47>(field(d,"packed_input_f32"),"input");
        r.accepted=flatArray<15>(field(d,"accepted_state_f32"),"accepted history");
        need(std::all_of(r.input.begin(),r.input.end(),[](float x){return std::isfinite(x);})&&
             std::all_of(r.accepted.begin(),r.accepted.end(),[](float x){return std::isfinite(x);})&&
             r.input[45]>0.0f&&r.input[46]>0.0f,"nonfinite/nonpositive packed inputs");
        rows.push_back(r);
    }
    if(!native) {
        std::cout<<"{\"record\":\"admission\",\"rows\":"<<rows.size()
          <<",\"package_sha256\":\""<<packageSHA<<"\",\"input_sha256\":\""<<inputSHA
          <<"\",\"source_capture_sha256\":\""<<captureSHA<<"\",\"native_gpu_executed\":false}\n";
        return 0;
    }
    need(std::filesystem::create_directory(outdir),"native output directory must be new");
    Metal metal;
    const auto result=runBatch(metal,world,"exact_package_budget32",rows,outdir,inputSHA,true);
    std::size_t invalid=0u;
    for(std::size_t i=0;i<rows.size();++i)
        invalid+=(result[i*kOutputStride]!=1.0f ||
                  result[i*kOutputStride+1u]!=1.0f ||
                  result[i*kOutputStride+2u]!=1.0f);
    std::ofstream receipt(outdir/"input-binding.json");
    receipt<<"{\"input_sha256\":\""<<inputSHA<<"\",\"package_sha256\":\""<<packageSHA
      <<"\",\"source_capture_sha256\":\""<<captureSHA<<"\",\"rows\":"<<rows.size()
      <<",\"invalid_rows\":"<<invalid<<",\"physical_transactions\":0,\"full_solver_qualification\":false}\n";
    receipt.flush();need(receipt.good(),"input binding write failed");
    std::cout<<"{\"record\":\"dispatch_complete\",\"rows\":"<<rows.size()<<",\"invalid_rows\":"<<invalid
      <<",\"physical_transactions\":0,\"full_solver_qualification\":false}\n";
    return invalid==0u?0:2;
} catch(const std::exception& e) {std::cerr<<"FAIL generic exact-package paper replay: "<<e.what()<<'\n';return 1;} } }
