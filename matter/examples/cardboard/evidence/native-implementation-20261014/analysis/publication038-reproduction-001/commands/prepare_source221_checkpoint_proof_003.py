from pathlib import Path
import json,hashlib,shutil,difflib,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');base=q/'snapshots/source-composed-strict-periodic-freebox-210-buildfix-003';s=q/'snapshots/source-qualification-checkpoint-proof-221-003';d=q/'analysis/source221-qualification-checkpoint-proof-003'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(base/'frozen-source.json')=='477fe6943fb1c28e782af9ce0c1c383c013146ab12f422bc55523a6834597931'
f=json.loads((base/'frozen-source.json').read_text());s.mkdir(exist_ok=False);d.mkdir(exist_ok=False)
for n,h in f['source_sha256'].items():
 assert sha(base/n)==h,n
 p=s/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/n,p)
rel='matter/tools/cardboard_probe.mm';p=s/rel;t=p.read_text()
def replace(a,b):
 global t
 assert t.count(a)==1,(a,t.count(a));t=t.replace(a,b)
replace('    const bool compareActualEnvironment = true) {','    const bool compareActualEnvironment = true,\n    const bool allowQualificationSource = false) {')
start=t.index('    require(couponRunnerString(prediction, @"schema", "prediction.schema") ==',t.index('couponValidateSourceBoundDiagnosticRunner('))
end=t.index('    NSDictionary* const launch =',start)
old=t[start:end]
new='''    const bool diagnosticRunner =
        couponRunnerString(prediction, @"schema", "prediction.schema") ==
            "numi.cardboard.native-diagnostic-prediction.v1" &&
        couponRunnerString(prediction, @"run_mode", "prediction.run_mode") == "diagnostic" &&
        couponRunnerString(record, @"schema", "environment.schema") ==
            "numi.cardboard.native-diagnostic-environment-run.v1" &&
        couponRunnerString(record, @"producer", "environment.producer") ==
            "source-bound-diagnostic-runner" &&
        couponRunnerString(record, @"mode", "environment.mode") == "diagnostic";
    // A qualification run has no output-prefix map. It can supply accepted
    // history to a diagnostic replay only after the same exact contract checks.
    const bool qualificationSource = allowQualificationSource &&
        !compareActualEnvironment &&
        !couponOutputPathMapHasAnySetValue(expectedPaths) &&
        couponRunnerString(prediction, @"schema", "prediction.schema") ==
            "numi.cardboard.native-prediction.v1" &&
        couponRunnerString(prediction, @"run_mode", "prediction.run_mode") == "qualification" &&
        couponRunnerString(record, @"schema", "environment.schema") ==
            "numi.cardboard.native-environment-run.v1" &&
        couponRunnerString(record, @"producer", "environment.producer") ==
            "source-bound-native-runner" && record[@"mode"] == nil;
    require(diagnosticRunner || qualificationSource,
            "coupon replay requires an exact source-bound diagnostic record or an output-free qualification source");
'''
t=t[:start]+new+t[end:]
start=t.index('    require(couponRunnerString(contract, @"schema", "contract.schema") ==',t.index('couponValidateSourceBoundDiagnosticRunner('))
end=t.index('    NSArray* const keys =',start)
old=t[start:end]
new='''    require(couponRunnerString(contract, @"schema", "contract.schema") ==
                (qualificationSource
                    ? "numi.cardboard.effective-native-controls.v1"
                    : "numi.cardboard.effective-native-diagnostic-controls.v1") &&
                (qualificationSource || couponRunnerString(contract,
                    @"case_directory", "contract.case_directory") == caseText),
            "source-bound contract schema or case directory is invalid");
    const std::string contractSha = couponRunnerString(
        contract, @"sha256", "contract.sha256");
    NSMutableDictionary* digestPayload = [@{
        @"schema": contract[@"schema"], @"keys": contract[@"keys"],
        @"effective": contract[@"effective"]} mutableCopy];
    if (!qualificationSource) {
        require([contract[@"capture_prefixes"] isKindOfClass:[NSDictionary class]],
                "diagnostic source contract has no capture-prefix map");
        digestPayload[@"case_directory"] = contract[@"case_directory"];
        digestPayload[@"capture_prefixes"] = contract[@"capture_prefixes"];
    }
    require(contract.count == (qualificationSource ? 4u : 6u) &&
                isLowerSha256(contractSha) &&
                hexDigest(sha256(couponCanonicalDiagnosticJson(digestPayload))) == contractSha &&
                couponRunnerString(prediction,
                    @"effective_native_environment_contract_sha256",
                    "prediction.contract_sha256") == contractSha &&
                couponRunnerString(record, @"contract_sha256",
                    "environment.contract_sha256") == contractSha,
            "source-bound contract digest is invalid");
    id keysValue = contract[@"keys"];
    NSDictionary* const effective =
        [contract[@"effective"] isKindOfClass:[NSDictionary class]]
            ? contract[@"effective"] : nil;
    NSDictionary* const prefixes = qualificationSource ? @{} :
        ([contract[@"capture_prefixes"] isKindOfClass:[NSDictionary class]]
            ? contract[@"capture_prefixes"] : nil);
    require([keysValue isKindOfClass:[NSArray class]] && effective != nil &&
                prefixes != nil && [(NSArray*)keysValue count] >= 49u &&
                [(NSArray*)keysValue count] <= 53u,
            "source-bound contract registry or maps are missing");
'''
t=t[:start]+new+t[end:]
# The strict inner-forcing key was added to the Python registry in source209;
# old records omit it, current 53-key records must include it in exact order.
replace('    "NM_FEM_STATUS5_CAPTURE_ROOT",\n    "NM_FEM_TRUE_RESIDUAL_CERTIFY",','    "NM_FEM_STATUS5_CAPTURE_ROOT",\n    "NM_FEM_STRICT_INNER_FORCING",\n    "NM_FEM_TRUE_RESIDUAL_CERTIFY",')
replace('    const bool contractHasDampingKey = keys.count >= 52u;','    const bool contractHasDampingKey = keys.count >= 52u;\n    const bool contractHasStrictInnerKey = keys.count >= 53u;')
replace('            if (expectedIndex == keyIndex) {','            if (expectedKey == "NM_FEM_STRICT_INNER_FORCING" &&\n                !contractHasStrictInnerKey)\n                continue;\n            if (expectedIndex == keyIndex) {')
replace('the exact 49/50/51/52-key source contract','the exact 49/50/51/52/53-key source contract')
replace('            sourceRunnerCase, {}, sourceOutputPaths, false);','            sourceRunnerCase, {}, sourceOutputPaths, false, true);')
# CPU-only admission diagnostic uses the exact production validator before
# argument parsing, compilation, Metal-device construction, or any state mutation.
replace('''            arguments = parseArguments(argc, argv);''','''            if (argc == 3 && std::string_view(argv[1]) ==
                    "--verify-coupon-source-runner") {
                const auto paths = numi::cardboard::coupon_resume::couponRuntimePathMap({});
                const auto proof = couponValidateSourceBoundDiagnosticRunner(
                    std::filesystem::path(argv[2]), {}, paths, false, true);
                std::cout << "{\\"schema\\":\\"numi.cardboard.coupon-source-runner-proof-check.v1\\","
                          << "\\"accepted\\":true,\\"native_runtime_created\\":false,"
                          << "\\"prediction_sha256\\":\\"" << proof.predictionSha256
                          << "\\",\\"contract_sha256\\":\\"" << proof.contractSha256 << "\\"}\\n";
                return 0;
            }
            arguments = parseArguments(argc, argv);''')
# array has a declared bound; increment only its exact registry declaration.
import re
matches=list(re.finditer(r'(std::array<std::string_view,\s*)(\d+)(u>\s+kSource161DiagnosticEnvironmentKeys)',t))
assert len(matches)==1,[(m.group(),m.span()) for m in matches]
m=matches[0];t=t[:m.start(2)]+str(int(m.group(2))+1)+t[m.end(2):]
p.write_text(t)
f['source_sha256'][rel]=sha(p)
f['source221_qualification_source_proof']={'parent_snapshot':str(base),'parent_frozen_sha256':sha(base/'frozen-source.json'),'changed':[rel],'physical_operator_or_acceptance_changed':False,'reason':'Read-only qualification runner proof supports source545 checkpoint-to-diagnostic replay; exact strict-inner key registry corrected. Historical checkpoint metadata unchanged.'}
(s/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n')
(d/'source-overlay.patch').write_text(''.join(difflib.unified_diff((base/rel).read_text().splitlines(True),t.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel)))
r={'schema':'numi.cardboard.source221-overlay.v1','snapshot':str(s),'frozen_sha256':sha(s/'frozen-source.json'),'patch_sha256':sha(d/'source-overlay.patch'),'changed':[rel],'gpu_execution':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'overlay.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
