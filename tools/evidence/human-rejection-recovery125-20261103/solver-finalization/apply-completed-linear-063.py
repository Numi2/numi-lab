from pathlib import Path
R=Path('/Users/numi/Work/skin-fat-native-integration-209')
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
import subprocess,shutil
running=subprocess.check_output(['pgrep','-f','/build-native/bin/numi-human-native'],text=True) if subprocess.run(['pgrep','-f','/build-native/bin/numi-human-native'],stdout=subprocess.DEVNULL).returncode==0 else ''
if running.strip():raise RuntimeError('native owner active; no edit')
files=['matter/src/runtime.mm','matter/src/metal/fgmres.metalinc','matter/tools/fgmres_true_residual_native_check.mm']
for rel in files:
 dst=D/'completed-linear-builders-063-preimages'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/rel,dst)
p=R/files[1];s=p.read_text()
old='''        } else if (mode == 2u) {
            skip = restartCycle > 0u && latched &&
                isfinite(diagnostics.w) && diagnostics.w == 0.0f;
        }'''
new='''        } else if (mode == 2u) {
            skip = restartCycle > 0u && latched &&
                isfinite(diagnostics.w) && diagnostics.w == 0.0f;
        } else if (mode == 3u) {
            // Preconditioner-only producers may retain scratch only after
            // the fresh begin certifies complete nonlinear convergence.
            // The host always retains the final Newton refresh.
            const float nonlinear = states[0].nonlinear.w;
            skip = latched && isfinite(nonlinear) && nonlinear > 0.5f;
        }'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('// retains direct dispatch whenever this experiment is disabled or unsupported.','// retains direct dispatch whenever this experiment is disabled or unsupported.\n// Mode 3 suppresses intermediate preconditioner-only producers after fresh\n// complete nonlinear convergence; invalid state/status always retains work.')
p.write_text(s)
p=R/files[0];s=p.read_text()
marker='''            if (buildFEMPreconditionerThisCycle) {
                dispatchThreads(
                    "nm_fgmres_build_preconditioner", femNodeTotal, [&] {'''
replacement='''            // These builders follow the real current-RHS begin measurement.
            // Suppress only intermediate complete-Newton tails. Active linear
            // solves and the final refresh retain every producer and consumer.
            const bool completedLinearBuildersMaySkip =
                femCompletedCycleWorkEnabled &&
                nonlinearIteration + 1u < nonlinearIterationBudget &&
                !state.forceCompletedTangentCacheRebuild &&
                !state.femOperatorFDRichCaptureEnabled &&
                !femRegionalRAPCaptureOnly && !femCoupledCaptureRequested &&
                !state.femHumanGlobalCoarseRapCaptureEnabled;
            const auto dispatchLinearBuilderThreads = [&](
                const char* name, const NSUInteger count, const auto& bind,
                const WorkDomain domain) {
                if (completedLinearBuildersMaySkip) {
                    prepareFGMRESCycleWorkRange(
                        3u, (count + 255u) / 256u, restartCycle);
                    if (dispatchIndirectAtOffset(name,
                            state.fgmresCycleWorkDispatch, 256u, 0u, bind))
                        return;
                }
                dispatchThreads(name, count, bind, domain);
            };
            const auto dispatchLinearBuilderGroups = [&](
                const char* name, const NSUInteger groups, const auto& bind) {
                if (completedLinearBuildersMaySkip) {
                    prepareFGMRESCycleWorkRange(3u, groups, restartCycle);
                    if (dispatchIndirectAtOffset(name,
                            state.fgmresCycleWorkDispatch, 32u, 0u, bind))
                        return;
                }
                dispatchGroups32(name, groups, bind);
            };
            if (buildFEMPreconditionerThisCycle) {
                dispatchLinearBuilderThreads(
                    "nm_fgmres_build_preconditioner", femNodeTotal, [&] {'''
assert s.count(marker)==1;s=s.replace(marker,replacement)
parse_marker='''            if (const char* zeroAction = std::getenv(
                    "NM_FGMRES_SKIP_COMPLETED_ZERO_OPERATOR_ACTION")) {'''
parse_new='''            bool completedLinearBuildersRequested = false;
            if (const char* setting = std::getenv(
                    "NM_FGMRES_SKIP_COMPLETED_LINEAR_BUILDERS")) {
                if (std::strcmp(setting, "1") == 0)
                    completedLinearBuildersRequested = true;
                else if (std::strcmp(setting, "0") != 0) {
                    diagnostics.message =
                        "NM_FGMRES_SKIP_COMPLETED_LINEAR_BUILDERS must be 0 or 1";
                    ownership->preDynamicsOpen = false;
                    return diagnostics;
                }
            }
            if (completedLinearBuildersRequested && !femCompletedCycleWorkRequested) {
                diagnostics.message =
                    "completed linear builders require completed-cycle work admission";
                ownership->preDynamicsOpen = false;
                return diagnostics;
            }
'''+parse_marker
assert s.count(parse_marker)==1;s=s.replace(parse_marker,parse_new)
s=s.replace('''                femCompletedCycleWorkEnabled &&
                nonlinearIteration + 1u < nonlinearIterationBudget &&''',
'''                completedLinearBuildersRequested && femCompletedCycleWorkEnabled &&
                nonlinearIteration + 1u < nonlinearIterationBudget &&''')
start=s.index('const bool completedLinearBuildersMaySkip')
end=s.index('            fgmresColumnDispatch =',start)
section=s[start:end]
assert section.count('dispatchGroups32(blockCavitySchur')==1
section=section.replace('dispatchGroups32(blockCavitySchur','dispatchLinearBuilderGroups(blockCavitySchur')
assert section.count('dispatchGroups32("nm_vascular_factor_principal_block"')==1
section=section.replace('dispatchGroups32("nm_vascular_factor_principal_block"','dispatchLinearBuilderGroups("nm_vascular_factor_principal_block"')
s=s[:start]+section+s[end:];p.write_text(s)
p=R/files[2];s=p.read_text()
start=s.index('void checkCompletedCycleWorkRanges(')
end=s.index('    auto runPrepare =',start)
section=s[start:end].replace('        std::uint32_t expectedGroups;\n','        std::uint32_t expectedGroups;\n        float nonlinear{1.0f};\n')
section=section.replace('std::array<Scenario, 10u>','std::array<Scenario, 17u>')
marker='''        {"accum-unhealthy", 2u, 1u, 1.0f, 0.0f, 0u, failed, 1u, 1u},
'''
add='''        {"builders-complete", 3u, 0u, 1.0f, 0.0f, 0u, healthy, 1247u, 0u},
        {"builders-linear-only", 3u, 0u, 1.0f, 0.0f, 0u, healthy, 1u, 1u, 0.0f},
        {"builders-active", 3u, 0u, 0.0f, 0.0f, 0u, healthy, 1u, 1u},
        {"builders-invalid-latch", 3u, 0u,
            std::numeric_limits<float>::quiet_NaN(), 0.0f, 0u, healthy, 1u, 1u},
        {"builders-invalid-outer", 3u, 0u, 1.0f, 0.0f, 0u, healthy, 1u, 1u,
            std::numeric_limits<float>::quiet_NaN()},
        {"builders-infinite-outer", 3u, 0u, 1.0f, 0.0f, 0u, healthy, 1u, 1u,
            std::numeric_limits<float>::infinity()},
        {"builders-unhealthy", 3u, 0u, 1.0f, 0.0f, 0u, failed, 1u, 1u},
'''
assert section.count(marker)==1;section=section.replace(marker,marker+add)
s=s[:start]+section+s[end:]
needle='''        selectedState.diagnostics.w = scenario.used;
'''
pos=s.index(needle,start);s=s[:pos]+s[pos:].replace(needle,needle+'        selectedState.nonlinear.w = scenario.nonlinear;\n',1)
p.write_text(s)
print('applied completed linear builder suppression')

