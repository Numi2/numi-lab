from pathlib import Path
import difflib,hashlib,importlib.util,json,shutil,subprocess,tempfile
b=Path('/Users/home/cardboard-implementation-20261014');old=b/'snapshots/source-contact-capture-and-resolved-material-sweep-095';new=b/'snapshots/source-current-newton-restart-augmentation-096';pkg=b/'analysis/restart-augmentation-304-005';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();prior=json.loads((old/'frozen-source.json').read_text());m=json.loads((pkg/'source-overlay.json').read_text())
assert h(pkg/'source-overlay.json')=='3384f6715fbdad52631bf52bb31d38b72e3eec71d6ce1b0413c8905ba4b556c9'
for rel,expected in m['overlay_sha256'].items():assert h(pkg/rel)==expected,rel
for rel,expected in prior['source_sha256'].items():assert h(old/rel)==expected,rel
assert all(r['exit_code']==0 for r in m['validation'])
assert not new.exists();shutil.copytree(old,new);logs={}
changes=['matter/include/numi/matter/matter.hpp','matter/src/metal/fgmres.metalinc','matter/src/runtime.mm']
for rel,name in zip(changes,['matter-header.diff','fgmres.diff','runtime.diff']):
 assert h(Path(m['base']['snapshot_path'])/rel)==m['base']['sha256'][rel]
 if rel!='matter/src/runtime.mm':assert h(old/rel)==m['base']['sha256'][rel]
 patch=(pkg/'patches'/name).read_text().splitlines(keepends=True);patch[0]='--- a/'+rel+'\n';patch[1]='+++ b/'+rel+'\n';result=subprocess.run(['patch','--batch','--forward','-p1'],input=''.join(patch),cwd=new,text=True,capture_output=True);logs[rel]=dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr);assert result.returncode==0,result.stderr+result.stdout
# Commutation proof for independent source095 capture repair and augmentation.
with tempfile.TemporaryDirectory() as t:
 root=Path(t);q=root/'matter/src/runtime.mm';q.parent.mkdir(parents=True);shutil.copy2(pkg/'overlay/matter/src/runtime.mm',q);p=b/'analysis/coupled-trial-capture-overlay-003/runtime.patch';result=subprocess.run(['patch','--batch','--forward','-p1','-i',str(p)],cwd=root,text=True,capture_output=True);assert result.returncode==0,result.stderr+result.stdout;assert h(q)==h(new/'matter/src/runtime.mm')
probe=new/'matter/tools/cardboard_probe.mm';original=probe.read_text();s=original
s=s.replace('if (!snapshot.femRegionalGalerkinDiagnosticsAvailable) return;','if (snapshot.femRegionalGalerkinModeCount == 0u &&\n            snapshot.femRegionalRestartBaseModeCount == 0u) return;',1)
needle='''            << "  \\"byte_order\\": \\"native_little_endian\\",\\n"'''
# Insert the metadata block immediately before the existing file-list emission.
needle='''            << "  \\"files\\": [\\n";'''
assert s.count(needle)==1,repr(needle)
replacement='''            << "  \\"regional_diagnostics_available\\": "
            << (snapshot.femRegionalGalerkinDiagnosticsAvailable ? "true" : "false") << ",\\n"
            << "  \\"regional_diagnostics_message\\": \\""
            << jsonEscape(snapshot.femRegionalGalerkinDiagnosticsMessage) << "\\",\\n"
            << "  \\"regional_readback_bytes\\": "
            << snapshot.femRegionalGalerkinDiagnosticReadbackBytes << ",\\n"
            << "  \\"restart_augmentation\\": {\\n"
            << "    \\"diagnostics_available\\": "
            << (snapshot.femRegionalRestartAugmentationDiagnosticsAvailable ? "true" : "false") << ",\\n"
            << "    \\"diagnostics_message\\": \\""
            << jsonEscape(snapshot.femRegionalRestartAugmentationDiagnosticsMessage) << "\\",\\n"
            << "    \\"base_mode_count\\": " << snapshot.femRegionalRestartBaseModeCount << ",\\n"
            << "    \\"augmented_mode_count\\": " << snapshot.femRegionalRestartAugmentedModeCount << ",\\n"
            << "    \\"control_step\\": " << snapshot.femRegionalRestartControlStep << ",\\n"
            << "    \\"physics_substep\\": " << snapshot.femRegionalRestartPhysicsSubstep << ",\\n"
            << "    \\"newton_iteration\\": " << snapshot.femRegionalRestartNewtonIteration << ",\\n"
            << "    \\"restart_cycle\\": " << snapshot.femRegionalRestartCycle << ",\\n"
            << "    \\"attempt_count\\": " << snapshot.femRegionalRestartAttemptCount << ",\\n"
            << "    \\"mode_valid\\": " << snapshot.femRegionalRestartModeValid << ",\\n"
            << "    \\"factor_valid\\": " << snapshot.femRegionalRestartAugmentedFactorValid << ",\\n"
            << "    \\"temporary_metal_readback_bytes\\": "
            << snapshot.femRegionalRestartDiagnosticReadbackBytes << ",\\n";
        const auto writeRestartScalar = [&](const char* name, const float value) {
            std::uint32_t word = 0u;
            std::memcpy(&word, &value, sizeof(word));
            metadata << "    \\"" << name << "\\": ";
            if (std::isfinite(value)) metadata << std::setprecision(9) << value;
            else metadata << "null";
            metadata << ",\\n    \\"" << name << "_f32_bits\\": " << word << ",\\n";
        };
        writeRestartScalar("target_scale", snapshot.femRegionalRestartTargetScale);
        writeRestartScalar("raw_norm_squared", snapshot.femRegionalRestartRawNormSquared);
        writeRestartScalar("mode_scale", snapshot.femRegionalRestartModeScale);
        metadata << "    \\"accepted_physical_state\\": false\\n  },\\n"
            << "  \\"files\\": [\\n";'''
s=s.replace(needle,replacement)
needle='''        metadata << "\\n  ]\\n}\\n";'''
assert s.count(needle)==1
replacement='''        if (snapshot.femRegionalRestartAugmentationDiagnosticsAvailable) {
            writeArena("restart_q", snapshot.femRegionalRestartModeQ);
            writeArena("restart_Aq", snapshot.femRegionalRestartOperatorAQ);
            writeArena("restart_pivoted_RAP", snapshot.femRegionalRestartPivotedRAP);
            writeArena("restart_pivots", snapshot.femRegionalRestartPivots);
        }
'''+needle
s=s.replace(needle,replacement);probe.write_text(s)
root=b/'analysis/restart-augmentation-probe-writer-096-001';root.mkdir(exist_ok=False);(root/'probe.patch').write_text(''.join(difflib.unified_diff(original.splitlines(keepends=True),s.splitlines(keepends=True),fromfile='a/matter/tools/cardboard_probe.mm',tofile='b/matter/tools/cardboard_probe.mm')));(root/'integration.json').write_text(json.dumps(dict(overlay_manifest_sha256=h(pkg/'source-overlay.json'),patch_results=logs,independent_runtime_patch_commutation=True,probe_base_sha256=prior['source_sha256']['matter/tools/cardboard_probe.mm'],probe_sha256=h(probe),scope='Diagnostic writer only, exact float words and nullable nonfinite scalars; no physical trajectory qualification.'),indent=2)+'\n')
spec=importlib.util.spec_from_file_location('bound',new/'matter/tools/cardboard_source_bound_build.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);bindings=module.source_bindings(new);scope='Source095 plus opt-in current-Newton restart augmentation and bounded generation/q/Aq/RAP readback with probe serialization. Physical operator, global tolerances, restart16 and material history unchanged. Compile/native qualification pending.';(new/'frozen-source.json').write_text(json.dumps({**prior,'base_snapshot':str(old),'source_sha256':bindings,'scope':scope},indent=2)+'\n');(b/'source-overlay-096.json').write_text(json.dumps(dict(schema='numi.cardboard.integrated-source-overlay.v1',base_snapshot=str(old),snapshot=str(new),packages=[dict(path=str(pkg),manifest_sha256=h(pkg/'source-overlay.json'))],integration_package=str(root),changed={p:dict(before=prior['source_sha256'].get(p),after=v) for p,v in bindings.items() if prior['source_sha256'].get(p)!=v},scope=scope),indent=2)+'\n');cmd=[v.replace(str(old),str(new)).replace('provenance-095','provenance-096') for v in json.loads((b/'build-command-095.json').read_text())];(b/'build-command-096.json').write_text(json.dumps(cmd,indent=2)+'\n');print('frozen096',len(bindings),'bindings',flush=True)
with (b/'build096-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
print('build096exit',code,flush=True);raise SystemExit(code)
