from pathlib import Path
import difflib
import hashlib
import json

B = Path('/Users/home/cardboard-implementation-20261014')
S = B / 'snapshots/source-right-grip-telemetry-157'
O = B / 'analysis/source163-rejected-constitutive-export-overlay-001'
O.mkdir(exist_ok=False)
relative = 'matter/tools/cardboard_probe.mm'
base = (S / relative).read_text()
needle = '        writeArena("candidate_nodes", snapshot.femRegionalGalerkinLinearizationNodes);\n'
assert base.count(needle) == 1
added = '''        // These arrays were already read back under Runtime's existing
        // regional 64 MiB bound. Preserve the exact AP linearization inputs
        // instead of leaving them only in the in-memory snapshot. This is
        // accepted constitutive history, never a current projected state.
        writeArena("linearization_accepted_material_history",
                   snapshot.femRegionalGalerkinLinearizationMaterialState);
        writeArena("linearization_scheduler_exponents",
                   snapshot.femRegionalGalerkinLinearizationSchedulerExponent);
        writeArena("linearization_F_high",
                   snapshot.femRegionalGalerkinLinearizationLocalFHigh);
        writeArena("linearization_F_low",
                   snapshot.femRegionalGalerkinLinearizationLocalFLow);
'''
candidate = base.replace(needle, needle + added)
needle2 = '            << "  \\\"restart_augmentation\\\": {\\n"\n'
assert candidate.count(needle2) == 1
added2 = '''            << "  \\\"linearization_constitutive_inputs\\\": {\\n"
            << "    \\\"available\\\": "
            << (snapshot.femRegionalGalerkinDiagnosticsAvailable ? "true" : "false") << ",\\n"
            << "    \\\"material_state_stride\\\": " << snapshot.materialStateStride << ",\\n"
            << "    \\\"tetrahedra_per_environment\\\": "
            << compiled.world.dispatch.tetrahedronCount << ",\\n"
            << "    \\\"history_kind\\\": \\\"accepted_history_at_AP_linearization_not_current_projected_state\\\",\\n"
            << "    \\\"F_layout\\\": \\\"environment_tetrahedron_row_float4_high_plus_low_in_material_frame\\\",\\n"
            << "    \\\"scheduler_layout\\\": \\\"environment_tetrahedron_uint32_rate_exponent\\\",\\n"
            << "    \\\"exact_newton_correction_available\\\": false,\\n"
            << "    \\\"perturbed_projected_states_available\\\": false\\n  },\\n"
'''
candidate = candidate.replace(needle2, added2 + needle2)
p = O / 'candidate' / relative
p.parent.mkdir(parents=True)
p.write_text(candidate)
patch = ''.join(difflib.unified_diff(base.splitlines(keepends=True), candidate.splitlines(keepends=True),
                                     fromfile='a/'+relative, tofile='b/'+relative))
(O / 'overlay.patch').write_text(patch)
(O / 'prepare.py').write_bytes(Path(__file__).read_bytes())
(O / 'preparation.json').write_text(json.dumps({'base':str(S), 'file':relative,
    'base_sha256':hashlib.sha256(base.encode()).hexdigest(),
    'candidate_sha256':hashlib.sha256(candidate.encode()).hexdigest(),
    'patch_sha256':hashlib.sha256(patch.encode()).hexdigest(),
    'physics_changed':False,'extra_gpu_readback_bytes':0}, indent=2)+'\n')
print(O)
