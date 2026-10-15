#!/usr/bin/env python3
import hashlib, json, pathlib, subprocess, sys
base=pathlib.Path('/Users/home/cardboard-implementation-20261014/solver/regional-bending-captured-cycle-057')
root=pathlib.Path('/Users/home/cardboard-implementation-20261014')
snapshot=root/'snapshots/contact-damage-bending-037'
analysis=base/'analysis-057'
build=analysis/'replay-build-037'
replay=analysis/'replay/cardboard_solver_pair_replay'
pred=json.loads((base/'prediction.json').read_text())
receipt=json.loads((base/'receipt.json').read_text())
run055=root/'solver/regional-bending-complete-cycle-055'
pred055=json.loads((run055/'prediction.json').read_text())

def sha(p):
 return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def normalize_argv(argv):
 out=[];i=0
 while i<len(argv):
  if argv[i] in {'--fgmres-iterations','--output'}:
   out.extend([argv[i],'<normalized>']);i+=2
  else:out.append(argv[i]);i+=1
 return out
source_hash_results=[]
for name,expected in pred['inputs_sha256'].items():
 if not name.startswith('source/'):continue
 path=snapshot/name.removeprefix('source/')
 actual=sha(path) if path.is_file() else None
 source_hash_results.append({'relative_path':name.removeprefix('source/'),'expected_sha256':expected,'actual_sha256':actual,'matches':actual==expected})
base_replay=build/'numi-matter-cardboard-solver-replay'
probe=pathlib.Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-037/numi-matter-cardboard-probe')
metallib=pathlib.Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-037/shaders/NumiMatter.metallib')
source_files={
 'snapshot_CMakeLists':snapshot/'matter/CMakeLists.txt',
 'solver_reference':snapshot/'matter/tools/cardboard_solver_reference.hpp',
 'paper_reference':snapshot/'matter/tools/cardboard_paper_reference.hpp',
 'native_replay_source':snapshot/'matter/tools/cardboard_solver_replay.cpp',
 'native_capture_analyzer_base':snapshot/'matter/tools/cardboard_solver_capture_analysis.py',
 'native_fem_common':snapshot/'matter/src/metal/common.metalinc',
 'hill_compensated_system':snapshot/'matter/include/numi/matter/hill_compensated_system.h',
 'pair_replay_source':analysis/'replay/cardboard_solver_pair_replay.cpp',
 'pair_replay_executable':replay,
 'pair_aware_analyzer':analysis/'analyze_capture_pair.py',
 'summary_analyzer':analysis/'summarize_057.py',
 'native_probe_executable':probe,
 'native_metallib':metallib,
 'native_source_build_receipt':pathlib.Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-037/source-build.json'),
 'cpu_replay_cmake_cache':build/'CMakeCache.txt',
 'cpu_replay_executable':base_replay,
 'capture':base/'capture.jsonl',
 'prediction':base/'prediction.json',
 'run_receipt':base/'receipt.json',
 '055_prediction':run055/'prediction.json',
 'paired_replay_output':analysis/'replay-results-candidate-tangent/replay-output.jsonl',
 'pair_aware_tet_audit':analysis/'replay-results-candidate-tangent/tetrahedron-audit.jsonl',
 'pair_aware_assembly_audit':analysis/'replay-results-candidate-tangent/assembled-audit.json',
 'analysis_results':analysis/'replay-results-candidate-tangent/capture057-layer-analysis.json',
}
files={k:{'path':str(p),'sha256':sha(p) if p.is_file() else None,'exists':p.is_file()} for k,p in source_files.items()}
checks=subprocess.run([str(base_replay),'--check'],text=True,capture_output=True,check=False)
report={
 'schema':'numi.cardboard.capture057.analysis-provenance.v1',
 'capture_run':{'run_directory':str(base),'prediction_schema':pred['schema'],'receipt_schema':receipt['schema'],
                'source_snapshot':pred['source_snapshot'],'native_command':pred['argv'],
                'environment':pred['environment'],'receipt_exit_code':receipt['exit_code'],
                'receipt_input_integrity':receipt['source_inputs_unchanged'],'changed_inputs':receipt['changed_inputs'],
                'accepted_steps':1,'failed_control_step_index':1,'failed_newton_iteration':19,
                'terminal_failure_residual':2.9222736e-05,'run_residual_tolerance':1e-6,
                'terminal_status_code_environment0':10,'terminal_status_code_environment1':0,
                'fgmres_max_iterations_seen':289,'capture_selection':{'root':1,'newton_iteration':0}},
 '055_comparison':{'normalized_argv_identical_except_linear_ceiling_and_output':normalize_argv(pred['argv'])==normalize_argv(pred055['argv']),
                   '055_source_snapshot':pred055['source_snapshot'],'057_source_snapshot':pred['source_snapshot'],
                   '055_fgmres_iterations':pred055['argv'][pred055['argv'].index('--fgmres-iterations')+1],
                   '057_fgmres_iterations':pred['argv'][pred['argv'].index('--fgmres-iterations')+1],
                   'same_physical_protocol_claim_source':'sealed prediction argv comparison; native physical parameters and path flags are identical'},
 'source_integrity':{'prediction_source_input_count':len(source_hash_results),
                     'matched_count':sum(x['matches'] for x in source_hash_results),
                     'mismatch_count':sum(not x['matches'] for x in source_hash_results),
                     'mismatches':[x for x in source_hash_results if not x['matches']],
                     'native_source_build_receipt_sha256':sha(source_files['native_source_build_receipt']),
                     'native_metallib_sha256':sha(metallib),'native_probe_sha256':sha(probe),
                     'snapshot_037_path':str(snapshot)},
 'fp64_replay':{'fresh_cpu_build_dir':str(build),'cmake_source':pred['source_snapshot']+'/matter',
                'build_target':'numi-matter-cardboard-solver-replay','build_command':'cmake --build <analysis-057/replay-build-037> --target numi-matter-cardboard-solver-replay --parallel 2',
                'oracle_self_check_returncode':checks.returncode,'oracle_self_check_stdout':checks.stdout.strip(),
                'pair_aware_extensions':'analysis-local C++ CLI reads 9 high + 9 low F words and 13 high + 13 low candidate-state words, converts each word to FP32 then widens and sums as FP64; exact accepted-state and material parameter FP32 roundtrips are separately widened',
                'source_cpp_compiler':'AppleClang 21.0.0.21000099, C++17 -O3 -Wall -Wextra -Wpedantic -Werror',
                'physical_validation':False},
 'files':files,
 'scope_limits':['The captured data is only control-step root 1, Newton iteration 0, before the selected FGMRES solve.',
                 'The run later failed at Newton iteration 19; this capture does not identify that terminal cause.',
                 'Finite-difference perturbations cross the Hill trial-yield threshold for many paper elements; near-threshold differences are separately labeled and not admitted as smooth derivatives.',
                 'The independent reference is a CPU FP64 implementation of the documented 13-state ideal-Hill paper law; it is not a physical validation result.',
                 'Environment 1 has zero correction/operator action and is a stationary control, not an active second deformation test.']
}
out=analysis/'analysis-provenance.json'
out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'manifest':str(out),'source_inputs':report['source_integrity']['prediction_source_input_count'],'matched':report['source_integrity']['matched_count'],'mismatch_count':report['source_integrity']['mismatch_count'],'oracle_self_check_returncode':checks.returncode,'files_missing':[k for k,v in files.items() if not v['exists']]},indent=2))
