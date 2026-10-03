#!/usr/bin/env python3
"""Qualify an exact native instrument and author a bounded, executable-model study."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from weight_effect_model import initial_model, prediction, save


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True, help='owning Git checkout')
    parser.add_argument('--directory', type=Path, required=True, help='new durable directory outside the checkout')
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--instrument-build', type=Path, required=True, help='isolated build_neuron_instrument.py output')
    parser.add_argument('--calibration-seed', type=int, required=True, help='separate known-control condition')
    parser.add_argument('--model', type=Path, help='revised executable model parameters; defaults to the initial hypothesis')
    parser.add_argument('--purpose', choices=('exploration', 'confirmation', 'replication'), default='exploration')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 < seed < 2**32 for seed in [*args.seeds, args.calibration_seed]):
        parser.error('use distinct positive 32-bit seeds')
    if args.calibration_seed in args.seeds:
        parser.error('calibration and test conditions must be separate')
    runtime, directory, build = args.runtime.resolve(), args.directory.resolve(), args.instrument_build.resolve()
    if runtime in directory.parents:
        parser.error('keep durable scientific evidence outside the source checkout')
    receipt_path = build / 'build-receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt['schema'] != 'numi.science.native-instrument-build.v1':
        parser.error('unsupported native build receipt')
    for path, expected in {**receipt['inputs'], **receipt['outputs']}.items():
        if sha(path) != expected:
            parser.error('native source or build artifact changed: ' + path)
    binary = build / 'metalrobo_neuron_culture_probe'
    shader = build / 'NumiNeuron.metallib'
    if {str(binary), str(shader)} != set(receipt['outputs']):
        parser.error('build receipt does not bind the selected native binary and shader')
    model = json.loads(args.model.read_text()) if args.model else initial_model()
    predicted = prediction(model)
    if args.purpose == 'confirmation' and any(unit['seed'] in args.seeds for unit in model['training_units']):
        parser.error('confirmation requires conditions not used to fit the model')
    directory.mkdir(parents=True, exist_ok=False)
    instrument = Path(__file__).with_name('neuron_weight_instrument.py').resolve()
    model_program = Path(__file__).with_name('weight_effect_model.py').resolve()
    calibration = json.loads(subprocess.check_output([sys.executable, str(instrument), '--calibrate'], text=True, timeout=30))
    spec = importlib.util.spec_from_file_location('science', Path(__file__).resolve().parents[2] / 'python/metalrobo/science_notebook.py')
    science = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(science)
    qualification_directory = directory / 'qualification'
    qualification_directory.mkdir()
    def run(argv, name):
        output = qualification_directory / name
        execution = science.execute_command({'argv': argv, 'env': {}, 'timeout_seconds': 300}, output, science.base_environment())
        science.seal(output / 'receipt.json', execution)
        if execution['failure']:
            raise ValueError('instrument qualification failed; partial output retained: ' + execution['failure'])
        return science.read(output / 'output/stdout.json')
    native = run([str(binary)], 'native-qualification')
    if native.get('schema') != 'numi.neuron-culture.qualification.v1' or native.get('wet_lab') is not False:
        raise ValueError('unexpected native qualification scope')
    for key in ('cpu_metal_parity', 'transactional_reject', 'bitwise_cpu_replay'):
        if native.get(key) is not True:
            raise ValueError('native qualification failed: ' + key)
        calibration['checks'].append({'id': key, 'passed': True})
    run([sys.executable, str(instrument), '--binary', str(binary), '--seed', str(args.calibration_seed), '--ablation', 'stdp-off'], 'frozen-control')
    calibration['checks'].append({'id': 'native_observable_and_frozen_weight_control', 'passed': True})
    instrument_artifacts = [str(Path(__file__).resolve()), str(instrument), str(binary), str(shader)]
    calibration['bindings'] = {path: sha(path) for path in instrument_artifacts}
    calibration['observed_units'] = [{'seed': args.calibration_seed}]
    calibration['evidence'] = {str(path): sha(path) for path in qualification_directory.rglob('*') if path.is_file()}
    calibration['scope'] = 'Known-value JSON reader, native CPU/Metal parity and frozen-weight control. No biological calibration.'
    save(directory / 'calibration.json', calibration)
    save(directory / 'model.json', model)
    trials = []
    for index, seed in enumerate(args.seeds):
        for arm in (('control', 'treatment') if index % 2 == 0 else ('treatment', 'control')):
            trials.append({'id': f'seed-{seed}-{arm}', 'pair': str(seed), 'arm': arm, 'unit': {'seed': seed},
                           'argv': [sys.executable, str(instrument), '--binary', str(binary), '--seed', str(seed),
                                    '--ablation', 'stdp-off' if arm == 'control' else 'none'],
                           'env': {}, 'timeout_seconds': 300})
    artifacts = [str(Path(__file__).resolve()), str(instrument), str(model_program), str(directory / 'calibration.json'), str(directory / 'model.json'),
                 str(receipt_path), *receipt['inputs'], *receipt['outputs'], *calibration['evidence']]
    plan = {'schema': 'numi.science.plan.v2', 'purpose': args.purpose,
            'question': 'What paired change in mean plastic weight does enabling STDP cause over the native quick synthetic protocol?',
            'hypothesis': model['statement'], 'owner': 'Numi Lab native neuron culture', 'repository': str(runtime),
            'backend': 'Apple Metal synthetic LIF/STDP, exact focused owner build', 'evidence_level': 'simulation',
            'model': model, 'model_file': str(directory / 'model.json'),
            'predictor': {'argv': [sys.executable, str(model_program), 'predict', '--model', str(directory / 'model.json')],
                          'env': {}, 'timeout_seconds': 30},
            'instrument': {'description': 'Native mean plastic weight plus calibrated reader and frozen-weight control',
                           'calibration': str(directory / 'calibration.json'), 'artifacts': instrument_artifacts},
            'artifacts': list(dict.fromkeys(artifacts)),
            'design': {'intervention': 'STDP enabled versus stdp-off; all other native protocol options identical',
                       'controls': 'Paired seed, topology, initial state, sensory mapping 0, current scales, 12 windows; off arm preserves 0.05',
                       'experimental_unit': 'One independently seeded synthetic culture; ticks are not replicates',
                       'unit_paths': {'seed': ['network_seed']},
                       'allocation': 'Declared seed order; alternating first arm; serial execution'},
            'observable': {'name': 'mean_plastic_weight', 'unit': 'authored weight units', 'path': ['mean_plastic_weight']},
            'prediction': predicted,
            'validity': [{'path': ['schema'], 'equals': 'numi.science.neuron-weight.v1'}, {'path': ['simulation_only'], 'equals': True}],
            'paired_equal': [[key] for key in ('network_seed', 'culture_fingerprint', 'starting_state_fingerprint',
                                               'synaptic_current_scale', 'stimulation_current', 'windows')],
            'trials': trials,
            'limitations': 'Small deterministic synthetic protocol from unconditioned weights. No biological calibration, learning advantage, full Potter reproduction, population confidence interval, or hardware evidence. Instrument qualification and test observations are separate.'}
    save(directory / 'plan.json', plan)
    print(directory / 'plan.json')


if __name__ == '__main__':
    main()
