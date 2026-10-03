#!/usr/bin/env python3
"""Executable descriptive model for the bounded synthetic STDP experiment."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def prediction(model):
    if model['schema'] != 'numi.science.weight-effect-model.v1':
        raise ValueError('unsupported model')
    centre, width = (model['parameters'][key] for key in ('centre', 'half_width'))
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in (centre, width)) or width < 0:
        raise ValueError('invalid model parameters')
    return {'estimand': 'paired_difference_mean', 'minimum': centre - width, 'maximum': centre + width}


def initial_model():
    return {'schema': 'numi.science.weight-effect-model.v1', 'version': 'stdp-effect-v1',
            'statement': 'Net potentiation over the quick protocol produces a mean paired increase from 0.0001 to 0.01.',
            'parameters': {'centre': 0.00505, 'half_width': 0.00495}, 'training_units': [],
            'scope': 'Descriptive synthetic protocol; no statistical coverage or biological claim.'}


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('predict')
    command.add_argument('--model', type=Path, required=True)
    for name in ('fit', 'retain'):
        command = sub.add_parser(name)
        command.add_argument('--study', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--revision', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'predict':
        model = json.loads(args.model.read_text())
        print(json.dumps({'schema': 'numi.science.prediction.v1',
                          'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
                          'prediction': prediction(model)}, allow_nan=False, sort_keys=True))
        return
    path = Path(__file__).resolve().parents[2] / 'python/metalrobo/science_notebook.py'
    spec = importlib.util.spec_from_file_location('science', path)
    science = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(science)
    study = args.study.resolve()
    with science.locked(study):
        registration, _ = science.verify(study)
        analysis = science.unseal(study / 'analysis.json')
        plan, data = registration['payload']['plan'], analysis['payload']
        if data['verdict'] == 'inconclusive' or len(data['paired_differences']) < 3:
            raise ValueError('requires at least three valid paired observations; never fit failed data')
        model = dict(plan['model'])
        model['version'] += '.' + args.command
        units = {science.digest(unit): unit for unit in [*model['training_units'], *(trial['unit'] for trial in plan['trials'])]}
        model['training_units'] = list(units.values())
        if args.command == 'fit':
            model['statement'] = 'Seed-dependent small weight effect; mean plus or minus three discovery sample SDs is a heuristic predictive range.'
            model['parameters'] = {'centre': data['mean_difference'], 'half_width': 3 * data['observed_spread']['sample_sd']}
            model['fit'] = {'analysis_sha256': analysis['sha256'], 'rule': 'paired_mean_plus_or_minus_3_sample_sd'}
            decision = 'revise'
        else:
            if data['verdict'] != 'supported':
                raise ValueError('cannot retain contradicted parameters')
            model['validation'] = {'analysis_sha256': analysis['sha256'], 'scope': 'These declared seeds only'}
            decision = 'retain'
        prediction(model)
        revision = {'decision': decision, 'model': model, 'model_file': str(args.output.resolve()),
                    'evidence': [analysis['sha256']], 'reason': 'Fit descriptive parameters to valid discovery pairs.' if decision == 'revise'
                    else 'Independent declared conditions support the frozen prediction at this scope.',
                    'next_test': 'Preregister unused conditions and a discriminating mechanistic alternative before further observations.',
                    'limitations': model['scope']}
        save(args.output, model)
        save(args.revision, revision)
    print(args.output)


if __name__ == '__main__':
    main()
