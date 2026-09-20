"""Native input admission only; this tool never steps physical state."""
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile


def variants(original, version):
    cases = {}

    def duplicate(name, key, escaped=None):
        needle = json.dumps(key) + ':'
        replacement = '"' + (escaped or key) + '":"duplicate",' + needle
        assert needle in original, name + ': fixture key missing'
        cases[name] = original.replace(needle, replacement, 1)

    def variant(name, mutate):
        obj = json.loads(original)
        mutate(obj)
        cases[name] = json.dumps(obj)

    duplicate('duplicate_root', 'schema')
    duplicate('duplicate_escaped_root', 'schema', r'sch\u0065ma')
    if version == 1:
        # Preserve the original twelve v1 rejection cases.
        duplicate('duplicate_nested', 'volume_m3')
        variant('unsupported_law', lambda x: x.update(law='pump_v99'))
        variant('unknown_field', lambda x: x.update(replenish_blood=True))
        variant('boolean_scale', lambda x: x['species'][0].update(amount_scale_mol=True))
        variant('duplicate_volume_owner', lambda x: x['tissue_reservoirs'][0].update(physical_volume_owner_id=x['compartments'][0]['physical_volume_owner_id']))
        variant('missing_endpoint', lambda x: x['connections'][0].update(to=900))
        variant('invalid_sha', lambda x: x.update(source_graph_sha256='x' * 64))
        variant('unsupported_qualification', lambda x: x.update(qualification='validated_human'))
        variant('negative_amount', lambda x: x['compartments'][0].update(initial_species_mol=[-1]))
        variant('unknown_species', lambda x: x['exchanges'][0].update(species=900))
        assert len(cases) == 12
        return cases

    def cardiac(x):
        return next(row for row in x['compartments'] if row['pressure_law'] != 'linear_compliance')

    def linear(x):
        return next(row for row in x['compartments'] if row['pressure_law'] == 'linear_compliance')

    def orifice(x):
        return next(row for row in x['connections'] if row['flow_law'] == 'one_way_orifice')

    def rlc(x):
        return next(row for row in x['connections'] if row['flow_law'] == 'resistance_inertance')

    def species(x):
        # Keep global IDs, amount shapes and hydraulic endpoints otherwise valid
        # so this exercises the missing absolute-volume admission boundary.
        x['species'] = [{'id': 'tracer', 'stable_identifier': 1, 'description': 'Admission fixture only',
                         'amount_scale_mol': 1e-6, 'amount_residual_tolerance': x['residual_tolerance']}]
        for row in x['compartments']:
            row['stable_identifier'] += 1
            row['initial_species_mol'] = [0.0]
        for row in x['connections']:
            row['stable_identifier'] += 1
            row['from'] += 1
            row['to'] += 1

    def tissue(x):
        identifier = 1 + max(row['stable_identifier'] for table in ('species', 'compartments', 'connections') for row in x[table])
        x['tissue_reservoirs'] = [{'id': 'tissue', 'stable_identifier': identifier,
                                  'anatomical_region_id': 'fixture:tissue', 'physical_volume_owner_id': 'fixture:tissue',
                                  'volume_m3': 1e-6, 'initial_species_mol': [0.0] * len(x['species'])}]

    def coupled_transport(x):
        species(x)
        tissue(x)
        target = x['tissue_reservoirs'][0]['stable_identifier']
        x['exchanges'] = [{'id': 'exchange', 'stable_identifier': target + 1,
                          'compartment': linear(x)['stable_identifier'], 'tissue_reservoir': target,
                          'species': 1, 'clearance_m3_per_s': 1e-8, 'partition_coefficient': 1.0}]

    duplicate('duplicate_nested', 'period_seconds')
    duplicate('duplicate_escaped_nested', 'pressure_law', r'pressure_\u006caw')
    variant('wrong_storage_enum', lambda x: linear(x).update(storage_kind='invented_blood_volume'))
    variant('wrong_pressure_enum', lambda x: cardiac(x).update(pressure_law='unreviewed_drive'))
    variant('wrong_flow_enum', lambda x: orifice(x).update(flow_law='unreviewed_valve'))
    variant('unsupported_law', lambda x: x.update(law='closed_linear_compliance_transport_v1'))
    variant('unsupported_qualification', lambda x: x.update(qualification='validated_human'))
    variant('unrepresentable_period', lambda x: cardiac(x).update(period_seconds=1e-20))
    variant('cardiac_nonzero_compliance', lambda x: cardiac(x).update(compliance_m3_per_pa=1e-9))
    variant('linear_unused_waveform', lambda x: linear(x).update(source_pi=3.14159))
    variant('negative_orifice_flow', lambda x: orifice(x).update(initial_flow_m3_per_s=-1e-6))
    variant('negative_orifice_coefficient', lambda x: orifice(x).update(orifice_coefficient_m3_per_s_sqrt_pa=-1e-5))
    variant('zero_orifice_coefficient', lambda x: orifice(x).update(orifice_coefficient_m3_per_s_sqrt_pa=0.0))
    variant('rlc_nonzero_orifice_coefficient', lambda x: rlc(x).update(orifice_coefficient_m3_per_s_sqrt_pa=1e-5))
    variant('storage_with_species', species)
    variant('storage_with_tissue', tissue)
    variant('storage_with_species_tissue_exchange', coupled_transport)
    variant('unknown_root_field', lambda x: x.update(absolute_volume_offset_m3=0.005))
    variant('unknown_nested_field', lambda x: orifice(x).update(leakage_m3_per_s=1e-10))
    variant('missing_waveform_field', lambda x: cardiac(x).pop('activation_start'))
    variant('missing_orifice_field', lambda x: orifice(x).pop('orifice_coefficient_m3_per_s_sqrt_pa'))
    variant('boolean_period', lambda x: cardiac(x).update(period_seconds=True))
    variant('nonfinite_elastance', lambda x: cardiac(x).update(elastance_max_pa_per_m3=float('nan')))
    return cases


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit('usage: physiology_input_check.py PHYSIOLOGYC V1_JSON [V2_JSON]')
    cli = sys.argv[1]
    root = pathlib.Path(tempfile.mkdtemp(prefix='numi-physiology-input-'))
    counts = {}
    try:
        for version, path in enumerate(sys.argv[2:], 1):
            original = pathlib.Path(path).read_text()
            base = json.loads(original)
            assert base['schema'] == 'HumanPack.physiology-native.v' + str(version), 'wrong positive fixture version'
            cases = variants(original, version)
            counts[version] = len(cases)
            for name, data in [('valid', original), *cases.items()]:
                label = 'v' + str(version) + '_' + name
                source = root / (label + '.json')
                output = root / (label + '.nmatterpack')
                source.write_text(data)

                def execute(suffix):
                    result = subprocess.run([cli, str(source), str(output), '--timestep', '0.01', '--environments', '2'],
                                            capture_output=True, text=True, timeout=30)
                    (root / (label + suffix + '.stdout.txt')).write_text(result.stdout)
                    (root / (label + suffix + '.stderr.txt')).write_text(result.stderr)
                    (root / (label + suffix + '.exit.json')).write_text(json.dumps({'returncode': result.returncode}))
                    return result

                result = execute('')
                if name == 'valid':
                    assert result.returncode == 0 and output.is_file(), label + ': ' + result.stdout + result.stderr
                    print(result.stdout.strip(), flush=True)
                else:
                    assert result.returncode != 0 and not output.exists(), label + ': invalid input accepted or output created\n' + result.stdout + result.stderr
                    sentinel = b'preexisting-output-must-survive-rejection\n'
                    output.write_bytes(sentinel)
                    repeated = execute('.preexisting')
                    assert repeated.returncode != 0 and output.read_bytes() == sentinel, label + ': rejection replaced existing output\n' + repeated.stdout + repeated.stderr
                    print('native_input_rejected=' + label + ' output_atomic=pass', flush=True)
    except BaseException:
        print('native_input_failure_artifacts=' + str(root), flush=True)
        raise
    else:
        shutil.rmtree(root)
    summary = 'native_input_admission=pass positive_cases=' + str(len(counts)) + ' negative_cases=' + str(sum(counts.values()))
    if len(counts) == 2:
        summary += ' v1_negative_cases=' + str(counts[1]) + ' v2_negative_cases=' + str(counts[2]) + ' atomic_rejection=pass'
    print(summary)


if __name__ == '__main__':
    main()
