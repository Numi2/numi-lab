"""Strict native v3 input admission; never advances a physical runtime.

Usage: cvsim_input_check.py PHYSIOLOGYC V3_JSON [V2_JSON]
A supplied v2 fixture is tested positively before schema-isolation mutations.
"""
import copy
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile


REGIONAL_FIELDS = ('maximum_volume_displacement_m3', 'phase_delay',
                   'period_numerator_seconds', 'period_denominator')
FLOOR = 'downstream_pressure_floor_pa'


def row(obj, table, key, value):
    return next(item for item in obj[table] if item[key] == value)


def compartment(obj, law):
    return row(obj, 'compartments', 'pressure_law', law)


def connection(obj, law):
    return row(obj, 'connections', 'flow_law', law)


def variants(original):
    base = json.loads(original)
    assert base['schema'] == 'HumanPack.physiology-native.v3', 'wrong v3 fixture'
    # Require the laws under test to be present, so mutations cannot silently
    # disappear when the source fixture changes.
    for law in ('linear_compliance', 'atan_compliance', 'cosine_pulse_elastance'):
        compartment(base, law)
    for law in ('resistance_inertance', 'one_way_resistance', 'starling_resistance'):
        connection(base, law)
    cases = {}

    def variant(name, mutate):
        changed = copy.deepcopy(base)
        mutate(changed)
        cases[name] = json.dumps(changed)

    def comp(name, law, **fields):
        variant(name, lambda x: compartment(x, law).update(fields))

    def flow(name, law, **fields):
        variant(name, lambda x: connection(x, law).update(fields))

    for key in ('schema', 'period_numerator_seconds', 'period_denominator'):
        # Serialize once with known spacing, retain duplicate keys in raw JSON.
        raw = json.dumps(base)
        needle = json.dumps(key) + ':'
        assert needle in raw
        cases['duplicate_' + key] = raw.replace(needle, '"' + key + '":"duplicate",' + needle, 1)
        escaped = key[:-1] + '\\u%04x' % ord(key[-1])
        cases['escaped_duplicate_' + key] = raw.replace(needle, '"' + escaped + '":"duplicate",' + needle, 1)

    for key in ('period_numerator_seconds', 'period_denominator'):
        invalid = {'numeric': 7, 'boolean': True, 'null': None, 'empty': '',
                   'leading_zero': '07', 'positive_sign': '+7', 'negative': '-7',
                   'whitespace': ' 7', 'trailing_whitespace': '7 ', 'decimal': '7.0',
                   'exponent': '7e0', 'unicode_digits': '\u0667',
                   'u64_overflow': '18446744073709551616', 'overlong': '1' * 21}
        for label, value in invalid.items():
            comp(key + '_' + label, 'cosine_pulse_elastance', **{key: value})

    for value in ('HumanPack.physiology-native.v4', 'HumanPack.physiology-native.v0'):
        variant('schema_' + value.rsplit('.', 1)[-1], lambda x, v=value: x.update(schema=v))
    for value in ('source_model_reproduction', 'validated_human', 'uncalibrated'):
        variant('qualification_' + value, lambda x, v=value: x.update(qualification=v))
    variant('unsupported_root_law', lambda x: x.update(law='closed_periodic_elastance_orifice_v2'))
    variant('extra_root_field', lambda x: x.update(accepted_time_seconds=1))
    variant('missing_root_field', lambda x: x.pop('source_graph_sha256'))
    comp('extra_compartment_field', 'atan_compliance', domain_clamp=True)
    flow('extra_connection_field', 'starling_resistance', retain_previous_flow=True)
    for field in REGIONAL_FIELDS:
        variant('missing_' + field, lambda x, f=field: compartment(x, 'cosine_pulse_elastance').pop(f))
    variant('missing_floor', lambda x: connection(x, 'starling_resistance').pop(FLOOR))
    comp('unknown_pressure_law', 'atan_compliance', pressure_law='clamped_atan')
    comp('unknown_storage_kind', 'atan_compliance', storage_kind='display_volume')
    flow('unknown_flow_law', 'starling_resistance', flow_law='previous_flow_starling')

    comp('rational_non_coprime', 'cosine_pulse_elastance', period_numerator_seconds='6', period_denominator='14')
    comp('rational_missing_numerator', 'cosine_pulse_elastance', period_numerator_seconds='0')
    comp('rational_missing_denominator', 'cosine_pulse_elastance', period_denominator='0')
    comp('rational_both_zero', 'cosine_pulse_elastance', period_numerator_seconds='0', period_denominator='0')
    comp('rational_numerator_tick_overflow', 'cosine_pulse_elastance', period_numerator_seconds='18446744073709551615', period_denominator='1')
    comp('two_period_authorities', 'cosine_pulse_elastance', period_seconds=1.0)
    comp('linear_unused_period', 'linear_compliance', period_numerator_seconds='6', period_denominator='7')
    comp('linear_unused_domain', 'linear_compliance', maximum_volume_displacement_m3=.001)
    comp('linear_unused_delay', 'linear_compliance', phase_delay=.1)
    comp('pulse_unused_domain', 'cosine_pulse_elastance', maximum_volume_displacement_m3=.001)
    comp('pulse_negative_delay', 'cosine_pulse_elastance', phase_delay=-.1)
    comp('pulse_wrapped_delay', 'cosine_pulse_elastance', phase_delay=1)
    comp('pulse_bad_intervals', 'cosine_pulse_elastance', activation_start=.7, activation_end=.3)
    comp('pulse_nonzero_compliance', 'cosine_pulse_elastance', compliance_m3_per_pa=1e-9)
    comp('pulse_nonpositive_elastance', 'cosine_pulse_elastance', elastance_min_pa_per_m3=0)
    comp('pulse_signed_storage', 'cosine_pulse_elastance', storage_kind='storage_displacement')
    comp('atan_nonpositive_domain', 'atan_compliance', maximum_volume_displacement_m3=0)
    comp('atan_nonpositive_compliance', 'atan_compliance', compliance_m3_per_pa=0)
    comp('atan_wrong_pi', 'atan_compliance', source_pi=3.14159)
    comp('atan_signed_storage', 'atan_compliance', storage_kind='storage_displacement')
    comp('atan_unused_period', 'atan_compliance', period_numerator_seconds='6', period_denominator='7')
    comp('atan_unused_elastance', 'atan_compliance', elastance_min_pa_per_m3=1)
    comp('atan_nonfinite_volume', 'atan_compliance', initial_volume_m3=float('nan'))
    for label, multiple in (('lower_endpoint', -1), ('upper_endpoint', 1), ('lower_exterior', -2), ('upper_exterior', 2)):
        def domain(x, m=multiple):
            v = compartment(x, 'atan_compliance')
            # A positive absolute volume on both sides prevents the generic
            # volume-positivity check from masking the atan boundary test.
            v.update(reference_volume_m3=2.0, maximum_volume_displacement_m3=.5, initial_volume_m3=2.0 + m * .5)
        variant('atan_' + label, domain)
    comp('atan_normalization_hits_endpoint', 'atan_compliance', reference_volume_m3=2,
         maximum_volume_displacement_m3=1, initial_volume_m3=2.999999761581421,
         volume_scale_m3=43561.0234375)
    for law in ('one_way_resistance', 'starling_resistance'):
        flow(law + '_reverse_initial', law, initial_flow_m3_per_s=-1e-6)
        flow(law + '_zero_resistance', law, resistance_pa_s_per_m3=0)
        flow(law + '_inertance', law, inertance_pa_s2_per_m3=1)
        flow(law + '_orifice_coefficient', law, orifice_coefficient_m3_per_s_sqrt_pa=1e-5)
    for law in ('resistance_inertance', 'one_way_resistance'):
        flow(law + '_unused_floor', law, downstream_pressure_floor_pa=1)
    flow('starling_nonfinite_floor', 'starling_resistance', downstream_pressure_floor_pa=float('inf'))
    flow('starling_boolean_floor', 'starling_resistance', downstream_pressure_floor_pa=True)
    return base, cases


def v2_variants(original):
    base = json.loads(original)
    assert base['schema'] == 'HumanPack.physiology-native.v2', 'wrong v2 positive control'
    cases = {}
    for field in REGIONAL_FIELDS:
        changed = copy.deepcopy(base)
        changed['compartments'][0][field] = '0' if field.startswith('period_') else 0.0
        cases['v2_cannot_smuggle_' + field] = json.dumps(changed)
    changed = copy.deepcopy(base)
    changed['connections'][0][FLOOR] = 0.0
    cases['v2_cannot_smuggle_' + FLOOR] = json.dumps(changed)
    for law in ('atan_compliance', 'cosine_pulse_elastance'):
        changed = copy.deepcopy(base)
        changed['compartments'][0]['pressure_law'] = law
        cases['v2_cannot_select_' + law] = json.dumps(changed)
    for law in ('one_way_resistance', 'starling_resistance'):
        changed = copy.deepcopy(base)
        changed['connections'][0]['flow_law'] = law
        cases['v2_cannot_select_' + law] = json.dumps(changed)
    changed = copy.deepcopy(base)
    changed['qualification'] = 'source_model_variant'
    cases['v2_cannot_smuggle_qualification'] = json.dumps(changed)
    return cases


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__.strip())
    cli = sys.argv[1]
    original = pathlib.Path(sys.argv[2]).read_text()
    base, negatives = variants(original)
    positives = {'valid_v3': original}
    # Prove the parser retains all 64 bits with a compiler-admissible period,
    # rather than accepting only small values or coercing JSON numbers.
    large = copy.deepcopy(base)
    compartment(large, 'cosine_pulse_elastance').update(period_numerator_seconds='1', period_denominator='18446744073709551615')
    positives['full_width_denominator'] = json.dumps(large)
    legacy = copy.deepcopy(base)
    compartment(legacy, 'cosine_pulse_elastance').update(period_seconds=1, period_numerator_seconds='0', period_denominator='0')
    positives['v3_legacy_binary_period'] = json.dumps(legacy)
    if len(sys.argv) == 4:
        v2 = pathlib.Path(sys.argv[3]).read_text()
        positives['valid_v2_control'] = v2
        negatives.update(v2_variants(v2))
    root = pathlib.Path(tempfile.mkdtemp(prefix='numi-cvsim-input-'))
    try:
        for name, data in {**positives, **negatives}.items():
            source, output = root / (name + '.json'), root / (name + '.nmatterpack')
            source.write_text(data)

            def execute(suffix):
                result = subprocess.run([cli, str(source), str(output), '--timestep', '0.002', '--environments', '2'],
                                        capture_output=True, text=True, timeout=30)
                (root / (name + suffix + '.stdout.txt')).write_text(result.stdout)
                (root / (name + suffix + '.stderr.txt')).write_text(result.stderr)
                (root / (name + suffix + '.exit.json')).write_text(json.dumps({'returncode': result.returncode}))
                return result

            result = execute('')
            if name in positives:
                assert result.returncode == 0 and output.is_file() and output.stat().st_size, name + ': ' + result.stdout + result.stderr
                print('cvsim_input_positive=' + name, flush=True)
            else:
                assert result.returncode != 0 and not output.exists(), name + ': accepted or output created\n' + result.stdout + result.stderr
                sentinel = b'preexisting-output-must-survive-v3-rejection\x00\xff\n'
                output.write_bytes(sentinel)
                repeated = execute('.preexisting')
                assert repeated.returncode != 0 and output.read_bytes() == sentinel, name + ': rejection replaced output\n' + repeated.stdout + repeated.stderr
                print('cvsim_input_rejected=' + name + ' output_atomic=pass', flush=True)
    except BaseException:
        print('cvsim_input_failure_artifacts=' + str(root), flush=True)
        raise
    else:
        shutil.rmtree(root)
    print('cvsim_input_admission=pass positive_cases=' + str(len(positives)) + ' negative_cases=' + str(len(negatives)) + ' atomic_rejection=pass physical_execution=none')


if __name__ == '__main__':
    main()
