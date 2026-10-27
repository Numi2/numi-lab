from pathlib import Path

import run_matched_fit583_v2 as m


class FakeContinue:
    FIT_TEMPLATE = Path('/owner/template.py')

    def __init__(self):
        self.run_args = None
        self.argv_kwargs = None

    def fit_argv_from_template(self, template, **kwargs):
        self.argv_kwargs = (template, kwargs)
        return ['python', 'template-entry.py', '--trust-mm', '0',
                '--max-edge-gradient', '0', '--source-cap-mm', '0',
                '--time-limit-s', '0']

    @staticmethod
    def replace_cli_value(argv, flag, value):
        index = argv.index(flag)
        argv[index + 1] = value

    @staticmethod
    def write_json(path, value):
        Path(path).write_text('{}\n')

    def run_stage(self, argv, stage_dir, *, slot, source_paths, timeout):
        self.run_args = {
            'argv': list(argv), 'stage_dir': stage_dir, 'slot': slot,
            'source_paths': tuple(source_paths), 'timeout': timeout,
        }


def test_fresh_fit_uses_path_pins_and_emits_expected_argv_without_solver(tmp_path):
    fake = FakeContinue()
    report = tmp_path / 'audit159.json'
    report.write_text('{}\n')
    base = tmp_path / 'candidate.npz'
    context = {
        'guard_npz': str(tmp_path / 'guards.npz'),
        'guard_report': str(tmp_path / 'guards.json'),
        'seed_pairs_path': str(tmp_path / 'seeds.npy'),
    }
    output = tmp_path / 'round'
    output.mkdir()
    point_npz = tmp_path / 'targets.npz'
    point_report = tmp_path / 'targets.json'
    matched = tmp_path / 'matched.json'
    state = {'surface_report': str(report)}

    failed = m._run_fresh_fit(
        fake, state, base, tmp_path / 'prep', context, point_npz,
        point_report, matched, output, 1.0, 0.1)

    assert failed is False
    captured = fake.run_args
    assert captured is not None
    assert captured['slot'] is True
    assert captured['timeout'] == 360
    assert all(isinstance(path, Path) for path in captured['source_paths'])
    assert report in captured['source_paths']
    assert str(m.FIT564) == captured['argv'][1]
    assert captured['argv'][captured['argv'].index('--trust-mm') + 1] == '1.0'
    assert captured['argv'][captured['argv'].index('--max-edge-gradient') + 1] == '0.1'
    assert '--allow-partial-timeout-zero-fallback-564' in captured['argv']
    assert output.joinpath('fit-argv.json').is_file()
