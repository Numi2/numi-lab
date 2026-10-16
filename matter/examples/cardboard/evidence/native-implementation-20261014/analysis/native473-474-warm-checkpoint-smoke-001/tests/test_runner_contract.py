from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


PACKAGE = Path(__file__).resolve().parents[1]
SPEC_PATH = PACKAGE / 'run-spec.json'
spec = importlib.util.spec_from_file_location('warm_smoke_prepare', PACKAGE / 'prepare.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
RUN_SPEC = json.loads(SPEC_PATH.read_text())


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def protocol_manifest(run_spec):
    protocol = run_spec['protocol']
    phase = protocol['phase_controls']
    material = protocol['materials']
    return {
        'schema': 'numi.cardboard.explicit-strip.v1',
        'preset': 'literature2009',
        'solver': {
            'steps': phase['loading'], 'hold_steps': phase['hold'],
            'unload_steps': phase['unloading'], 'relax_steps': phase['relaxation'],
            'release_steps': phase['release'],
            'post_release_relaxation_steps': phase['post_release_relaxation'],
            'dt_s': 0.00005, 'newton_iteration_budget': 64,
            'fgmres_restart': 16, 'fgmres_iteration_budget': 1024,
            'line_search_steps': 24,
            'fgmres_preconditioner': 'regional_galerkin_graph_schwarz_conditioning_only',
            'fgmres_algorithmic_tangent_cache': True,
            'deformable_self_contact': False,
            'deformation_gradient': 'persistent_reference_displacement',
            'paper_compensated_arithmetic': True,
            'bend_angle_deg': 0.375,
        },
        'mesh': {'nx_per_pitch': 8, 'ny': 2, 'thickness_slices': 1},
        'materials': {
            'liner_path': material['liner']['path'],
            'liner_sha256': material['liner']['sha256'],
            'medium_path': material['medium']['path'],
            'medium_sha256': material['medium']['sha256'],
        },
    }


class RunnerContractTests(unittest.TestCase):
    def test_intended_command_contract_and_warm_flag_are_required(self):
        self.assertTrue(prepare.verify_command_contract(RUN_SPEC)['same_physical_argv'])
        altered = copy.deepcopy(RUN_SPEC)
        altered['environment_overrides']['NM_FEM_BASE_CGLS_WARM_START_REFINE'] = '0'
        with self.assertRaisesRegex(ValueError, 'warm-start refinement'):
            prepare.verify_command_contract(altered)

    def test_resume_cannot_create_a_second_checkpoint_and_must_match_physics(self):
        altered = copy.deepcopy(RUN_SPEC)
        altered['cases']['474']['native_argv'] += ['--coupon-checkpoint-after-control', '0']
        with self.assertRaisesRegex(ValueError, 'cannot request another checkpoint'):
            prepare.verify_command_contract(altered)
        altered = copy.deepcopy(RUN_SPEC)
        idx = altered['cases']['474']['native_argv'].index('--fgmres-iterations') + 1
        altered['cases']['474']['native_argv'][idx] = '512'
        with self.assertRaisesRegex(ValueError, 'arguments differ'):
            prepare.verify_command_contract(altered)

    def test_external_input_collisions_are_detected_before_hash_mapping(self):
        with tempfile.TemporaryDirectory(dir=Path.home()) as td:
            root = Path(td)
            a = root / 'a' / 'same.bin'
            b = root / 'b' / 'same.bin'
            a.parent.mkdir(); b.parent.mkdir()
            a.write_bytes(b'a'); b.write_bytes(b'b')
            with self.assertRaisesRegex(ValueError, 'colliding basenames'):
                prepare._expected_external_input_hashes([a, b], '0' * 64)
            unique = root / 'unique.bin'
            unique.write_bytes(b'c')
            result = prepare._expected_external_input_hashes([a, unique], '1' * 64)
            self.assertEqual(set(result), {'external/same.bin', 'external/unique.bin', 'external/source-build.json'})
            self.assertEqual(result['external/same.bin'], digest(b'a'))

    def test_prefix_reader_checks_manifest_members_and_exact_observation_prefix(self):
        with tempfile.TemporaryDirectory(dir=Path.home()) as td:
            output = Path(td) / 'output'
            output.mkdir()
            observations = b'control,env\n0,0\n'
            manifest = b'protocol\n'
            (output / 'observations.csv').write_bytes(observations)
            (output / 'manifest.json').write_bytes(manifest)
            prefix = Path(td) / 'prefix'
            prefix.write_text(
                'schema=numi.cardboard.accepted-coupon-prefix.v1\n'
                f'source_run_directory={output.resolve()}\n'
                'accepted_control_index=0\n'
                f'observations_prefix_bytes={len(observations)}\n'
                f'observations_prefix_sha256={digest(observations)}\n'
                f'file=manifest.json|{len(manifest)}|{digest(manifest)}|0\n'
                f'file=observations.csv|{len(observations)}|{digest(observations)}|1\n')
            result = prepare.parse_prefix(prefix, 0, output)
            self.assertEqual(result['entries'], 2)
            self.assertEqual(result['observations_prefix_bytes'], len(observations))
            prefix.write_text(prefix.read_text().replace(digest(observations), '0' * 64))
            with self.assertRaisesRegex(ValueError, 'prefix header differs|binding mismatch'):
                prepare.parse_prefix(prefix, 0, output)

    def test_prefix_reader_rejects_parent_escape_even_with_valid_hash(self):
        with tempfile.TemporaryDirectory(dir=Path.home()) as td:
            root = Path(td)
            output = root / 'output'; output.mkdir()
            observations = b'x'
            (output / 'observations.csv').write_bytes(observations)
            prefix = root / 'prefix'
            prefix.write_text(
                'schema=numi.cardboard.accepted-coupon-prefix.v1\n'
                f'source_run_directory={output.resolve()}\naccepted_control_index=0\n'
                f'observations_prefix_bytes=1\nobservations_prefix_sha256={digest(observations)}\n'
                f'file=../outside|1|{digest(b"x")}|0\n'
                f'file=observations.csv|1|{digest(observations)}|1\n')
            with self.assertRaisesRegex(ValueError, 'unsafe/duplicate'):
                prepare.parse_prefix(prefix, 0, output)

    def test_bounded_file_reader_rejects_symlink_and_hardlink_aliases(self):
        with tempfile.TemporaryDirectory(dir=Path.home()) as td:
            root = Path(td)
            real = root / 'real'; real.write_bytes(b'payload')
            alias = root / 'alias'
            alias.symlink_to(real)
            with self.assertRaisesRegex(ValueError, 'regular file'):
                prepare.regular_file(alias, 1024)
            hard = root / 'hard'
            hard.hardlink_to(real)
            with self.assertRaisesRegex(ValueError, 'unlinked regular'):
                prepare.regular_file(hard, 1024)

    def test_protocol_manifest_rejects_shortened_or_rebound_run(self):
        good = protocol_manifest(RUN_SPEC)
        self.assertTrue(all(prepare.verify_protocol_manifest_from_mapping(good, RUN_SPEC).values()))
        short = copy.deepcopy(good)
        short['solver']['release_steps'] = 5
        with self.assertRaisesRegex(ValueError, 'release_steps|total_controls'):
            prepare.verify_protocol_manifest_from_mapping(short, RUN_SPEC)
        rebound = copy.deepcopy(good)
        rebound['materials']['medium_sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'medium_sha256'):
            prepare.verify_protocol_manifest_from_mapping(rebound, RUN_SPEC)

    def test_resume_verifier_requires_all_59_controls_and_exact_restored_identity(self):
        with tempfile.TemporaryDirectory(dir=Path.home()) as td:
            root = Path(td)
            case = root / '474'; output = case / 'output'
            output.mkdir(parents=True)
            protocol = protocol_manifest(RUN_SPEC)
            (output / 'manifest.json').write_text(json.dumps(protocol))
            source_meta = root / 'control0.resume'
            checkpoint_hash = 'a' * 64
            manifest_hash = digest((output / 'manifest.json').read_bytes())
            source_meta.write_text(
                'source_run_id=run-identity\n'
                f'checkpoint_sha256={checkpoint_hash}\n'
                f'protocol_manifest_sha256={manifest_hash}\n')
            spec_copy = copy.deepcopy(RUN_SPEC)
            spec_copy['cases']['474']['case'] = str(case)
            spec_copy['cases']['473']['expected_metadata'] = str(source_meta)
            (case / 'receipt.json').write_text(json.dumps({'exit_code': 0}))

            def write_output(accepted=59, restored=True, full=False):
                (output / 'result.json').write_text(json.dumps({
                    'schema': 'numi.cardboard.probe-result.v1',
                    'status': 'resumed_diagnostic_segment_reached_schedule_end',
                    'accepted_steps': accepted, 'physical_validation': False}))
                (output / 'coupon-resume-segment.json').write_text(json.dumps({
                    'schema': 'numi.cardboard.coupon-resume-segment.v1',
                    'evidence_class': 'diagnostic_resumed_segment',
                    'status': 'resumed_diagnostic_segment_reached_schedule_end',
                    'source_run_id': 'run-identity',
                    'source_prefix_accepted_control_count': 1,
                    'segment_start_control_index': 1,
                    'segment_accepted_controls': accepted,
                    'checkpoint_sha256': checkpoint_hash,
                    'protocol_manifest_sha256': manifest_hash,
                    'accepted_state_restored_from_native_snapshot': restored,
                    'accepted_prefix_reconstructed_from_obj_or_json': False,
                    'full_protocol_complete': full,
                    'trajectory_parity_established': False,
                    'physical_validation': False}))

            original_effective = prepare.verify_effective_source_run
            original_wrapper = prepare.build_dynamic_resume_wrapper
            prepare.verify_effective_source_run = lambda *args: {'receipt_sha256': '0' * 64}
            prepare.build_dynamic_resume_wrapper = lambda _spec: []
            try:
                write_output()
                good = prepare.verify_resumed_output(spec_copy)
                self.assertEqual(good['segment_controls'], 59)
                write_output(accepted=58)
                with self.assertRaisesRegex(ValueError, 'scheduled end|wrong start/count'):
                    prepare.verify_resumed_output(spec_copy)
                write_output(restored=False)
                with self.assertRaisesRegex(ValueError, 'wrong start/count'):
                    prepare.verify_resumed_output(spec_copy)
            finally:
                prepare.verify_effective_source_run = original_effective
                prepare.build_dynamic_resume_wrapper = original_wrapper


if __name__ == '__main__':
    unittest.main(verbosity=2)
