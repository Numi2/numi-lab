#!/usr/bin/env python3
"""Verify a complete native ten-second unassisted quiet-standing episode.

These are engineering admission limits, not physiological qualification.
The verifier reads native evidence and never changes the physical state.
"""
import argparse
import json
import math
from pathlib import Path
from extract_human_brain_committed_transitions import manifest, number, progress, sensor, vector, sha256


def verify(run):
    steps, interval, hashes, manifest_sha = manifest(run)
    if steps != 10000 or interval != 1:
        raise ValueError('quiet standing requires all 10000 one-millisecond roots and motor audits')
    manifest_text = (run / 'launch.manifest').read_text()
    if 'native_timestep_microseconds=1000\n' not in manifest_text:
        raise ValueError('quiet standing requires the native one-millisecond physical clock')
    seen = {kind: 0 for kind in ['physical', 'sensor', 'motor']}
    first = None
    maximum_drift = maximum_speed = maximum_penetration = maximum_tilt = 0.0
    minimum_contacts = 10
    minimum_height = math.inf
    completed = terminal = False
    fingerprints = {}
    program_fingerprint = None
    terminal_joint = False
    with (run / 'launch.log').open() as stream:
        for line in stream:
            line = line.rstrip("\n")
            if line.startswith(('human_brain_sensor_audit=failed', 'human_brain_motor_audit=failed',
                                'human_brain_completion=failed', 'human_brain_joint_commit=rejected')):
                raise ValueError('native run contains a failed transaction')
            if line.startswith('human_brain_native_binding=configured '):
                program_fingerprint = number(line, 'locomotor_program_fingerprint', True)
                if program_fingerprint == 0 or number(line, 'timestep_microseconds', True) != 1000:
                    raise ValueError('native controller identity or physical clock is invalid')
            if line.startswith('human_brain_joint_commit=accepted '):
                step = number(line, 'step', True)
                if number(line, 'brain_generation', True) != step or not all(
                    f'{flag}=true' in line for flag in ['physical_motor_same_command',
                        'accepted_consequence_followup_command', 'same_native_owner_queue']):
                    raise ValueError('joint native ownership witness is invalid')
                terminal_joint |= step == steps
            kind = None
            if line.startswith('human_standing_progress=accepted '): kind = 'physical'
            elif line.startswith('human_brain_sensor_audit=accepted '): kind = 'sensor'
            elif line.startswith('human_brain_motor_audit=accepted '): kind = 'motor'
            if kind:
                step = number(line, 'step', True)
                if step != seen[kind] + 1: raise ValueError(f'{kind} roots are missing, repeated or reordered')
                seen[kind] = step
                if kind == 'sensor':
                    sensor(line, step)
                    fingerprints.setdefault(step, {})['sensor'] = (
                        number(line, 'physical_fingerprint', True), number(line, 'joint_commit_fingerprint', True))
                elif kind == 'motor':
                    if number(line, 'locomotor_program_fingerprint', True) != program_fingerprint:
                        raise ValueError('motor controller identity changed during the episode')
                    fingerprints.setdefault(step, {})['motor'] = (
                        number(line, 'physical_fingerprint', True), number(line, 'joint_commit_fingerprint', True))
                    if number(line, 'brain_generation', True) != step:
                        raise ValueError('motor generation differs from its physical root')
                    for field in ['excitation', 'activation']:
                        if any(x < 0 or x > 1 for x in vector(line, field, 416)):
                            raise ValueError('motor channel is outside its admitted range')
                else:
                    frame = progress(line)
                    xyz = frame['centerOfMassXYZM']
                    if first is None: first = xyz
                    maximum_drift = max(maximum_drift, math.hypot(xyz[0]-first[0], xyz[1]-first[1]))
                    minimum_height = min(minimum_height, xyz[2])
                    minimum_contacts = min(minimum_contacts, frame['contactCount'])
                    maximum_penetration = max(maximum_penetration, frame['penetrationM'])
                    maximum_speed = max(maximum_speed, number(line, 'root_linear_speed_m_s'))
                    q = vector(line, 'root_orientation_xyzw', 4)
                    norm = math.sqrt(sum(x*x for x in q))
                    if abs(norm-1) > 1e-4: raise ValueError('root orientation is not normalized')
                    maximum_tilt = max(maximum_tilt, 2*math.acos(min(1.0, abs(q[3])/norm)))
            if line.startswith('human_execution_stage=native_horizon_end'):
                completed |= number(line, 'stage_step', True) == steps
            terminal |= line.startswith('stand_terminal_state=')
    if any(count != steps for count in seen.values()) or not completed or not terminal or not terminal_joint:
        raise ValueError('native ten-second episode is incomplete')
    if any(v.get('sensor') != v.get('motor') for v in fingerprints.values()):
        raise ValueError('motor and sensory records belong to different joint transactions')
    checks = {'six_support_contacts': minimum_contacts >= 6,
              'penetration_at_most_five_micrometres': maximum_penetration <= 5e-6,
              'root_speed_at_most_point_one_m_per_s': maximum_speed <= .1,
              'com_drift_at_most_two_centimetres': maximum_drift <= .02,
              'com_height_loss_at_most_five_centimetres': minimum_height >= first[2]-.05,
              'root_rotation_at_most_point_zero_five_radians': maximum_tilt <= .05}
    return {'schema': 'numi.human.quiet-standing.verification.v1',
            'evidence_boundary': 'native simulated engineering stance; no physiological or learned-skill qualification',
            'steps': steps, 'complete': True, 'unassisted': True, 'sensor_coverage': True,
            'maximum_com_drift_m': maximum_drift, 'maximum_root_speed_m_s': maximum_speed,
            'minimum_contacts': minimum_contacts, 'maximum_penetration_m': maximum_penetration,
            'minimum_com_height_m': minimum_height, 'maximum_root_rotation_rad': maximum_tilt,
            'checks': checks, 'quiet_standing': all(checks.values()),
            'launch_manifest_sha256': manifest_sha, 'launch_log_sha256': sha256(run / 'launch.log'),
            'locomotor_program_fingerprint': program_fingerprint, 'source_sha256': hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['quiet_standing'] else 1)

if __name__ == '__main__': main()
