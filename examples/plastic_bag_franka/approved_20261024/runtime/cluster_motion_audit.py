"""Decompose saved coin motion into collective rigid motion and internal motion.

This is an observation of the trajectory. It applies no forces or corrections.
"""
import json
import sys
from pathlib import Path

import numpy as np


def measure(frame):
    m, radius, half = 0.01134, 0.015305, 0.001075
    p = np.asarray(frame['coins'], dtype=float)
    v = np.asarray(frame['coin_velocities'], dtype=float)
    w = np.asarray(frame['coin_angular_velocities'], dtype=float)
    q = np.asarray(frame['coin_orientations'], dtype=float)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, s = q.T
    axes = np.column_stack((2*(x*y-s*z), 1-2*(x*x+z*z), 2*(y*z+s*x)))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    axial = 0.5*m*radius**2
    radial = m*(0.25*radius**2+half**2/3)
    inertias = radial*np.eye(3)[None, :, :] + (axial-radial)*axes[:, :, None]*axes[:, None, :]
    r = p-p.mean(axis=0)
    center_velocity = v.mean(axis=0)
    relative = v-center_velocity
    cluster_inertia = np.sum(inertias+m*(np.sum(r*r, axis=1)[:, None, None]*np.eye(3)-r[:, :, None]*r[:, None, :]), axis=0)
    angular_momentum = np.sum(m*np.cross(r, relative)+np.einsum('nij,nj->ni', inertias, w), axis=0)
    cluster_w = np.linalg.solve(cluster_inertia, angular_momentum)
    residual_v = relative-np.cross(cluster_w, r)
    residual_w = w-cluster_w
    internal = 0.5*m*np.sum(residual_v**2)+0.5*np.einsum('ni,nij,nj->', residual_w, inertias, residual_w)
    total = 0.5*m*np.sum(v*v)+0.5*np.einsum('ni,nij,nj->', w, inertias, w)
    coherent = 0.5*len(p)*m*np.dot(center_velocity, center_velocity)+0.5*cluster_w@cluster_inertia@cluster_w
    return {
        'step': frame['step'], 'time_s': frame['time_s'],
        'total_coin_kinetic_J': float(total), 'internal_coin_kinetic_J': float(internal),
        'collective_coin_kinetic_J': float(coherent),
        'decomposition_error_J': float(total-internal-coherent),
        'maximum_internal_center_speed_m_s': float(np.linalg.norm(residual_v, axis=1).max()),
        'maximum_internal_angular_speed_rad_s': float(np.linalg.norm(residual_w, axis=1).max()),
        'center_of_mass_speed_m_s': float(np.linalg.norm(center_velocity)),
        'collective_angular_speed_rad_s': float(np.linalg.norm(cluster_w)),
    }


frames = json.loads(Path(sys.argv[1]).read_text())['frames']
rows = [measure(f) for f in frames]
hold = [r for r in rows if r['time_s'] >= 5.2]
tail = [r for r in rows if r['time_s'] >= 5.5]
result = {
    'states': len(rows), 'through_s': rows[-1]['time_s'], 'final': rows[-1],
    'maximum_decomposition_error_J': max(abs(r['decomposition_error_J']) for r in rows),
    'last_half_second': {
        'states': len(tail),
        **{label: max(r[key] for r in tail) for label, key in {
            'maximum_internal_coin_kinetic_J': 'internal_coin_kinetic_J',
            'maximum_internal_center_speed_m_s': 'maximum_internal_center_speed_m_s',
            'maximum_internal_angular_speed_rad_s': 'maximum_internal_angular_speed_rad_s',
            'maximum_total_coin_kinetic_J': 'total_coin_kinetic_J',
        }.items()},
    } if tail else None,
    'hold_maximum_internal_coin_kinetic_J': max(r['internal_coin_kinetic_J'] for r in hold) if hold else None,
    'method': 'FP64 mass and inertia weighted least-squares rigid velocity field for the ten coins. Residual motion includes both center motion and spin relative to this collective motion. Exact kinetic-energy decomposition for the ideal uniform-cylinder inertia model. No state is altered.',
    'limits': 'Small residual motion indicates coherent cluster motion; it does not establish correct contact forces or calibrated physical behavior.',
}
print(json.dumps(result, indent=2))
