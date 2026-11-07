#!/usr/bin/env python3
"""Independent FP64 point-mass analogy for interpreting fixed-area FEM traces.

This is not the FEM fixture or a validation reference for it. It tests whether
implicit integration of the same pressure law can itself change release timing.
"""
import json, math, sys

mass, area, width, stiffness, gravity = .0088, .0004, .0001, 1e16, 9.81
initial_gap, duration = .0002, .01
steps = (.0005, .00025, .000125, .0000625)

def pressure(g):
    if g >= width:
        return 0.0
    if g <= 0:
        raise ValueError('nonpositive gap')
    q = g*g - width*width
    return 2*g*stiffness*(2*q*math.log(g*g/(width*width)) + q*q/(g*g))

def advance(g, v, h):
    predicted = g + h*v - gravity*h*h
    lo, hi = 1e-14, max(width, predicted) + width
    def residual(z):
        return z - predicted - h*h*area*pressure(z)/mass
    assert residual(lo) < 0 and residual(hi) > 0
    for _ in range(100):
        mid = (lo + hi)/2
        if residual(mid) > 0:
            hi = mid
        else:
            lo = mid
    next_gap = (lo + hi)/2
    return next_gap, (next_gap - g)/h

lo, hi = 1e-14, width
for _ in range(100):
    mid = (lo + hi)/2
    if area*pressure(mid) > mass*gravity:
        lo = mid
    else:
        hi = mid
equilibrium = (lo + hi)/2
delta = 1e-10
contact_tangent = -area*(pressure(equilibrium+delta)-pressure(equilibrium-delta))/(2*delta)
omega = math.sqrt(contact_tangent/mass)

results=[]
for h in steps:
    g, v, first, peak, total = initial_gap, 0., None, 0., 0.
    episodes=[]; bins=[0.]*10
    for i in range(round(duration/h)):
        g, v = advance(g, v, h)
        f = area*pressure(g)
        total += f*h; peak=max(peak, f)
        bins[int((i+.5)*h/.001)] += f*h/.001
        if f > 0. and first is None:
            first=i*h
        if f == 0. and first is not None:
            episodes.append([first, i*h]); first=None
    if first is not None:
        episodes.append([first, duration])
    # BE's exact per-step amplification of a linear undamped oscillator.
    per_period=(1+(omega*h)**2)**(-math.pi/(omega*h))
    results.append(dict(h=h, peak_normal_force_N=peak, impulse_Ns=total,
                        contact_episodes_s=episodes, force_bins_N=bins,
                        linear_oscillator_amplitude_retained_per_period=per_period))
def explicit_rk4(h):
    # Reference only for the point-mass analogy; state is gap, speed, impulse.
    state=(initial_gap, 0., 0.); bins=[0.]*10
    def rhs(y):
        f=area*pressure(y[0])
        return y[1], f/mass-gravity, f
    for i in range(round(duration/h)):
        k1=rhs(state)
        k2=rhs(tuple(a+h*b/2 for a,b in zip(state,k1)))
        k3=rhs(tuple(a+h*b/2 for a,b in zip(state,k2)))
        k4=rhs(tuple(a+h*b for a,b in zip(state,k3)))
        nxt=tuple(a+h*(b+2*c+2*d+e)/6 for a,b,c,d,e in zip(state,k1,k2,k3,k4))
        bins[int((i+.5)*h/.001)]+=(nxt[2]-state[2])/.001
        state=nxt
    return dict(h=h, final_gap_m=state[0], final_velocity_m_per_s=state[1],
                impulse_Ns=state[2], force_bins_N=bins)
rk_coarse, rk_fine=explicit_rk4(1e-6), explicit_rk4(5e-7)
rk_force_change=max(abs(a-b) for a,b in zip(rk_coarse['force_bins_N'],rk_fine['force_bins_N']))/max(rk_fine['force_bins_N'])
rk_impulse_change=abs(rk_coarse['impulse_Ns']-rk_fine['impulse_Ns'])/abs(rk_fine['impulse_Ns'])
assert rk_force_change<1e-5 and rk_impulse_change<1e-5
continuous_freefall_onset=math.sqrt(2*(initial_gap-width)/gravity)
assert abs(continuous_freefall_onset - .004515236409857309)<1e-12
out=dict(schema='numi.contact.area-timestep-diagnosis.v1',
    scope='independent FP64 point-mass analogy; not the 3D FEM fixture',
    equilibrium_gap_m=equilibrium, linearized_contact_tangent_N_per_m=contact_tangent,
    linearized_frequency_rad_per_s=omega, linearized_period_s=2*math.pi/omega,
    exact_freefall_onset_s=continuous_freefall_onset, simulations=results,
    point_mass_rk4=[rk_coarse,rk_fine], point_mass_rk4_force_bin_change=rk_force_change,
    point_mass_rk4_impulse_change=rk_impulse_change,
    limitations=['uniform gap and lumped mass omit FEM deformation and internal modes',
                 'does not establish a defect or force accuracy in Matter'])
json.dump(out,sys.stdout,indent=2);print()
