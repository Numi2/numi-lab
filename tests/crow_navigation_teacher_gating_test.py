"""Regression tests for navigation teacher attribution at physical milestones."""
from dataclasses import replace
import numpy as np
from metalrobo.mlx_policy_learning import NativePolicyRollout, _TRANSITION_DTYPE

def fixture():
    n=6
    transitions=np.zeros(n,dtype=_TRANSITION_DTYPE)
    transitions["impact_event_flags"] = 1 << 22
    return NativePolicyRollout(
        id="navigation-gating",task_fingerprint=1,policy_fingerprint=2,policy_revision=1,
        environment_count=1,control_step_count=n,actor_observation_count=1,critic_observation_count=1,
        action_count=2,motion_feature_count=0,actor_observations=np.zeros(n,np.float32),
        critic_observations=np.zeros(n,np.float32),motion_features=np.empty(0,np.float32),
        latents=np.zeros(n*2,np.float32),old_log_probabilities=np.zeros(n,np.float32),
        old_values=np.zeros(n,np.float32),bootstrap_values=np.zeros(1,np.float32),
        transitions=transitions,outcomes={"root_height":np.full(n,.18,np.float32),
          "tilt":np.zeros(n,np.float32),"navigation_waypoints_reached":np.zeros(n,np.float32),
          "navigation_completion":np.zeros(n,np.float32)},
        teacher_actions=np.zeros(n*2,np.float32))
def weights(r,threshold=5,strict=True):
    return r.policy_batch(navigation_teacher_minimum_waypoint=threshold,
                          strict_navigation_teacher_gating=strict).teacher_weights
def main():
    r=fixture()
    assert np.all(weights(r)==0), "unqualified prefix received teacher weight"
    assert np.all(weights(r,strict=False)>0), "fixture did not reproduce legacy attribution"
    o={k:v.copy() for k,v in r.outcomes.items()}
    o["navigation_waypoints_reached"][2]=5
    o["navigation_completion"][2]=1
    t=r.transitions.copy();t["done"][2]=1;t["termination_reason"][2]=7
    succeeded=replace(r,outcomes=o,transitions=t)
    assert np.all(weights(succeeded)[:3]>0) and np.all(weights(succeeded)[3:]==0)
    o={k:v.copy() for k,v in r.outcomes.items()}
    o["navigation_waypoints_reached"][2:4]=1
    t=r.transitions.copy();t["done"][3]=1;t["termination_reason"][3]=3
    partial=replace(r,outcomes=o,transitions=t)
    assert np.all(weights(partial,1)[:3]>0) and np.all(weights(partial,1)[3:]==0)
    assert np.all(weights(partial,5)==0)
    t=partial.transitions.copy();t["done"][2]=1;t["termination_reason"][2]=3
    collision=replace(partial,transitions=t)
    assert np.all(weights(collision,1)==0), "contact at threshold counted as success"
    o={k:v for k,v in r.outcomes.items() if k!="navigation_completion"}
    try: weights(replace(r,outcomes=o))
    except ValueError: pass
    else: raise AssertionError("missing native navigation outcome accepted")
    print("crow_navigation_teacher_gating=ok")
if __name__=="__main__":main()
