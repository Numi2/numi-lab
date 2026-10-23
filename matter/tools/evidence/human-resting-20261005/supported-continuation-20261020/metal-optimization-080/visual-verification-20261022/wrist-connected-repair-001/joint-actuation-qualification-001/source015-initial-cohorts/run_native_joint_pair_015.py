#!/usr/bin/env python3
"""Matched, hash-pinned native core trials for shoulder, elbow, ankle and MTP owners.

These test causal selected-muscle/tendon/articulated response, not physical skin
or whole-body joint motion. No held joint targets or assistance are introduced.
"""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parent
COHORTS={
    "shoulder-supraspinatus":[213,276],
    "elbow-flexors":[232,233,295,296],
    "ankle-dorsiflexors":[344,345,371,384,385,411],
    "mtp-flexors":[346,347,386,387],
}
def load(name, filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
if __name__=="__main__":
    owner=load("joint_trial_owner_011","run_native_joint_pair_011.py")
    owner.COHORTS.update(COHORTS)
    owner.main()
