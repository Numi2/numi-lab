#!/usr/bin/env python3
"""Extend the existing raw-log core assessment with source-identified joint cohorts."""
from run_native_joint_pair_015 import load, COHORTS
EXPECTED={
    "shoulder-supraspinatus":[("right shoulder elevation",36,35,1,0.,3.142),("left shoulder elevation",74,73,1,0.,3.142)],
    "elbow-flexors":[("right elbow",39,38,1,0.,2.269),("left elbow",77,76,1,0.,2.269)],
    "ankle-dorsiflexors":[("right ankle",109,108,1,-.698132,.523599),("left ankle",123,122,1,-.698132,.523599)],
    "mtp-flexors":[("right MTP",111,110,1,-.523599,.523599),("left MTP",125,124,1,-.523599,.523599)],
    "mtp-nonsaturated-flexors":[("right MTP",111,110,1,-.523599,.523599),("left MTP",125,124,1,-.523599,.523599)],
}
if __name__=="__main__":
    owner=load("joint_assessment_owner_011","assess_native_joint_pair_011.py")
    owner.EXPECTED.update({k:{"ids":v,"targets":EXPECTED[k]} for k,v in COHORTS.items()})
    owner.main()
