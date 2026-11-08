import copy,importlib.util,json,sys
from pathlib import Path
root=Path("/Users/numi/policy-evidence-integrity-20261108")
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=load("old",root/"before-policy_selection.py")
new=load("new",Path("/Users/numi/numi-lab-tracker-step-rollback-20261108/python/metalrobo/policy_selection.py"))
a=json.loads((root/"native-incumbent.json").read_text());b=json.loads((root/"native-candidate.json").read_text())
for label,candidate in (("retained_native",b),("synthetic_reward_positive",{**copy.deepcopy(b),"mean_task_reward":.2,"mean_reward":.2})):
 print(json.dumps({"case":label,"old":old.compare_evidence(a,candidate),"new":new.compare_evidence(a,candidate)},sort_keys=True))
