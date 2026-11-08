"""Compare retained different discretizations; no equivalence is assumed."""
import json
for n in (4,8):
 a=json.load(open(f"dynamics-n{n}.json"));o=json.load(open(f"tensor-n{n}-coarse.json"))
 reflected=a["full"];bins=reflected["force_bins_N"];other=o["force_bins_N"][4:]
 before=[x for x in o['trace'] if x[0]<=.004500001]
 assert before and abs(before[-1][0]-.0045)<1e-9
 assert max(abs(x[1]) for x in before)==0. and before[-1][9]==0.
 assert len(bins)==len(other)==6
 peak=max(abs(x[1]) for x in reflected["trace"])
 impulse=reflected["trace"][-1][5]
 print(json.dumps({"n":n,"prior_contact_force_and_impulse_zero":True,"reflected_peak_N":peak,"binned_force_error_relative_reflected_peak":max(abs(x-y) for x,y in zip(bins,other))/peak,"impulse_error_relative_tensor":abs(impulse-o["impulse_Ns"])/abs(o["impulse_Ns"]),"reflected_bins_N":bins,"tensor_bins_N":other},sort_keys=True))
