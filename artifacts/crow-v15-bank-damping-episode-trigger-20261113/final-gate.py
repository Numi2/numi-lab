from pathlib import Path
import json,sys
p=Path('artifacts/crow-v15-bank-damping-episode-trigger-20261113')/sys.argv[1]
a=json.loads((p/'gate.json').read_text());b=json.loads((p/'bank-parity.json').read_text());c=json.loads((p/'readback.json').read_text());out={'existing_physical_and_signed_turn_gate':a['passed_identification'],'same_seed_full_reference_and_prefix_parity':b['matched'],'actual_gpu_parameter_readback':c['passed'],'passed':a['passed_identification'] and b['matched'] and c['passed'],'scope':'Used-seed observed damping-sensitivity identification only; no biological calibration, original-host promotion, frozen flight or navigation.'};(p/'final-gate.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
