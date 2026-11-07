#!/usr/bin/env python3
"""Bounded manufactured-load transfer check, not dynamic qualification."""
import json,subprocess,sys
for n in (16,32):
 for rule in ('nodal','gaussian'):
  x=json.loads(subprocess.run([sys.argv[1],str(n),rule],capture_output=True,text=True,check=True).stdout)
  assert x['integrated_force_relative']<=.0025
  assert x['external_4_vs_8_relative']<1e-7 and x['reference_integrals_4_vs_8_relative']<1e-7
  if rule=='gaussian':assert x['max_local_residual_relative']<1e-5
print('Manufactured nonuniform n=16/32 force and independent-integration checks passed.')
