#!/usr/bin/env python3
import json,struct,pathlib,hashlib
B=pathlib.Path('/Users/n/cardboard-implementation-20261014-host-macmini')
F=[('sutureProxyEdges', 4), ('particles', 160), ('femNodes', 80), ('femFields', 32), ('vascularState', 16), ('vascularClock', 16), ('femTopologyNodes', 16), ('femTopologyTetrahedra', 96), ('cohesiveFaces', 384), ('punctureChannels', 48), ('topologyStates', 48), ('statuses', 48), ('solverCertificates', 64), ('mpmActiveNodeIndices', 4), ('mpmNodeToActive', 4), ('mpmActiveNodeCounts', 4), ('rigidGeneralizedCandidate', 4), ('learnedWeights', 4), ('adaptive', 160), ('schedulers', 80), ('reactions', 32), ('rigidStates', 144), ('contactSamples', 160), ('deformableContactFailures', 384), ('contactHistories', 16), ('humanSupportHistories', 16), ('humanSupportConsequences', 64), ('deformableContactHistories', 48), ('particleMaterialState', 4), ('femMaterialState', 4), ('identification', 48), ('environmentParameters', 4)]
M=['centerAndRadius', 'extent', 'orientation', 'linearVelocityAndInverseMass', 'angularVelocity', 'bodyCenter', 'inverseInertiaRow0', 'inverseInertiaRow1', 'inverseInertiaRow2']
def read(p,c):
 d=p.read_bytes();o=4+struct.unpack_from('<I',d,0)[0]+8+8+16+1+8+28
 for n,s in F:
  k,=struct.unpack_from('<Q',d,o);o+=8
  if k!=c[n]:raise ValueError((n,k,c[n]))
  if n=='rigidStates':
   raw=d[o:o+k*s];v=struct.unpack('<72f',raw)
   return {'offset':o,'sha256':hashlib.sha256(raw).hexdigest(),'states':[{M[i]:list(v[j*36+i*4:j*36+i*4+4]) for i in range(9)} for j in range(2)]}
  o+=k*s
 raise ValueError('rigidStates absent')
for case in ('contact-trial-observer-off-583','contact-trial-observer-on-584'):
 p=B/'native'/case/'accepted-state-output/dense';t=json.load(open(p/'transaction-000000.json'))
 print(case,json.dumps({'before':read(p/'transaction-000000-before.bin',t['before']['vector_counts']),'after':read(p/'transaction-000000-after.bin',t['after']['vector_counts'])},sort_keys=True))
